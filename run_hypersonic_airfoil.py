"""Hypersonic Airfoil Simulation Runner (Space Shuttle & NASA X-43A).

Executes high-order discontinuous Galerkin with subcell FV shock capturing,
WALE Large Eddy Simulation, and real-time aerodynamic force integration (Cl, Cd, Cm).
"""

import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from hybrid import (
    AerodynamicForceCalculator,
    AerodynamicHistoryTracker,
    FarfieldWallBoundary,
    HybridCompressibleSolver,
    HybridMOODSubcellLimiter,
    HybridModalShockLimiter,
    HybridNavierStokesOperator,
    HybridPositivityLimiter,
    HybridSSPRK3,
    HybridTroubledCellSensor,
)
from hybrid.output import (
    _midplane_triangles,
    compute_schlieren,
    shock_colormap,
)


def assert_physical_state(states, solver, context):
    """Refuse to turn a failed solver state into a plausible-looking file."""
    for name, state in states.items():
        if not bool(jnp.all(jnp.isfinite(state))):
            bad = int(jnp.sum(~jnp.isfinite(state)))
            raise FloatingPointError(
                f"{context}: {name} contains {bad} non-finite values"
            )
        primitive = solver.conserved_to_primitive(state)
        density_min = float(jnp.min(primitive[..., 0]))
        pressure_min = float(jnp.min(primitive[..., 4]))
        if density_min <= 0.0 or pressure_min <= 0.0:
            raise FloatingPointError(
                f"{context}: {name} is nonphysical "
                f"(rho_min={density_min}, p_min={pressure_min})"
            )


def render_airfoil_snapshot(
    operator,
    states,
    aero_result,
    step,
    current_time,
    output_path,
    alpha_deg=35.0,
    cl_wind=0.0,
    cd_wind=0.0,
    extent=(-0.5, 3.5, -1.8, 1.8),
):
    """Render live 4-panel diagnostic figure: Mach, Pressure, Vorticity, Schlieren."""
    xy, _, W = _midplane_triangles(operator, states, plane_z=0.0)
    rounded = np.round(xy, decimals=5)
    unique_xy, inverse = np.unique(rounded, axis=0, return_inverse=True)
    counts = np.bincount(inverse)

    unique_W = np.zeros((len(unique_xy), W.shape[1]), dtype=float)
    for v in range(W.shape[1]):
        unique_W[:, v] = np.bincount(inverse, weights=W[:, v]) / counts

    xmin, xmax, ymin, ymax = extent
    nx, ny = 1000, 600
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)

    num_vars = unique_W.shape[-1]
    resampled = np.zeros((ny, nx, num_vars), dtype=float)
    for v in range(num_vars):
        val = griddata(unique_xy, unique_W[:, v], (GX, GY), method="linear")
        nan_pts = np.isnan(val)
        if np.any(nan_pts):
            val_near = griddata(unique_xy, unique_W[:, v], (GX[nan_pts], GY[nan_pts]), method="nearest")
            val[nan_pts] = val_near
        resampled[:, :, v] = val

    rho = resampled[:, :, 0]
    u = resampled[:, :, 1]
    v = resampled[:, :, 2]
    p = resampled[:, :, 4]
    speed = np.sqrt(u**2 + v**2)
    sound = np.sqrt(1.4 * np.maximum(p, 1e-6) / np.maximum(rho, 1e-6))
    mach = speed / np.maximum(sound, 1e-6)

    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])
    mask = np.zeros((ny, nx), dtype=bool)
    schlieren = compute_schlieren(rho, dx, dy, mask, sigma=3.0, beta=0.1)

    # Vorticity magnitude: |dv/dx - du/dy|
    dudy, dudx = np.gradient(u, dy, dx)
    dvdy, dvdx = np.gradient(v, dy, dx)
    vorticity = np.abs(dvdx - dudy)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), facecolor="#030712")

    # Panel 1: Mach Number
    ax1 = axes[0, 0]
    ax1.set_facecolor("#030712")
    im1 = ax1.imshow(mach, origin="lower", extent=extent, cmap="turbo", vmin=0.0, vmax=5.5)
    ax1.contour(gx, gy, mach, levels=[1.0], colors="white", linewidths=1.5, linestyles="--")
    ax1.set_title(
        f"Mach Number Field (White dashed: Sonic Line M=1)\n"
        f"Step {step} | t = {current_time:.5f} s | α = {alpha_deg:.1f}° | "
        f"Lift $C_L$={cl_wind:.3f}, Drag $C_D$={cd_wind:.3f}",
        color="white", fontsize=11, fontweight="bold", pad=8
    )
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors="#94a3b8")

    # Panel 2: Static Temperature Field (Aerothermal Heating on Windward Tiles)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#030712")
    T_kelvin = (p / np.maximum(rho, 1e-6)) * 300.0
    T_max = float(np.percentile(T_kelvin, 99.8))
    im2 = ax2.imshow(T_kelvin, origin="lower", extent=extent, cmap="inferno", vmin=250.0, vmax=max(T_max, 1800.0))
    ax2.set_title(f"Static Temperature $T$ [K] (Windward Aerodynamic Heating, Max: {T_max:.0f} K)", color="white", fontsize=11, fontweight="bold", pad=8)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="#94a3b8")

    # Panel 3: Vorticity Magnitude (Turbulent Eddies & Shear Layer)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#030712")
    vort_max = float(np.percentile(vorticity, 99.5))
    im3 = ax3.imshow(vorticity, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=max(vort_max, 10.0))
    ax3.set_title("Vorticity Magnitude $|\\nabla \\times \\vec{u}|$ (Separated Shear Layer & WALE Eddies)", color="white", fontsize=11, fontweight="bold", pad=8)
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(colors="#94a3b8")

    # Panel 4: Numerical Schlieren |grad rho|
    ax4 = axes[1, 1]
    ax4.set_facecolor("#030712")
    im4 = ax4.imshow(schlieren, origin="lower", extent=extent, cmap=shock_colormap(), vmin=0.0, vmax=1.0)
    ax4.set_title("Numerical Schlieren $|\\nabla\\rho|$ (Bow Shock, Expansion Fan & Separated Wake)", color="white", fontsize=11, fontweight="bold", pad=8)
    cb4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.ax.tick_params(colors="#94a3b8")

    for ax in axes.flat:
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.set_xlabel("x / c", color="#94a3b8", fontsize=10)
        ax.set_ylabel("y / c", color="#94a3b8", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def run_hypersonic_airfoil(args):
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)

    gamma = 1.4
    density = 1.0
    pressure = 1.0
    sound_speed = (gamma * pressure / density) ** 0.5
    velocity = args.mach * sound_speed
    viscosity = density * velocity / args.reynolds

    alpha_rad = np.radians(args.alpha)
    u_inf = float(velocity * np.cos(alpha_rad))
    v_inf = float(velocity * np.sin(alpha_rad))

    print("=" * 76)
    print("HYPERSONIC AIRFOIL NAVIER-STOKES / WALE LES SIMULATION")
    print(f"  Mesh:              {args.mesh}")
    print(f"  Mach Number:       M = {args.mach:.2f}")
    print(f"  Angle of Attack:   α = {args.alpha:.1f}°")
    print(f"  Reynolds Number:   Re = {args.reynolds:.1e}")
    print(f"  Inflow Velocity:   u_inf = {u_inf:.3f}, v_inf = {v_inf:.3f} (V = {velocity:.3f})")
    print(f"  DG Polynomial:     P = {args.polynomial_degree}")
    print(f"  WALE LES:          {'ENABLED' if args.wale else 'DISABLED'}")
    print(f"  Subcell FV Shock:  {'ENABLED (WENO)' if args.subcell_fv else 'DISABLED'}")
    print(f"  Output Directory:  {output}")
    print("=" * 76)

    solver = HybridCompressibleSolver(
        args.mesh,
        species_names=(),
        gamma=gamma,
        polynomial_degree=args.polynomial_degree,
        inviscid_flux=args.inviscid_flux,
        dynamic_viscosity=viscosity,
        use_wale=args.wale,
    )
    operator = HybridNavierStokesOperator(solver)
    farfield = jnp.asarray([density, u_inf, v_inf, 0.0, pressure])
    boundary = FarfieldWallBoundary(
        solver,
        farfield,
        inlet_names=("inlet",),
        outlet_names=("outlet",),
        wall_names=("walls", "airfoil_wall", "symmetry"),
        no_slip_names=("airfoil_wall",),
        slip_names=("walls", "symmetry"),
        wall_temperature=args.wall_temperature,
    )
    q_inf = 0.5 * density * velocity**2
    aero_calc = AerodynamicForceCalculator(
        operator,
        boundary_state=boundary,
        wall_tag_names=("airfoil_wall",),
        reference_area=args.reference_area,
        reference_length=args.reference_length,
        moment_center=tuple(args.moment_center),
    )
    aero_tracker = AerodynamicHistoryTracker(output / "aerodynamics.csv")

    positivity = HybridPositivityLimiter(
        operator,
        density_floor=args.density_floor,
        pressure_floor=args.pressure_floor,
        relative_floor=args.relative_floor,
    )
    if args.subcell_fv:
        limiter = HybridMOODSubcellLimiter(
            operator,
            positivity=positivity,
            reconstruction=args.reconstruction,
        )
    elif args.shock_limiter:
        limiter = HybridModalShockLimiter(operator, positivity=positivity)
    else:
        limiter = positivity

    sensor = HybridTroubledCellSensor(operator)
    integrator = HybridSSPRK3(
        operator,
        cfl=args.cfl,
        limiter=limiter,
        density_floor=args.density_floor,
        pressure_floor=args.pressure_floor,
        maximum_rejections=args.maximum_rejections,
    )

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from checkpoint: {args.resume}")
        data = np.load(args.resume)
        states = {
            block.element_type: jnp.asarray(data[f"U_{block.element_type}"])
            for block in solver.mesh.element_blocks
        }
        current_time = float(data.get("time", 0.0))
        step = int(data.get("step", 0))
    else:
        print("Initializing free-stream uniform state...")
        states = solver.uniform_state_blocks(
            density,
            jnp.asarray([u_inf, v_inf, 0.0]),
            pressure,
        )
        current_time = 0.0
        step = 0

    print("Compiling JAX SSPRK3 step and CFL estimators (XLA JIT)...")
    t0 = time.perf_counter()
    compiled_dt = jax.jit(lambda s: integrator.stable_dt(s, boundary))
    compiled_step = lambda s, dt: integrator.checked_step(s, dt, boundary)

    dummy_dt = float(compiled_dt(states))
    dummy_state, _, _, _ = compiled_step(states, dummy_dt)
    for v in dummy_state.values():
        jax.block_until_ready(v)
    print(f"JIT Compilation completed in {time.perf_counter() - t0:.2f} s")

    assert_physical_state(states, solver, "initial state")
    start = time.perf_counter()
    next_progress = 1
    total_rejections = 0
    consecutive_rejections = 0
    retry_scale = 1.0

    while current_time < args.target_time and step < args.maximum_steps:
        output_time = min(
            args.target_time,
            next_progress * args.target_time / args.progress_updates,
        )
        substep = 0
        while current_time < output_time and step < args.maximum_steps:
            cfl_dt = float(compiled_dt(states))
            if np.isnan(cfl_dt) or cfl_dt <= 0.0:
                raise FloatingPointError("CFL evaluation produced invalid dt")

            dt_step = min(
                cfl_dt * retry_scale,
                output_time - current_time,
            )

            candidate_state, valid, rho_att, p_att = compiled_step(states, dt_step)
            valid_val = bool(jax.device_get(valid))
            if valid_val:
                states = candidate_state
                current_time += dt_step
                step += 1
                substep += 1
                consecutive_rejections = 0
                retry_scale = min(1.0, retry_scale * 1.05)
            else:
                total_rejections += 1
                consecutive_rejections += 1
                retry_scale = max(0.1, retry_scale * 0.5)
                if consecutive_rejections > args.maximum_rejections:
                    raise RuntimeError(f"Step {step}: exceeded maximum consecutive step rejections")

        aero_result = aero_calc.compute(states, q_inf, step=step, current_time=current_time)
        aero_tracker.record(aero_result)

        # Wind-axis projection
        cd_wind = aero_result.CD * np.cos(alpha_rad) + aero_result.CL * np.sin(alpha_rad)
        cl_wind = -aero_result.CD * np.sin(alpha_rad) + aero_result.CL * np.cos(alpha_rad)

        _, troubled_masks = sensor(states)
        troubled = {
            name: int(jnp.sum(troubled_masks[name]))
            for name in states
        }
        troubled_str = " ".join(f"{k}={v}" for k, v in troubled.items())

        pct = 100.0 * step / args.maximum_steps
        elapsed = time.perf_counter() - start
        rate = step / max(elapsed, 1e-6)
        eta = (args.maximum_steps - step) / max(rate, 1e-6)

        print(
            f"progress={pct:6.2f}% step={step:4d}/{args.maximum_steps} t={current_time:.6f}s "
            f"Lift_wind={cl_wind:+.4f} Drag_wind={cd_wind:.4f} (Body: Cx={aero_result.CD:.4f}, Cy={aero_result.CL:+.4f}) "
            f"troubled=[{troubled_str}] dt={cfl_dt:.2e} rate={rate:.1f}steps/s eta={eta:.0f}s",
            flush=True,
        )

        if step % args.checkpoint_interval == 0 or step >= args.maximum_steps:
            save_dict = {f"U_{name}": np.asarray(states[name]) for name in states}
            save_dict["time"] = current_time
            save_dict["step"] = step
            np.savez(output / "checkpoint_latest.npz", **save_dict)

        if step % args.render_interval == 0 or step >= args.maximum_steps:
            try:
                render_path = output / f"snapshot_step_{step:04d}.png"
                render_airfoil_snapshot(
                    operator, states, aero_result, step, current_time, render_path,
                    alpha_deg=args.alpha, cl_wind=cl_wind, cd_wind=cd_wind,
                )
                render_airfoil_snapshot(
                    operator, states, aero_result, step, current_time, output / "live_snapshot.png",
                    alpha_deg=args.alpha, cl_wind=cl_wind, cd_wind=cd_wind,
                )
            except Exception as e:
                print(f"Warning: snapshot render failed: {e}")

        next_progress += 1

    print(f"Simulation completed: {step} steps in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hypersonic Airfoil Simulation Runner")
    parser.add_argument("--mesh", default="mesh/shuttle_cgrid.msh")
    parser.add_argument("--mach", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=35.0, help="Angle of attack in degrees")
    parser.add_argument("--reynolds", type=float, default=100000.0)
    parser.add_argument("--polynomial-degree", type=int, default=2)
    parser.add_argument("--inviscid-flux", default="hllc", choices=["hllc", "rusanov"])
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--target-time", type=float, default=1.0)
    parser.add_argument("--maximum-steps", type=int, default=2500)
    parser.add_argument("--progress-updates", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--render-interval", type=int, default=250)
    parser.add_argument("--wale", action="store_true", default=True)
    parser.add_argument("--subcell-fv", action="store_true", default=True)
    parser.add_argument("--shock-limiter", action="store_true", default=True)
    parser.add_argument("--reconstruction", default="weno", choices=["donor", "muscl", "weno"])
    parser.add_argument("--density-floor", type=float, default=1e-4)
    parser.add_argument("--pressure-floor", type=float, default=1e-4)
    parser.add_argument("--relative-floor", type=float, default=1e-3)
    parser.add_argument("--maximum-rejections", type=int, default=10)
    parser.add_argument("--reference-area", type=float, default=0.1)
    parser.add_argument("--reference-length", type=float, default=1.0)
    parser.add_argument("--moment-center", nargs=3, type=float, default=[0.25, 0.0, 0.0])
    parser.add_argument("--output-directory", default="outputs/shuttle_a35_m5")
    parser.add_argument("--wall-temperature", type=float, default=1.0, help="Cold wall temperature (1.0 = 300 K)")
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()
    run_hypersonic_airfoil(args)
