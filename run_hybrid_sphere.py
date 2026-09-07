"""Run the curved P2 prism/tetra Navier--Stokes sphere test."""

import argparse
from pathlib import Path
import time

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from hybrid import (
    FarfieldWallBoundary,
    HybridCompressibleSolver,
    HybridModalShockLimiter,
    HybridMOODSubcellLimiter,
    HybridNavierStokesOperator,
    HybridPositivityLimiter,
    HybridSSPRK3,
    HybridTroubledCellSensor,
    save_cinematic_schlieren,
    save_midplane_heatmap,
    write_tessellated_vtu,
)
from hybrid.output import write_pvd


def save_restart(path, states, current_time, step):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        time=np.asarray(current_time),
        step=np.asarray(step),
        **{f"U_{name}": np.asarray(value) for name, value in states.items()},
    )
    temporary.replace(path)


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


def run(args):
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.compilation_cache)
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
    gamma = 1.4
    density = 1.0
    pressure = 1.0
    sound_speed = (gamma * pressure / density) ** 0.5
    velocity = args.mach * sound_speed
    viscosity = density * velocity / args.reynolds

    solver = HybridCompressibleSolver(
        args.mesh,
        gamma=gamma,
        polynomial_degree=args.polynomial_degree,
        inviscid_flux=args.inviscid_flux,
        dynamic_viscosity=viscosity,
        use_wale=args.wale,
    )
    operator = HybridNavierStokesOperator(solver)
    farfield = jnp.asarray([density, velocity, 0.0, 0.0, pressure])
    boundary = FarfieldWallBoundary(solver, farfield)
    if args.resume:
        restart = np.load(args.resume)
        states = {
            block.element_type: jnp.asarray(restart[f"U_{block.element_type}"])
            for block in solver.mesh.element_blocks
        }
        current_time = float(restart["time"])
        step = int(restart["step"]) if "step" in restart else 0
        print(f"Resuming {args.resume} from t={current_time:.8f}, step={step}")
    else:
        states = solver.uniform_state_blocks(
            density, jnp.asarray([velocity, 0.0, 0.0]), pressure
        )
        current_time = 0.0
        step = 0
    run_start_time = current_time
    positivity = HybridPositivityLimiter(
        operator,
        density_floor=args.density_floor,
        pressure_floor=args.pressure_floor,
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

    # Keep CFL and RK executables separate on 6 GiB GPUs.  Fusing the complete
    # adaptive interval reduces dispatch overhead but creates a much larger
    # CUDA module and workspace than the production mesh can reliably load.
    compiled_dt = jax.jit(lambda state: integrator.stable_dt(state, boundary))
    compiled_step = lambda state, dt: integrator.checked_step(
        state, dt, boundary
    )
    counts = ", ".join(
        f"{block.element_type}={len(block.connectivity)}"
        for block in solver.mesh.element_blocks
    )
    print(
        f"mesh={args.mesh} p=2 {counts} Mach={args.mach} "
        f"Re_D={args.reynolds:g} mu={viscosity:.6e} WALE={args.wale} "
        f"flux={args.inviscid_flux} subcell_FV={args.subcell_fv}"
    )
    print("Compiling on the first step...", flush=True)

    frames = []
    assert_physical_state(states, solver, "initial/restart state")
    if args.vtk:
        frame = write_tessellated_vtu(output / "sphere_0000.vtu", operator, states)
        frames.append((current_time, frame))
    start = time.perf_counter()
    next_progress = int(
        current_time / args.target_time * args.progress_updates
    ) + 1

    total_rejections = 0
    consecutive_rejections = 0
    retry_scale = 1.0
    minimum_dt_value = float("inf")
    while current_time < args.target_time and step < args.maximum_steps:
        output_time = min(
            args.target_time,
            next_progress * args.target_time / args.progress_updates,
        )
        cfl_dt = float(compiled_dt(states))
        dt = min(retry_scale * cfl_dt, output_time - current_time)
        if not np.isfinite(dt) or dt <= 0.0:
            raise FloatingPointError(f"invalid timestep at step {step}: {dt}")
        candidate, valid, attempted_rho, attempted_pressure = compiled_step(
            states, jnp.asarray(dt)
        )
        valid_value, rho_attempt, pressure_attempt = jax.device_get(
            (valid, attempted_rho, attempted_pressure)
        )
        if bool(valid_value):
            states = candidate
            current_time += dt
            step += 1
            consecutive_rejections = 0
            retry_scale = min(1.0, 1.25 * retry_scale)
            minimum_dt_value = min(minimum_dt_value, dt)
            if output_time - current_time <= (
                32.0 * np.finfo(np.float32).eps * max(abs(output_time), 1.0e-12)
            ):
                current_time = output_time
        else:
            del candidate
            total_rejections += 1
            consecutive_rejections += 1
            retry_scale *= 0.5
            if consecutive_rejections >= args.maximum_rejections:
                raise FloatingPointError(
                    "Unable to find an admissible SSP-RK3 step: "
                    f"t={current_time:.8g}, rejected={total_rejections}, "
                    f"rho_min={float(rho_attempt):.6g}, "
                    f"p_min={float(pressure_attempt):.6g}. "
                    "The state remains at the last accepted step."
                )
            continue
        fraction = current_time / args.target_time
        run_fraction = (
            (current_time - run_start_time)
            / max(args.target_time - run_start_time, 1.0e-30)
        )
        if fraction * args.progress_updates >= next_progress or current_time >= args.target_time:
            W = {
                name: solver.conserved_to_primitive(value)
                for name, value in states.items()
            }
            rho_min = min(float(jnp.min(value[...,0])) for value in W.values())
            rho_max = max(float(jnp.max(value[...,0])) for value in W.values())
            p_min = min(float(jnp.min(value[...,4])) for value in W.values())
            p_max = max(float(jnp.max(value[...,4])) for value in W.values())
            _, flags = sensor(states)
            troubled = sum(int(jnp.sum(value)) for value in flags.values())
            elapsed = time.perf_counter() - start
            eta = elapsed * (1.0-run_fraction) / max(run_fraction, 1.0e-12)
            print(
                f"progress={100*fraction:6.2f}% time={current_time:.6f} "
                f"steps={step} rho=[{rho_min:.6g},{rho_max:.6g}] "
                f"p=[{p_min:.6g},{p_max:.6g}] troubled={troubled} "
                f"rejected={total_rejections} dt_min={minimum_dt_value:.3e} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
                , flush=True
            )
            save_restart(
                output / "checkpoint_latest.npz", states, current_time, step
            )
            if args.vtk:
                index = len(frames)
                frame = write_tessellated_vtu(
                    output / f"sphere_{index:04d}.vtu", operator, states
                )
                frames.append((current_time, frame))
                write_pvd(output / "sphere_animation.pvd", frames)
            next_progress += 1

    assert_physical_state(states, solver, "final state")
    if args.vtk and (
        not frames
        or abs(frames[-1][0] - current_time) > 1.0e-14
    ):
        index = len(frames)
        frame = write_tessellated_vtu(
            output / f"sphere_{index:04d}.vtu", operator, states
        )
        frames.append((current_time, frame))
        write_pvd(output / "sphere_animation.pvd", frames)
    save_restart(output / "final_state.npz", states, current_time, step)
    image = save_midplane_heatmap(
        output / "sphere_final.png",
        operator,
        states,
        title=f"Hybrid P{operator.solver.polynomial_degree} DG sphere, Mach {args.mach}, t={current_time:.6f}",
    )
    cinematic_image = save_cinematic_schlieren(
        output / "sphere_cinematic.png",
        operator,
        states,
        sim_time=current_time,
        step=step,
        mach=args.mach,
        reynolds=args.reynolds,
    )
    if args.vtk:
        write_pvd(output / "sphere_animation.pvd", frames)
    print(f"Completed {step} steps at t={current_time:.8f}")
    print(f"Saved state: {output / 'final_state.npz'}")
    print(f"Saved heatmap: {image}")
    print(f"Saved cinematic Schlieren: {cinematic_image}")
    if args.vtk:
        print(f"Open {output / 'sphere_animation.pvd'} in ParaView")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default="mesh/sphere_tunnel_hybrid.msh")
    parser.add_argument(
        "-p", "--polynomial-degree",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="Polynomial approximation degree (default: 2)",
    )
    parser.add_argument("--target-time", type=float, default=0.002)
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--reynolds", type=float, default=100000.0)
    parser.add_argument("--cfl", type=float, default=0.08)
    parser.add_argument(
        "--inviscid-flux",
        choices=("hllc", "rusanov"),
        default="hllc",
    )
    parser.add_argument("--wale", action="store_true")
    parser.add_argument("--shock-limiter", action="store_true")
    parser.add_argument("--subcell-fv", action="store_true")
    parser.add_argument(
        "--reconstruction",
        choices=("constant", "tvd", "weno"),
        default="weno",
        help="Subcell FV reconstruction: constant, tvd, or weno (default: weno)",
    )
    parser.add_argument("--vtk", action="store_true")
    parser.add_argument("--progress-updates", type=int, default=10)
    parser.add_argument("--maximum-steps", type=int, default=100000)
    parser.add_argument("--maximum-rejections", type=int, default=16)
    parser.add_argument("--density-floor", type=float, default=1.0e-5)
    parser.add_argument("--pressure-floor", type=float, default=1.0e-5)
    parser.add_argument("--output-directory", default="outputs/hybrid_sphere")
    parser.add_argument(
        "--compilation-cache",
        default=".jax_cache",
        help="Persistent XLA cache reused by later runs with the same mesh/shapes.",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="Resume from checkpoint_latest.npz or final_state.npz.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
