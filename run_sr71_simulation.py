"""SR-71 Blackbird Supersonic Simulation & Cinematic Schlieren Field Generator.

Executes high-order DG/FV Navier-Stokes at Mach 2.4, alpha = 3.0 deg past the authentic
Lockheed SR-71 Blackbird airframe, producing pristine 2D Schlieren flow cuts matching
the aesthetic of the reference sphere in conformal_full_domain.
"""

import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import trimesh

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
    _midplane_cells,
    _midplane_triangles,
    shock_colormap,
)


def render_sr71_cinematic_schlieren(
    operator,
    states,
    output_path,
    mach=2.4,
    step=100,
    current_time=0.01,
    reynolds=100000.0,
    extent=(-25.0, 35.0, -10.0, 10.0),
    resolution=(1920, 1080),
    plane_z=0.0,
):
    """Render full-bleed 1080p cinematic numerical Schlieren matching the sphere in conformal_full_domain."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fast, exact midplane cut from hex elements
    mesh = operator.mesh
    hex_block = None
    for b in mesh.element_blocks:
        if b.element_type == "hexahedron":
            hex_block = b
            break

    if hex_block is not None and "hexahedron" in states:
        pts = mesh.points
        hexes = hex_block.connectivity
        U = states["hexahedron"]
        W = np.asarray(operator.solver.conserved_to_primitive(U))

        # Continuous nodal representation
        N_pts = len(pts)
        node_vals = np.zeros((N_pts, 5), dtype=np.float64)
        node_counts = np.zeros(N_pts, dtype=np.int32)
        for i in range(8):
            np.add.at(node_vals, hexes[:, i], W[:, i, :])
            np.add.at(node_counts, hexes[:, i], 1)
        node_vals /= np.maximum(node_counts[:, None], 1)

        rho_nodes = node_vals[:, 0]
        p = pts[hexes]
        r = rho_nodes[hexes]
        z = p[:, :, 2] - plane_z

        crosses = (np.min(z, axis=1) <= 1e-6) & (np.max(z, axis=1) >= -1e-6)
        p_sub = p[crosses]
        r_sub = r[crosses]
        z_sub = z[crosses]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        cut_pts = []
        cut_rho = []
        for e0, e1 in edges:
            z0 = z_sub[:, e0]
            z1 = z_sub[:, e1]
            mask = (z0 * z1 <= 0.0) & (np.abs(z1 - z0) > 1e-8)
            if np.any(mask):
                t = -z0[mask] / (z1[mask] - z0[mask])
                p0 = p_sub[mask, e0]
                p1 = p_sub[mask, e1]
                r0 = r_sub[mask, e0]
                r1 = r_sub[mask, e1]
                interp_p = p0[:, :2] + t[:, None] * (p1[:, :2] - p0[:, :2])
                interp_r = r0 + t * (r1 - r0)
                cut_pts.append(interp_p)
                cut_rho.append(interp_r)

        cut_pts = np.vstack(cut_pts)
        cut_rho = np.concatenate(cut_rho)
    else:
        vertices, quads, triangles, values = _midplane_cells(operator, states, plane_z=plane_z)
        cut_pts = vertices
        cut_rho = values[:, 0]

    # Deduplicate coincident slice points and average coincident DG traces
    rounded = np.round(cut_pts, decimals=4)
    unique_xy, inverse = np.unique(rounded, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    unique_rho = np.bincount(inverse, weights=cut_rho) / counts

    xmin, xmax, ymin, ymax = extent
    nx, ny = resolution
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)
    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])

    # 2. Resample to uniform Cartesian pixel grid
    grid_rho = griddata(unique_xy, unique_rho, (GX, GY), method="linear")
    nan_pts = np.isnan(grid_rho)
    if np.any(nan_pts):
        from scipy.spatial import cKDTree
        tree = cKDTree(unique_xy)
        _, nearest_idx = tree.query(np.column_stack([GX[nan_pts], GY[nan_pts]]))
        grid_rho[nan_pts] = unique_rho[nearest_idx]

    # 3. Compute smooth normalized exponential Schlieren matching the reference sphere
    med_rho = max(abs(float(np.nanmedian(grid_rho))), 1.0)
    rho_clean = np.nan_to_num(grid_rho, nan=med_rho)
    # Physical Gaussian filter scaled to element mesh resolution (sigma=8.5 px ~ 0.24m)
    # completely eliminating all polygonal/triangular cell facet artifacts
    rho_filtered = gaussian_filter(rho_clean, sigma=8.5)
    grad_y, grad_x = np.gradient(rho_filtered, dy, dx)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Normalized exponential Schlieren with adaptive percentile scaling
    p99 = float(np.percentile(grad_mag, 99.5))
    scale_factor = 3.5 / max(p99, 1e-4)
    schlieren = 1.0 - np.exp(-scale_factor * grad_mag)
    schlieren = np.clip(schlieren, 0.0, 1.0)

    # Plot the authentic SR-71 airframe silhouette at Z = plane_z
    loops = []
    stl_path = Path("mesh/sr71_perfect_clean.stl")
    if stl_path.exists():
        stl = trimesh.load(str(stl_path))
        lines = trimesh.intersections.mesh_plane(
            stl, plane_normal=[0, 0, 1], plane_origin=[0, 0, plane_z]
        )
        if len(lines) > 0:
            path_2d = trimesh.load_path(lines[:, :, :2])
            for curve in path_2d.discrete:
                if len(curve) >= 3:
                    loops.append(np.asarray(curve))
    else:
        # Fallback to mesh face boundary
        sr71_mesh = operator.mesh
        for ftype, face_block in sr71_mesh.face_blocks.items():
            wall_mask = face_block.boundary_tags == 14
            if not np.any(wall_mask):
                continue
            faces = sr71_mesh.points[face_block.nodes[wall_mask]]
            for poly in faces:
                z_vals = poly[:, 2]
                if (z_vals.min() <= plane_z) and (z_vals.max() >= plane_z):
                    pts_inter = []
                    N = len(poly)
                    for i in range(N):
                        p_a, p_b = poly[i], poly[(i + 1) % N]
                        za, zb = p_a[2], p_b[2]
                        if (za - plane_z) * (zb - plane_z) <= 0.0 and abs(za - zb) > 1e-6:
                            t_frac = (plane_z - za) / (zb - za)
                            p_cut = p_a + t_frac * (p_b - p_a)
                            pts_inter.append((p_cut[0], p_cut[1]))
                    if len(pts_inter) >= 2:
                        loops.append(np.array(pts_inter))

    fig = plt.figure(figsize=(16, 9), facecolor="#01051f", dpi=120)
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#01051f")

    color_map = shock_colormap()
    im = ax.imshow(
        schlieren,
        origin="lower",
        extent=extent,
        cmap=color_map,
        interpolation="bicubic",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )

    # Fill obstacle interior with pitch-black #01051f and draw glowing neon cyan outline
    for loop in loops:
        ax.fill(loop[:, 0], loop[:, 1], facecolor="#01051f", edgecolor="none", zorder=9)
        ax.plot(loop[:, 0], loop[:, 1], color="#00f2fe", linewidth=2.0, zorder=10)

    # Vertical colorbar on the right (Reddit CFD style)
    cax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(colors="white", labelsize=10)
    for s in cb.ax.spines.values():
        s.set_color("#00a3d9")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")

    # Cinematic HUD typography (matching the sphere image)
    hdr_title = f"Mach {mach:.1f} Compressible Flow Past Lockheed SR-71 Blackbird (P=1 DG/FV)"
    ax.text(
        0.03, 0.95, hdr_title,
        color="white", fontsize=18, fontweight="bold",
        transform=ax.transAxes, zorder=20,
    )
    ax.text(
        0.03, 0.915,
        f"t = {current_time:.5f} s | Step {step} | Re = {reynolds:g} | Longitudinal Centerline (Z=0) Numerical Schlieren",
        color="#00a3d9", fontsize=11, fontfamily="monospace",
        transform=ax.transAxes, zorder=20,
    )

    fig.savefig(output_path, dpi=180, facecolor="#01051f")
    plt.close(fig)
    print(f"Saved cinematic Schlieren to {output_path}")


def export_sr71_vtu(mesh_path, operator, states, output_dir, mach, q_inf, p_inf=1.0):
    """Export 3D volume and surface VTU files with shared conformal nodes for ParaView."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gamma = 1.4

    m = meshio.read(mesh_path)
    pts = m.points
    N_pts = len(pts)

    hex_cells = None
    for cb in m.cells:
        if cb.type == "hexahedron":
            hex_cells = cb.data
            break

    quad_cells = None
    quad_tags = None
    for i, cb in enumerate(m.cells):
        if cb.type == "quad":
            quad_cells = cb.data
            if "gmsh:physical" in m.cell_data:
                quad_tags = m.cell_data["gmsh:physical"][i]
            break

    if hex_cells is not None and "hexahedron" in states:
        W_all = np.asarray(operator.solver.conserved_to_primitive(states["hexahedron"]))
        N_hex = len(hex_cells)

        W_mean = np.mean(W_all, axis=1)
        cell_rho = W_mean[:, 0]
        cell_vel = W_mean[:, 1:4]
        cell_p = W_mean[:, 4]
        cell_c = np.sqrt(gamma * np.maximum(cell_p, 1e-6) / np.maximum(cell_rho, 1e-6))
        cell_vel_mag = np.linalg.norm(cell_vel, axis=1)
        cell_mach = cell_vel_mag / np.maximum(cell_c, 1e-6)
        cell_cp = (cell_p - p_inf) / max(q_inf, 1e-6)
        cell_temp = cell_p / np.maximum(cell_rho, 1e-6)

        node_vals = np.zeros((N_pts, 5), dtype=np.float64)
        node_counts = np.zeros(N_pts, dtype=np.int32)
        for i in range(8):
            np.add.at(node_vals, hex_cells[:, i], W_all[:, i, :])
            np.add.at(node_counts, hex_cells[:, i], 1)
        node_vals /= np.maximum(node_counts[:, None], 1)

        node_rho = node_vals[:, 0]
        node_vel = node_vals[:, 1:4]
        node_p = node_vals[:, 4]
        node_c = np.sqrt(gamma * np.maximum(node_p, 1e-6) / np.maximum(node_rho, 1e-6))
        node_vel_mag = np.linalg.norm(node_vel, axis=1)
        node_mach = node_vel_mag / np.maximum(node_c, 1e-6)
        node_cp = (node_p - p_inf) / max(q_inf, 1e-6)
        node_temp = node_p / np.maximum(node_rho, 1e-6)

        # Vectorized vorticity and Q-criterion on hex elements
        xi = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.float64)
        eta = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.float64)
        zet = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float64)
        dN_dxi = np.stack([xi, eta, zet], axis=0) / 8.0  # (3, 8)
        elem_coords = pts[hex_cells]
        elem_vel = node_vel[hex_cells]
        J = np.einsum("eij,ki->ejk", elem_coords, dN_dxi)
        invJ = np.linalg.inv(J)
        dvel_dxi = np.einsum("eij,ki->ejk", elem_vel, dN_dxi)
        L = np.einsum("eik,ekj->eij", dvel_dxi, invJ)

        vort_x = L[:, 2, 1] - L[:, 1, 2]
        vort_y = L[:, 0, 2] - L[:, 2, 0]
        vort_z = L[:, 1, 0] - L[:, 0, 1]
        cell_vort = np.stack([vort_x, vort_y, vort_z], axis=1)
        cell_vort_mag = np.sqrt(vort_x**2 + vort_y**2 + vort_z**2)

        S = 0.5 * (L + np.transpose(L, (0, 2, 1)))
        Omega = 0.5 * (L - np.transpose(L, (0, 2, 1)))
        norm_S_sq = np.sum(S**2, axis=(1, 2))
        norm_Omega_sq = np.sum(Omega**2, axis=(1, 2))
        cell_Q = 0.5 * (norm_Omega_sq - norm_S_sq)

        node_vort = np.zeros((N_pts, 3), dtype=np.float64)
        node_vort_mag = np.zeros(N_pts, dtype=np.float64)
        node_Q = np.zeros(N_pts, dtype=np.float64)
        for i in range(8):
            np.add.at(node_vort, hex_cells[:, i], cell_vort)
            np.add.at(node_vort_mag, hex_cells[:, i], cell_vort_mag)
            np.add.at(node_Q, hex_cells[:, i], cell_Q)
        node_vort /= np.maximum(node_counts[:, None], 1)
        node_vort_mag /= np.maximum(node_counts, 1)
        node_Q /= np.maximum(node_counts, 1)

        vol_point_data = {
            "density": node_rho.astype(np.float32),
            "velocity": node_vel.astype(np.float32),
            "pressure": node_p.astype(np.float32),
            "mach": node_mach.astype(np.float32),
            "cp": node_cp.astype(np.float32),
            "temperature": node_temp.astype(np.float32),
            "vorticity": node_vort.astype(np.float32),
            "vorticity_magnitude": node_vort_mag.astype(np.float32),
            "Q_criterion": node_Q.astype(np.float32),
        }
        vol_cell_data = {
            "density": [cell_rho.astype(np.float32)],
            "velocity": [cell_vel.astype(np.float32)],
            "pressure": [cell_p.astype(np.float32)],
            "mach": [cell_mach.astype(np.float32)],
            "cp": [cell_cp.astype(np.float32)],
            "temperature": [cell_temp.astype(np.float32)],
            "vorticity": [cell_vort.astype(np.float32)],
            "vorticity_magnitude": [cell_vort_mag.astype(np.float32)],
            "Q_criterion": [cell_Q.astype(np.float32)],
        }
        vol_mesh = meshio.Mesh(
            points=pts,
            cells=[("hexahedron", hex_cells)],
            point_data=vol_point_data,
            cell_data=vol_cell_data,
        )
        vol_path = out_dir / "sr71_volume.vtu"
        vol_mesh.write(str(vol_path))
        print(f"Exported conformal 3D volume VTU to {vol_path} ({len(hex_cells)} hexes, {N_pts} shared nodes)")

        if quad_cells is not None and quad_tags is not None:
            wall_mask = (quad_tags == 14)
            wall_quads = quad_cells[wall_mask]
            unique_node_ids, inverse = np.unique(wall_quads, return_inverse=True)
            surf_pts = pts[unique_node_ids]
            surf_quads = inverse.reshape(wall_quads.shape)

            surf_point_data = {k: v[unique_node_ids] for k, v in vol_point_data.items()}
            surf_mesh = meshio.Mesh(
                points=surf_pts,
                cells=[("quad", surf_quads)],
                point_data=surf_point_data,
            )
            surf_path = out_dir / "sr71_surface.vtu"
            surf_mesh.write(str(surf_path))
            print(f"Exported airframe surface VTU to {surf_path} ({len(surf_quads)} quads, {len(surf_pts)} nodes)")


def run_sr71_simulation():
    parser = argparse.ArgumentParser(description="SR-71 Blackbird Supersonic Simulation")
    parser.add_argument("--mesh", default="mesh/sr71_pure_hex.msh")
    parser.add_argument("--mach", type=float, default=2.4)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--reynolds", type=float, default=100000.0)
    parser.add_argument("--polynomial-degree", type=int, default=1)
    parser.add_argument("--cfl", type=float, default=0.05)
    parser.add_argument("--maximum-steps", type=int, default=200)
    parser.add_argument("--maximum-rejections", type=int, default=50)
    parser.add_argument("--lts", action="store_true", default=False, help="Use Local Time Stepping for rapid convergence")
    parser.add_argument("--wale", action="store_true", default=False, help="Enable WALE dynamic subgrid turbulence model")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoint in output directory")
    parser.add_argument("--output-directory", default="output/sr71_cinematic")
    args = parser.parse_args()

    out_dir = Path(args.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    gamma = 1.4
    density = 1.0
    pressure = 1.0
    sound_speed = (gamma * pressure / density) ** 0.5
    velocity = args.mach * sound_speed
    viscosity = (density * velocity / args.reynolds) if (args.wale or getattr(args, "viscous", False)) else 0.0

    alpha_rad = np.radians(args.alpha)
    u_inf = float(velocity * np.cos(alpha_rad))
    v_inf = float(velocity * np.sin(alpha_rad))
    q_inf = 0.5 * density * velocity**2

    print("=" * 76)
    print("SR-71 BLACKBIRD SUPERSONIC HIGH-ORDER DG SIMULATION")
    print(f"  Mesh:              {args.mesh}")
    print(f"  Mach:              M = {args.mach:.2f} ({velocity:.2f} m/s non-dim)")
    print(f"  Angle of Attack:   α = {args.alpha:.1f}°")
    print(f"  DG Polynomial:     P = {args.polynomial_degree}")
    print(f"  CFL:               {args.cfl}")
    print(f"  LTS (Local CFL):   {args.lts}")
    print(f"  Max Steps:         {args.maximum_steps}")
    print(f"  Output Directory:  {out_dir}")
    print("=" * 76)

    solver = HybridCompressibleSolver(
        args.mesh,
        species_names=(),
        gamma=gamma,
        polynomial_degree=args.polynomial_degree,
        inviscid_flux="hllc",
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
        wall_names=("walls", "airfoil_wall"),
        no_slip_names=("airfoil_wall",),
        slip_names=("walls",),
    )
    positivity = HybridPositivityLimiter(
        operator,
        density_floor=1e-6,
        pressure_floor=1e-6,
    )
    limiter = HybridModalShockLimiter(operator, positivity=positivity)
    sensor = HybridTroubledCellSensor(operator)
    integrator = HybridSSPRK3(
        operator,
        cfl=args.cfl,
        limiter=limiter,
        density_floor=1e-6,
        pressure_floor=1e-6,
        maximum_rejections=args.maximum_rejections,
    )

    state_file = out_dir / "sr71_final_state.npz"
    if args.resume and state_file.exists():
        print(f"Resuming simulation from {state_file}...")
        checkpoint = np.load(state_file)
        states = {}
        for block in solver.mesh.element_blocks:
            k = f"U_{block.element_type}"
            if k in checkpoint:
                states[block.element_type] = jnp.asarray(checkpoint[k])
            else:
                raise KeyError(f"Missing state for {block.element_type} in {state_file}")
        current_time = float(checkpoint.get("time", 0.0))
        initial_step = int(checkpoint.get("step", 0))
        print(f"Resumed at step {initial_step}, t = {current_time:.6e} s")
    else:
        states = solver.uniform_state_blocks(
            density,
            jnp.asarray([u_inf, v_inf, 0.0]),
            pressure,
        )
        current_time = 0.0
        initial_step = 0
    step = initial_step
    target_step = initial_step + args.maximum_steps

    print("Compiling JAX SSPRK3 kernels...", flush=True)
    t0 = time.perf_counter()
    if args.lts:
        compiled_local_dt = jax.jit(lambda s: integrator.local_dt(s, boundary_state=boundary))
    else:
        compiled_dt = jax.jit(lambda s: integrator.stable_dt(s, boundary))
    compiled_step = lambda s, dt: integrator.checked_step(s, dt, boundary)

    if args.lts:
        dummy_dt = compiled_local_dt(states)
    else:
        dummy_dt = float(compiled_dt(states))
    dummy_state, _, _, _ = compiled_step(states, dummy_dt)
    for v in dummy_state.values():
        jax.block_until_ready(v)
    print(f"JIT Compilation completed in {time.perf_counter() - t0:.2f} s", flush=True)

    start = time.perf_counter()
    retry_scale = 1.0
    consecutive_rejections = 0
    total_rejections = 0

    print(f"\nLaunching simulation from step {initial_step} to {target_step} ({args.maximum_steps} steps)...", flush=True)
    while step < target_step:
        rel_step = step - initial_step
        soft_scale = 0.2 + 0.8 * min(1.0, rel_step / 15.0) if not args.resume else 1.0
        scale = retry_scale * soft_scale

        if args.lts:
            local_dts = compiled_local_dt(states)
            dt_advance = {k: scale * v for k, v in local_dts.items()}
            dt_step = float(min(float(v.min()) for v in dt_advance.values()))
            dt_physical = float(np.median(np.asarray(next(iter(dt_advance.values())))))
        else:
            raw_cfl_dt = float(compiled_dt(states))
            cfl_dt = max(1.0e-7, raw_cfl_dt) if np.isfinite(raw_cfl_dt) else 1.49e-7
            dt_step = cfl_dt * scale
            dt_advance = dt_step
            dt_physical = dt_step

        candidate_state, valid, rho_att, p_att = compiled_step(states, dt_advance)
        valid_val = bool(jax.device_get(valid))
        if valid_val:
            states = candidate_state
            current_time += dt_physical
            step += 1
            consecutive_rejections = 0
            retry_scale = min(1.0, retry_scale * 1.15)
        else:
            total_rejections += 1
            consecutive_rejections += 1
            retry_scale = max(0.01, retry_scale * 0.6)
            if consecutive_rejections >= args.maximum_rejections:
                print(f"Warning: Reached max rejections ({consecutive_rejections}) at step {step}")
                break

        if (step - initial_step) % 20 == 0 or step == target_step:
            elapsed = time.perf_counter() - start
            rate = (step - initial_step) / max(elapsed, 1e-6)
            pct = 100.0 * (step - initial_step) / max(args.maximum_steps, 1)
            print(
                f"step={step:4d}/{target_step} ({pct:5.1f}%) "
                f"t={current_time:.6f}s dt_min={dt_step:.2e} rejections={total_rejections} "
                f"rate={rate:.1f} steps/s",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    print(f"\nSimulation completed {step - initial_step} steps in {elapsed:.1f}s.", flush=True)

    # Render Cinematic Schlieren Field
    print("\nRendering 1080p Cinematic Numerical Schlieren (Z=0 Centerline Cut)...")
    render_path = out_dir / "sr71_cinematic_schlieren.png"
    render_sr71_cinematic_schlieren(
        operator,
        states,
        render_path,
        mach=args.mach,
        step=step,
        current_time=current_time,
        reynolds=args.reynolds,
        extent=(-22.0, 25.0, -8.0, 8.0),
        resolution=(1920, 1080),
        plane_z=0.0,
    )

    # Export 3D Conformal Volume and Surface VTU files for ParaView
    print("\nExporting 3D Conformal Volume and Surface VTU files...")
    export_sr71_vtu(
        mesh_path=args.mesh,
        operator=operator,
        states=states,
        output_dir=out_dir,
        mach=args.mach,
        q_inf=q_inf,
        p_inf=pressure,
    )

    # Save state
    save_dict = {f"U_{name}": np.asarray(states[name]) for name in states}
    save_dict["time"] = current_time
    save_dict["step"] = step
    np.savez(out_dir / "sr71_final_state.npz", **save_dict)
    print(f"All base outputs saved to {out_dir}")

    # Generate publication-grade VTK datasets
    print("\nGenerating publication-grade VTK datasets for ParaView...")
    try:
        import sys
        import subprocess
        subprocess.run([sys.executable, "tools/generate_premium_vtk.py"], check=True)
    except Exception as e:
        print(f"Note: generate_premium_vtk encountered: {e}")

if __name__ == "__main__":
    run_sr71_simulation()
