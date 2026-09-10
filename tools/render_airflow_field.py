import numpy as np
import trimesh
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

def main():
    print("Loading CFD volume solution...")
    m_vol = meshio.read("output/sr71_cinematic/sr71_volume.vtu")
    pts_vol = m_vol.points
    vel_vol = m_vol.point_data["velocity"]
    mach_vol = m_vol.point_data["mach"]
    p_vol = m_vol.point_data["pressure"]
    tree = cKDTree(pts_vol)

    print("Loading smooth surface and flow slices...")
    m_surf = meshio.read("output/sr71_cinematic/sr71_surface_smooth.vtu")
    pts_surf = m_surf.points
    faces_surf = m_surf.cells_dict["triangle"]
    mach_surf = m_surf.point_data["mach"]
    cp_surf = m_surf.point_data["cp"]

    m_slices = meshio.read("output/sr71_cinematic/sr71_flow_slices.vtu")
    pts_sl = m_slices.points
    mach_sl = m_slices.point_data["mach"]
    vort_sl = m_slices.point_data["vorticity_magnitude"]

    # 1. Tracing 3D Streamlines close to the body
    print("Tracing 3D streamlines...")
    seeds = []
    # Upstream chine and nose seeds
    for z_s in np.linspace(-2.2, 2.2, 12):
        seeds.append([-16.0, 0.0, z_s])
        seeds.append([-14.0, 0.25, z_s])
        seeds.append([-14.0, -0.25, z_s])
        seeds.append([-10.0, 0.45, z_s])

    # Wing and nacelle seeds
    for z_s in np.linspace(-6.8, 6.8, 16):
        if abs(z_s) > 1.2:
            seeds.append([-2.0, 0.25, z_s])
            seeds.append([-2.0, -0.3, z_s])
            seeds.append([4.0, 0.1, z_s])

    streamlines = []
    stream_machs = []
    ds = 0.28
    for s in seeds:
        traj = [s]
        pos = np.array(s, dtype=float)
        m_vals = []
        for _ in range(140):
            d, idx = tree.query(pos, k=1)
            if d > 2.8:
                break
            v = vel_vol[idx]
            speed = np.linalg.norm(v)
            if speed < 1.0 or np.isnan(speed):
                break
            m_vals.append(mach_vol[idx])
            pos = pos + (v / speed) * ds
            traj.append(pos.copy())
            if pos[0] > 17.5 or abs(pos[2]) > 9.0 or abs(pos[1]) > 4.0:
                break
        if len(traj) > 15:
            streamlines.append(np.array(traj))
            stream_machs.append(np.mean(m_vals))

    print(f"Generated {len(streamlines)} streamlines.")

    fig = plt.figure(figsize=(24, 15), facecolor="#030712", dpi=180)
    plt.suptitle("Lockheed SR-71 Blackbird — Mach 2.4 Supersonic Airflow Physics & Shock Aerodynamics",
                 fontsize=18, color="#f8fafc", fontweight="bold", y=0.985)

    # -------------------------------------------------------------
    # Panel 1: 3D Supersonic Streamlines & Airframe Surface Pressure
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.set_facecolor("#030712")
    ax1.set_title("3D Supersonic Streamlines & Surface Pressure ($C_p$)",
                  color="#38bdf8", fontsize=13, fontweight="bold", pad=12)

    # SR-71 Airframe: coordinate swap (X, Z, Y) for 3D visual orientation
    v_3d = np.column_stack([pts_surf[:, 0], pts_surf[:, 2], pts_surf[:, 1]])
    f_3d = v_3d[faces_surf]
    face_cp = np.mean(cp_surf[faces_surf], axis=1)
    cmap_surf = plt.cm.plasma
    norm_surf = plt.Normalize(vmin=-0.15, vmax=0.35)
    poly = Poly3DCollection(f_3d, facecolors=cmap_surf(norm_surf(face_cp)), alpha=0.92, edgecolors="none")
    ax1.add_collection3d(poly)

    # Streamlines: convert (x, y, z) -> (x, z, y)
    sl_3d = []
    sl_colors = []
    cmap_sl = plt.cm.cool
    for traj, m_avg in zip(streamlines, stream_machs):
        traj_plot = np.column_stack([traj[:, 0], traj[:, 2], traj[:, 1]])
        sl_3d.append(traj_plot)
        norm_m = np.clip((m_avg - 2.1) / 0.4, 0.0, 1.0)
        sl_colors.append(cmap_sl(norm_m))

    line_coll = Line3DCollection(sl_3d, colors=sl_colors, linewidths=1.2, alpha=0.85)
    ax1.add_collection3d(line_coll)

    ax1.set_box_aspect([33.3, 16.9, 6.0])
    ax1.set_xlim(-17.5, 17.5); ax1.set_ylim(-9.0, 9.0); ax1.set_zlim(-3.0, 4.0)
    ax1.view_init(elev=22, azim=-58)
    ax1.set_axis_off()
    ax1.dist = 5.6

    # -------------------------------------------------------------
    # Panel 2: Longitudinal Centerline Flow Cut (Z = 0)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#0b132b")
    ax2.set_title("Longitudinal Centerline Flow Slice (Z = 0) — Bow Shock, Cockpit Shock & Belly Compression",
                  color="#38bdf8", fontsize=13, fontweight="bold", pad=12)

    mask_cl = np.abs(pts_sl[:, 2]) < 0.05
    pts_cl = pts_sl[mask_cl]
    mach_cl = mach_sl[mask_cl]

    if len(pts_cl) > 0:
        xi = np.linspace(-20.0, 22.0, 350)
        yi = np.linspace(-3.5, 4.5, 120)
        grid_m = griddata((pts_cl[:, 0], pts_cl[:, 1]), mach_cl, (xi[None, :], yi[:, None]), method="linear")
        cnt2 = ax2.contourf(xi, yi, grid_m, levels=np.linspace(1.85, 2.55, 60), cmap="coolwarm", extend="both")
        cb2 = fig.colorbar(cnt2, ax=ax2, orientation="vertical", shrink=0.75, pad=0.03)
        cb2.set_label("Local Mach Number", color="#94a3b8", fontsize=11)
        cb2.ax.tick_params(colors="#94a3b8")

    # Airframe outline at Z=0
    stl_cl = trimesh.load("mesh/sr71_faceted_clean.stl").section(plane_origin=[0,0,0], plane_normal=[0,0,1])
    if stl_cl:
        for p in stl_cl.discrete:
            ax2.plot(p[:, 0], p[:, 1], color="#00f2fe", lw=2.2, zorder=10)

    # Annotations
    ax2.annotate("Conical Nose Shock\n(Mach 2.4 Inflow)", xy=(-16.5, -0.6), xytext=(-19.0, 2.6),
                 color="#facc15", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.6))
    ax2.annotate("Cockpit Canopy Shock\n(Mach Deceleration)", xy=(-10.8, 0.4), xytext=(-10.5, 3.4),
                 color="#facc15", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.6))
    ax2.annotate("Flat Compression Belly\n(Uniform High Pressure)", xy=(3.0, -1.0), xytext=(2.0, -2.6),
                 color="#34d399", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#34d399", lw=1.6))

    ax2.set_xlim(-20, 22); ax2.set_ylim(-3.5, 4.5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Fuselage Axial Station X [m] (Nose at -16.65m, Tail at +16.65m)", color="#94a3b8", fontsize=10)
    ax2.set_ylabel("Height Y [m]", color="#94a3b8", fontsize=10)
    ax2.tick_params(colors="#64748b")
    ax2.grid(True, color="#1e293b", linestyle="--", alpha=0.4)

    # -------------------------------------------------------------
    # Panel 3: Chine Vortex Cross-Flow Slice at Mid-Fuselage (X = +6.0m)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#0b132b")
    ax3.set_title("Crossflow Slice (X = +6.0m) — Chine Vortices & Engine Nacelle Inflow",
                  color="#4ade80", fontsize=13, fontweight="bold", pad=12)

    mask_x6 = np.abs(pts_sl[:, 0] - 6.0) < 0.05
    pts_x6 = pts_sl[mask_x6]
    vort_x6 = vort_sl[mask_x6]

    if len(pts_x6) > 0:
        zi = np.linspace(-8.0, 8.0, 160)
        yi = np.linspace(-2.5, 3.5, 90)
        grid_v = griddata((pts_x6[:, 2], pts_x6[:, 1]), np.log10(np.maximum(vort_x6, 1e-2)),
                          (zi[None, :], yi[:, None]), method="linear")
        cnt3 = ax3.contourf(zi, yi, grid_v, levels=np.linspace(-0.5, 3.2, 50), cmap="inferno", extend="both")
        cb3 = fig.colorbar(cnt3, ax=ax3, orientation="vertical", shrink=0.75, pad=0.03)
        cb3.set_label("log10(Vorticity Magnitude [1/s])", color="#94a3b8", fontsize=11)
        cb3.ax.tick_params(colors="#94a3b8")

    # Airframe slice at X=6.0m
    stl_x6 = trimesh.load("mesh/sr71_faceted_clean.stl").section(plane_origin=[6.0, 0, 0], plane_normal=[1, 0, 0])
    if stl_x6:
        for p in stl_x6.discrete:
            ax3.plot(p[:, 2], p[:, 1], color="#00f2fe", lw=2.2, zorder=10)

    # Annotations
    ax3.annotate("Port Chine Vortex Sheet", xy=(-3.8, -0.3), xytext=(-6.8, 2.0),
                 color="#facc15", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.6))
    ax3.annotate("Starboard Chine Vortex Sheet", xy=(3.8, -0.3), xytext=(3.5, 2.0),
                 color="#facc15", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.6))
    ax3.annotate("Port Nacelle Inflow", xy=(-5.12, -0.7), xytext=(-7.5, -1.8),
                 color="#38bdf8", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.6))
    ax3.annotate("Starboard Nacelle Inflow", xy=(5.12, -0.7), xytext=(4.5, -1.8),
                 color="#38bdf8", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.6))

    ax3.set_xlim(-8.0, 8.0); ax3.set_ylim(-2.5, 3.5)
    ax3.set_aspect("equal")
    ax3.set_xlabel("Spanwise Station Z [m] (Port: -Z, Starboard: +Z)", color="#94a3b8", fontsize=10)
    ax3.set_ylabel("Height Y [m]", color="#94a3b8", fontsize=10)
    ax3.tick_params(colors="#64748b")
    ax3.grid(True, color="#1e293b", linestyle="--", alpha=0.4)

    # -------------------------------------------------------------
    # Panel 4: Horizontal Wing Planform Slice (Y = +0.2m)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#0b132b")
    ax4.set_title("Planform Flow Slice (Y = +0.2m) — Supersonic Wing Expansion Fan & Nacelle Shocks",
                  color="#38bdf8", fontsize=13, fontweight="bold", pad=12)

    mask_y0 = np.abs(pts_sl[:, 1] - 0.2) < 0.05
    pts_y0 = pts_sl[mask_y0]
    mach_y0 = mach_sl[mask_y0]

    if len(pts_y0) > 0:
        xi_p = np.linspace(-18.0, 18.0, 300)
        zi_p = np.linspace(-8.5, 8.5, 150)
        grid_mp = griddata((pts_y0[:, 0], pts_y0[:, 2]), mach_y0, (xi_p[None, :], zi_p[:, None]), method="linear")
        cnt4 = ax4.contourf(xi_p, zi_p, grid_mp, levels=np.linspace(1.85, 2.55, 60), cmap="coolwarm", extend="both")
        cb4 = fig.colorbar(cnt4, ax=ax4, orientation="vertical", shrink=0.75, pad=0.03)
        cb4.set_label("Local Mach Number", color="#94a3b8", fontsize=11)
        cb4.ax.tick_params(colors="#94a3b8")

    # Airframe outline at Y=0.2m
    stl_y0 = trimesh.load("mesh/sr71_faceted_clean.stl").section(plane_origin=[0, 0.2, 0], plane_normal=[0, 1, 0])
    if stl_y0:
        for p in stl_y0.discrete:
            ax4.plot(p[:, 0], p[:, 2], color="#00f2fe", lw=2.2, zorder=10)

    # Annotations
    ax4.annotate("Chine Expansion\n(Accelerating Flow)", xy=(-6.0, 1.8), xytext=(-14.0, 5.0),
                 color="#facc15", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.6))
    ax4.annotate("Nacelle Inflow Capture", xy=(4.2, 5.12), xytext=(-1.0, 7.2),
                 color="#38bdf8", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.6))
    ax4.annotate("Symmetric Delta Expansion\n(Port = Starboard)", xy=(9.0, -3.5), xytext=(6.5, -6.8),
                 color="#34d399", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#34d399", lw=1.6))

    ax4.set_xlim(-18, 18); ax4.set_ylim(-8.5, 8.5)
    ax4.set_aspect("equal")
    ax4.set_xlabel("Fuselage Axial Station X [m]", color="#94a3b8", fontsize=10)
    ax4.set_ylabel("Spanwise Station Z [m]", color="#94a3b8", fontsize=10)
    ax4.tick_params(colors="#64748b")
    ax4.grid(True, color="#1e293b", linestyle="--", alpha=0.4)

    out_png = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sr71_airflow_showcase.png"
    plt.tight_layout()
    plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved refined airflow showcase to {out_png}")

if __name__ == "__main__":
    main()
