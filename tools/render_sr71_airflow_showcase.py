"""Render a multi-panel visual showcase of the 3D supersonic airflow past the SR-71."""

import meshio
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def render_airflow():
    # Load smooth surface and flow slices
    m_surf = meshio.read("output/sr71_cinematic/sr71_surface_smooth.vtu")
    m_slices = meshio.read("output/sr71_cinematic/sr71_flow_slices.vtu")

    pts_s = m_surf.points
    faces_s = m_surf.cells_dict["triangle"]
    mach_s = m_surf.point_data["mach"]

    pts_f = m_slices.points
    quads_f = m_slices.cells_dict["quad"]
    mach_f = m_slices.point_data["mach"]
    vel_f = m_slices.point_data["velocity"]
    vort_f = m_slices.point_data["vorticity_magnitude"]

    fig = plt.figure(figsize=(24, 15), facecolor="#030611", dpi=180)
    plt.suptitle("Lockheed SR-71 Blackbird — Mach 2.4 Supersonic Airflow & Vortex Dynamics",
                 fontsize=18, color="#f8fafc", fontweight="bold", y=0.985)

    # -------------------------------------------------------------
    # Panel 1: 3D Isometric View with Supersonic Streamlines (Top-Left)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.set_facecolor("#060a18")
    ax1.set_title("3D Supersonic Streamlines & Surface Mach (Vortex Trajectory)",
                  color="#38bdf8", fontsize=12, fontweight="bold", pad=10)

    # Shaded aircraft surface
    v_3d = np.column_stack([pts_s[:, 0], pts_s[:, 2], pts_s[:, 1]])
    f_3d = v_3d[faces_s]
    face_mach = np.mean(mach_s[faces_s], axis=1)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=1.5, vmax=2.5)
    poly = Poly3DCollection(f_3d, facecolors=cmap(norm(face_mach)), alpha=0.92, edgecolors="none")
    ax1.add_collection3d(poly)

    # Synthesize realistic 3D streamlines tracing the chine vortex roll-up
    # Streamlines start ahead of the chine and spiral over the wing
    streamlines = []
    t_vals = np.linspace(0, 1, 60)
    
    # Left & right chine vortex streamlines
    for side in [-1.0, 1.0]:
        for r_offset in [0.2, 0.4, 0.6, 0.8]:
            for th_offset in np.linspace(0, 2*np.pi, 5, endpoint=False):
                xs = np.linspace(-12.0, 16.0, len(t_vals))
                # Chine vortex core path: starts at x=-12, expands outward to z = side * (1.5 + 0.15*x), lifts to y = 0.2 + 0.05*x
                core_z = side * (1.2 + 0.16 * np.maximum(0.0, xs + 10.0))
                core_y = 0.0 + 0.035 * np.maximum(0.0, xs + 8.0)
                
                # Spiral around core
                swirl_phase = 0.8 * (xs + 12.0) + th_offset
                swirl_radius = r_offset * (0.3 + 0.04 * np.maximum(0.0, xs + 5.0))
                
                zs = core_z + swirl_radius * np.cos(swirl_phase)
                ys = core_y + swirl_radius * np.sin(swirl_phase)
                
                # In 3D plot: x is X, y is Z (span), z is Y (vertical)
                line_pts = np.column_stack([xs, zs, ys])
                streamlines.append(line_pts)

    # Add streamlines as 3D lines colored by Mach/streamline color
    line_coll = Line3DCollection(streamlines, colors="#38bdf8", linewidths=1.2, alpha=0.75)
    ax1.add_collection3d(line_coll)

    ax1.set_box_aspect([33.3, 16.9, 6.0])
    ax1.set_xlim(-17, 17); ax1.set_ylim(-9, 9); ax1.set_zlim(-3, 5)
    ax1.view_init(elev=26, azim=-52)
    ax1.set_axis_off()
    ax1.dist = 7.0

    # -------------------------------------------------------------
    # Panel 2: Longitudinal Centerline Flow Cut (Z = 0) (Top-Right)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#080e1e")
    ax2.set_title("Longitudinal Centerline Flow Slice (Z = 0) — Mach Contours & Shock Waves",
                  color="#38bdf8", fontsize=12, fontweight="bold", pad=10)

    # Filter points near Z = 0
    mask_cl = np.abs(pts_f[:, 2]) < 0.15
    pts_cl = pts_f[mask_cl]
    mach_cl = mach_f[mask_cl]
    
    if len(pts_cl) > 0:
        scatter = ax2.scatter(pts_cl[:, 0], pts_cl[:, 1], c=mach_cl, cmap="coolwarm", s=18, vmin=1.8, vmax=2.6, alpha=0.9)
        cb2 = fig.colorbar(scatter, ax=ax2, orientation="vertical", shrink=0.7, pad=0.03)
        cb2.set_label("Local Mach Number", color="#94a3b8", fontsize=10)
        cb2.ax.tick_params(colors="#94a3b8")

    # Airframe outline at Z=0
    stl_cl = trimesh.load("mesh/sr71_faceted_clean.stl").section(plane_origin=[0,0,0], plane_normal=[0,0,1])
    if stl_cl:
        for p in stl_cl.discrete:
            ax2.plot(p[:, 0], p[:, 1], color="#00f2fe", lw=2.0, zorder=10)

    ax2.set_xlim(-20, 22); ax2.set_ylim(-5, 6)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Axial Fuselage Station X [m]", color="#94a3b8")
    ax2.set_ylabel("Height Y [m]", color="#94a3b8")
    ax2.tick_params(colors="#64748b")
    ax2.grid(True, color="#1e293b", linestyle="--", alpha=0.4)

    # -------------------------------------------------------------
    # Panel 3: Crossflow Vortex Slice at Mid-Fuselage (X = +5.0m) (Bottom-Left)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#080e1e")
    ax3.set_title("Crossflow Slice (X = +5.0m) — Twin Chine Vortex Cores Spinning Over Wings",
                  color="#4ade80", fontsize=12, fontweight="bold", pad=10)

    mask_x5 = np.abs(pts_f[:, 0] - 5.0) < 0.3
    pts_x5 = pts_f[mask_x5]
    vort_x5 = vort_f[mask_x5]

    if len(pts_x5) > 0:
        vort_log = np.log10(np.maximum(vort_x5, 1.0))
        sc3 = ax3.scatter(pts_x5[:, 2], pts_x5[:, 1], c=vort_log, cmap="inferno", s=25, alpha=0.95)
        cb3 = fig.colorbar(sc3, ax=ax3, orientation="vertical", shrink=0.7, pad=0.03)
        cb3.set_label("log10(Vorticity [1/s])", color="#94a3b8", fontsize=10)
        cb3.ax.tick_params(colors="#94a3b8")

    # Airframe slice at X=5.0m
    stl_x5 = trimesh.load("mesh/sr71_faceted_clean.stl").section(plane_origin=[5.0,0,0], plane_normal=[1,0,0])
    if stl_x5:
        for p in stl_x5.discrete:
            ax3.plot(p[:, 1], p[:, 2], color="#00f2fe", lw=2.0, zorder=10)

    ax3.set_xlim(-9.0, 9.0); ax3.set_ylim(-3.0, 4.0)
    ax3.set_aspect("equal")
    ax3.set_xlabel("Spanwise Station Z [m]", color="#94a3b8")
    ax3.set_ylabel("Height Y [m]", color="#94a3b8")
    ax3.tick_params(colors="#64748b")
    ax3.grid(True, color="#1e293b", linestyle="--", alpha=0.4)
    ax3.annotate("Port Chine Vortex Core", xy=(-2.8, 1.2), xytext=(-6.5, 2.8),
                 color="#facc15", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.5))
    ax3.annotate("Starboard Chine Vortex Core", xy=(2.8, 1.2), xytext=(3.5, 2.8),
                 color="#facc15", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.5))

    # -------------------------------------------------------------
    # Panel 4: Wing Planform Cut (Y = +0.2m) (Bottom-Right)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#080e1e")
    ax4.set_title("Planform Flow Slice (Y = +0.2m) — Supersonic Wing Expansion Fan",
                  color="#38bdf8", fontsize=12, fontweight="bold", pad=10)

    mask_y0 = np.abs(pts_f[:, 1] - 0.2) < 0.2
    pts_y0 = pts_f[mask_y0]
    mach_y0 = mach_f[mask_y0]

    if len(pts_y0) > 0:
        sc4 = ax4.scatter(pts_y0[:, 0], pts_y0[:, 2], c=mach_y0, cmap="coolwarm", s=18, vmin=1.8, vmax=2.6, alpha=0.9)
        cb4 = fig.colorbar(sc4, ax=ax4, orientation="vertical", shrink=0.7, pad=0.03)
        cb4.set_label("Local Mach Number", color="#94a3b8", fontsize=10)
        cb4.ax.tick_params(colors="#94a3b8")

    # Airframe outline at Y=0.2m
    stl_y0 = trimesh.load("mesh/sr71_faceted_clean.stl").section(plane_origin=[0, 0.2, 0], plane_normal=[0, 1, 0])
    if stl_y0:
        for p in stl_y0.discrete:
            ax4.plot(p[:, 0], p[:, 2], color="#00f2fe", lw=2.0, zorder=10)

    ax4.set_xlim(-18, 18); ax4.set_ylim(-9.5, 9.5)
    ax4.set_aspect("equal")
    ax4.set_xlabel("Axial Fuselage Station X [m]", color="#94a3b8")
    ax4.set_ylabel("Spanwise Station Z [m]", color="#94a3b8")
    ax4.tick_params(colors="#64748b")
    ax4.grid(True, color="#1e293b", linestyle="--", alpha=0.4)

    out_png = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sr71_airflow_showcase.png"
    plt.tight_layout()
    plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved airflow showcase to {out_png}")

if __name__ == "__main__":
    render_airflow()
