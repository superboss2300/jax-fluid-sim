import numpy as np
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import cKDTree

def main():
    print("Loading CFD solution and surface...")
    m_vol = meshio.read("output/sr71_cinematic/sr71_volume.vtu")
    pts_vol = m_vol.points
    vel_vol = m_vol.point_data["velocity"]
    mach_vol = m_vol.point_data["mach"]
    tree = cKDTree(pts_vol)

    m_surf = meshio.read("output/sr71_cinematic/sr71_surface_smooth.vtu")
    pts_surf = m_surf.points
    faces_surf = m_surf.cells_dict["triangle"]
    cp_surf = m_surf.point_data["cp"]

    # Compute extended wake streamlines (from nose/chines/wings all the way to X = 44m)
    seeds = []
    # Chines & Nose
    for z_s in np.linspace(-2.5, 2.5, 12):
        seeds.append([-15.5, 0.0, z_s])
        seeds.append([-12.0, 0.2, z_s])
        seeds.append([-12.0, -0.2, z_s])
    # Wings & Nacelles
    for z_s in np.linspace(-7.5, 7.5, 16):
        if abs(z_s) > 1.0:
            seeds.append([-2.0, 0.2, z_s])
            seeds.append([-2.0, -0.2, z_s])
            seeds.append([6.0, 0.1, z_s])
            seeds.append([6.0, -0.6, z_s]) # Engine wake

    streamlines = []
    stream_machs = []
    ds = 0.4
    for s in seeds:
        traj = [s]
        pos = np.array(s, dtype=float)
        m_vals = []
        for _ in range(250):
            d, idx = tree.query(pos, k=1)
            if d > 4.5:
                break
            v = vel_vol[idx]
            speed = np.linalg.norm(v)
            if speed < 1.0 or np.isnan(speed):
                break
            m_vals.append(mach_vol[idx])
            pos = pos + (v / speed) * ds
            traj.append(pos.copy())
            if pos[0] > 44.0 or abs(pos[2]) > 14.0 or abs(pos[1]) > 8.0:
                break
        if len(traj) > 20:
            streamlines.append(np.array(traj))
            stream_machs.append(np.mean(m_vals))

    fig = plt.figure(figsize=(24, 15), facecolor="#030712", dpi=180)
    plt.suptitle("Lockheed SR-71 Blackbird — Long-View Perspectives (Extended Downstream Wake)",
                 fontsize=18, color="#f8fafc", fontweight="bold", y=0.985)

    # 3D surface
    v_3d = np.column_stack([pts_surf[:, 0], pts_surf[:, 2], pts_surf[:, 1]])
    f_3d = v_3d[faces_surf]
    face_cp = np.mean(cp_surf[faces_surf], axis=1)
    cmap_surf = plt.cm.plasma
    norm_surf = plt.Normalize(vmin=-0.15, vmax=0.35)

    sl_3d = []
    sl_colors = []
    cmap_sl = plt.cm.cool
    for traj, m_avg in zip(streamlines, stream_machs):
        traj_plot = np.column_stack([traj[:, 0], traj[:, 2], traj[:, 1]])
        sl_3d.append(traj_plot)
        norm_m = np.clip((m_avg - 2.1) / 0.4, 0.0, 1.0)
        sl_colors.append(cmap_sl(norm_m))

    long_angles = [
        {
            "id": 1,
            "title": "Long-View 1: Elevated Front 3/4 (Full Trajectory & Downstream Wake)",
            "elev": 22, "azim": -60, "dist": 6.8,
            "desc": "Wide framing from nose (-17m) to trailing wake (+45m), showing the entire airframe and downstream wake trail."
        },
        {
            "id": 2,
            "title": "Long-View 2: High Planform Sweep (Spanwise Vortex Breakdown & Exhaust Plumes)",
            "elev": 60, "azim": -80, "dist": 6.5,
            "desc": "Top-down extended view showing the full wingspan, twin engine wake paths, and downstream spreading."
        },
        {
            "id": 3,
            "title": "Long-View 3: Rear Chase-Plane Quarter (Looking Forward through the Wake)",
            "elev": 16, "azim": -140, "dist": 6.8,
            "desc": "Looking from downstream (+45m) forward towards the aircraft tail, showcasing the twin trailing vortex plumes."
        },
        {
            "id": 4,
            "title": "Long-View 4: Side Supersonic Flyby (33m Airframe + 30m Trailing Plume)",
            "elev": 8, "azim": -90, "dist": 6.8,
            "desc": "True side-profile elevation showing the bow shock angle, under-belly compression, and horizontal wake development."
        }
    ]

    for i, a in enumerate(long_angles):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.set_facecolor("#030712")
        ax.set_title(a["title"], color="#38bdf8", fontsize=13, fontweight="bold", pad=12)

        poly = Poly3DCollection(f_3d, facecolors=cmap_surf(norm_surf(face_cp)), alpha=0.92, edgecolors="none")
        ax.add_collection3d(poly)

        line_coll = Line3DCollection(sl_3d, colors=sl_colors, linewidths=1.2, alpha=0.85)
        ax.add_collection3d(line_coll)

        # Extended framing: X from -20 to +45 (65m total!), Z from -14 to +14, Y from -5 to +7
        ax.set_box_aspect([65.0, 28.0, 12.0])
        ax.set_xlim(-20, 45); ax.set_ylim(-14, 14); ax.set_zlim(-5, 7)
        ax.view_init(elev=a["elev"], azim=a["azim"])
        ax.dist = a["dist"]
        ax.set_axis_off()

        fig.text(0.26 + 0.5 * (i % 2), 0.52 - 0.47 * (i // 2), a["desc"],
                 ha="center", color="#94a3b8", fontsize=10, style="italic")

    out_png = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sr71_long_view_angles.png"
    plt.tight_layout()
    plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved long view camera angles showcase to {out_png}")

if __name__ == "__main__":
    main()
