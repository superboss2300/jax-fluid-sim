import numpy as np
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import cKDTree

def main():
    print("Loading CFD solution and smooth surface...")
    m_vol = meshio.read("output/sr71_cinematic/sr71_volume.vtu")
    pts_vol = m_vol.points
    vel_vol = m_vol.point_data["velocity"]
    mach_vol = m_vol.point_data["mach"]
    tree = cKDTree(pts_vol)

    m_surf = meshio.read("output/sr71_cinematic/sr71_surface_smooth.vtu")
    pts_surf = m_surf.points
    faces_surf = m_surf.cells_dict["triangle"]
    cp_surf = m_surf.point_data["cp"]

    # Compute streamlines
    seeds = []
    for z_s in np.linspace(-2.2, 2.2, 10):
        seeds.append([-16.0, 0.0, z_s])
        seeds.append([-14.0, 0.25, z_s])
        seeds.append([-14.0, -0.25, z_s])
    for z_s in np.linspace(-6.5, 6.5, 14):
        if abs(z_s) > 1.2:
            seeds.append([-3.0, 0.2, z_s])
            seeds.append([-3.0, -0.25, z_s])

    streamlines = []
    stream_machs = []
    ds = 0.35
    for s in seeds:
        traj = [s]
        pos = np.array(s, dtype=float)
        m_vals = []
        for _ in range(120):
            d, idx = tree.query(pos, k=1)
            if d > 3.0:
                break
            v = vel_vol[idx]
            speed = np.linalg.norm(v)
            if speed < 1.0 or np.isnan(speed):
                break
            m_vals.append(mach_vol[idx])
            pos = pos + (v / speed) * ds
            traj.append(pos.copy())
            if pos[0] > 18.0 or abs(pos[2]) > 9.0 or abs(pos[1]) > 4.0:
                break
        if len(traj) > 12:
            streamlines.append(np.array(traj))
            stream_machs.append(np.mean(m_vals))

    fig = plt.figure(figsize=(24, 15), facecolor="#030712", dpi=180)
    plt.suptitle("Lockheed SR-71 Blackbird — Proposed Cinematic Video Angles (CFD Flowfield)",
                 fontsize=18, color="#f8fafc", fontweight="bold", y=0.985)

    # Convert coordinates for 3D plot: (X, Z, Y)
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

    angles = [
        {
            "id": 1,
            "title": "Angle 1: Front 3/4 Chase-Plane (Lead Shock & Chines)",
            "elev": 20, "azim": -55, "dist": 5.8,
            "desc": "Focuses on needle nose shock, canopy compression, and chine vortex roll-up."
        },
        {
            "id": 2,
            "title": "Angle 2: Top-Down Planform Sweep (Vortex Lift & Symmetry)",
            "elev": 65, "azim": -75, "dist": 5.5,
            "desc": "Captures full double-delta wing, symmetric vortex sheets, and nacelle inlet ingestion."
        },
        {
            "id": 3,
            "title": "Angle 3: Rear 3/4 Trailing Wake (Rudders & Afterbody)",
            "elev": 18, "azim": -135, "dist": 5.8,
            "desc": "Highlights canted vertical rudders, trailing edge expansion, and engine exhaust wakes."
        },
        {
            "id": 4,
            "title": "Angle 4: Low-Angle Supersonic Flyby (Windward Compression Belly)",
            "elev": 4, "azim": -82, "dist": 5.5,
            "desc": "Demonstrates the flat high-pressure compression belly that generates compression lift."
        }
    ]

    for i, a in enumerate(angles):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.set_facecolor("#030712")
        ax.set_title(a["title"], color="#38bdf8", fontsize=13, fontweight="bold", pad=12)

        poly = Poly3DCollection(f_3d, facecolors=cmap_surf(norm_surf(face_cp)), alpha=0.92, edgecolors="none")
        ax.add_collection3d(poly)

        line_coll = Line3DCollection(sl_3d, colors=sl_colors, linewidths=1.2, alpha=0.85)
        ax.add_collection3d(line_coll)

        ax.set_box_aspect([33.3, 16.9, 6.0])
        ax.set_xlim(-17.5, 17.5); ax.set_ylim(-9.0, 9.0); ax.set_zlim(-3.0, 4.0)
        ax.view_init(elev=a["elev"], azim=a["azim"])
        ax.dist = a["dist"]
        ax.set_axis_off()

        fig.text(0.26 + 0.5 * (i % 2), 0.52 - 0.47 * (i // 2), a["desc"],
                 ha="center", color="#94a3b8", fontsize=10, style="italic")

    out_png = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sr71_proposed_video_angles.png"
    plt.tight_layout()
    plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved CFD camera angles showcase to {out_png}")

if __name__ == "__main__":
    main()
