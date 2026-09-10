import os
import shutil
import subprocess
import numpy as np
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import cKDTree

def render_all_angles():
    print("=" * 76)
    print("RENDERING ALL 4 LONG-VIEW CINEMATIC VIDEOS FROM CFD FLOWFIELD")
    print("=" * 76)

    m_vol = meshio.read("output/sr71_cinematic/sr71_volume.vtu")
    pts_vol = m_vol.points
    vel_vol = m_vol.point_data["velocity"]
    mach_vol = m_vol.point_data["mach"]
    tree = cKDTree(pts_vol)

    m_surf = meshio.read("output/sr71_cinematic/sr71_surface_smooth.vtu")
    pts_surf = m_surf.points
    faces_surf = m_surf.cells_dict["triangle"]
    cp_surf = m_surf.point_data["cp"]

    # Compute dense 3D streamlines extending deep into downstream wake
    seeds = []
    for z_s in np.linspace(-2.5, 2.5, 14):
        seeds.append([-16.0, 0.0, z_s])
        seeds.append([-13.0, 0.25, z_s])
        seeds.append([-13.0, -0.25, z_s])
    for z_s in np.linspace(-7.5, 7.5, 18):
        if abs(z_s) > 1.2:
            seeds.append([-3.0, 0.2, z_s])
            seeds.append([-3.0, -0.25, z_s])
            seeds.append([5.0, 0.1, z_s])
            seeds.append([5.0, -0.6, z_s])

    streamlines = []
    ds = 0.4
    for s in seeds:
        traj = [s]
        pos = np.array(s, dtype=float)
        for _ in range(260):
            d, idx = tree.query(pos, k=1)
            if d > 4.5:
                break
            v = vel_vol[idx]
            speed = np.linalg.norm(v)
            if speed < 1.0 or np.isnan(speed):
                break
            pos = pos + (v / speed) * ds
            traj.append(pos.copy())
            if pos[0] > 44.0 or abs(pos[2]) > 14.0 or abs(pos[1]) > 8.0:
                break
        if len(traj) > 20:
            streamlines.append(np.array(traj))

    # 3D surface
    v_3d = np.column_stack([pts_surf[:, 0], pts_surf[:, 2], pts_surf[:, 1]])
    f_3d = v_3d[faces_surf]
    face_cp = np.mean(cp_surf[faces_surf], axis=1)
    cmap_surf = plt.cm.plasma
    norm_surf = plt.Normalize(vmin=-0.15, vmax=0.35)

    sl_3d = []
    sl_lengths = []
    for traj in streamlines:
        traj_plot = np.column_stack([traj[:, 0], traj[:, 2], traj[:, 1]])
        sl_3d.append(traj_plot)
        dists = np.sqrt(np.sum(np.diff(traj_plot, axis=0)**2, axis=1))
        sl_lengths.append(np.concatenate([[0], np.cumsum(dists)]))

    # Detect ffmpeg encoder
    enc = "h264_nvenc"
    try:
        res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "h264_nvenc" in res.stdout:
            enc = "h264_nvenc"
        elif "libopenh264" in res.stdout:
            enc = "libopenh264"
        else:
            enc = "mpeg4"
    except Exception:
        enc = "mpeg4"

    angles = [
        ("rear_chase_wake", "Rear Chase-Plane View (Wake Turbulence & Vortex Breakdown)", 16, -142, 4.8),
        ("front_quarter", "Front 3/4 Perspective (Bow Shock to Trailing Plume)", 20, -58, 4.8),
        ("top_planform", "Top-Down Planform Sweep (Wingspan & Twin Wake Tracks)", 65, -80, 4.6),
        ("side_flyby", "Supersonic Side Flyby (Compression Belly & Wake Layer)", 6, -90, 4.6)
    ]

    for name, title, base_elev, base_azim, base_dist in angles:
        out_dir = f"output/sr71_cinematic/frames_{name}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        num_frames = 60
        print(f"\nRendering {num_frames} frames for {title}...")
        for f in range(num_frames):
            t_frac = f / float(num_frames)
            elev = base_elev + 2.0 * np.sin(2 * np.pi * t_frac)
            azim = base_azim + 3.0 * t_frac
            dist = base_dist - 0.2 * np.sin(np.pi * t_frac)

            fig = plt.figure(figsize=(16, 9), facecolor="#02050f", dpi=120)
            ax = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="#02050f")

            poly = Poly3DCollection(f_3d, facecolors=cmap_surf(norm_surf(face_cp)), alpha=0.92, edgecolors="none")
            ax.add_collection3d(poly)

            # Pulsing flow segments
            pulse_segments = []
            pulse_colors = []
            pulse_speed = 1.2
            for traj_plot, arc_len in zip(sl_3d, sl_lengths):
                total_len = arc_len[-1]
                if total_len < 5.0:
                    continue
                phase = (arc_len - pulse_speed * f) % 12.0
                active = phase < 6.0
                if np.sum(active) > 2:
                    pts = traj_plot[active]
                    for k in range(len(pts) - 1):
                        pulse_segments.append([pts[k], pts[k+1]])
                        pulse_colors.append("#00f2fe")

            line_coll = Line3DCollection(pulse_segments, colors=pulse_colors, linewidths=1.4, alpha=0.85)
            ax.add_collection3d(line_coll)

            base_lines = Line3DCollection(sl_3d, colors="#38bdf8", linewidths=0.6, alpha=0.25)
            ax.add_collection3d(base_lines)

            ax.set_box_aspect([65.0, 28.0, 12.0])
            ax.set_xlim(-20, 45); ax.set_ylim(-14, 14); ax.set_zlim(-5, 7)
            ax.view_init(elev=elev, azim=azim)
            ax.dist = dist
            ax.set_axis_off()

            ax.text2D(0.04, 0.94, "Lockheed SR-71 Blackbird — Mach 2.4 Supersonic Trajectory & Wake",
                      color="#ffffff", fontsize=15, fontweight="bold", transform=ax.transAxes)
            ax.text2D(0.04, 0.90, f"Camera: {title} | 65-Meter Extended Flowfield",
                      color="#38bdf8", fontsize=10, transform=ax.transAxes)

            fig.savefig(f"{out_dir}/frame_{f:04d}.png", facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)

        mp4_out = f"output/sr71_cinematic/sr71_{name}.mp4"
        cmd = [
            "ffmpeg", "-y", "-r", "24",
            "-i", f"{out_dir}/frame_%04d.png",
            "-c:v", enc, "-pix_fmt", "yuv420p",
            mp4_out
        ]
        subprocess.run(cmd, check=True)
        print(f"  -> Encoded {mp4_out} successfully!")

    print("\nAll 4 cinematic videos encoded successfully!")

if __name__ == "__main__":
    render_all_angles()
