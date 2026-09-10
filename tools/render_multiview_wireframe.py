"""Render multi-view CAD wireframe of the repaired SR-71 geometry."""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def render_multiview():
    stl_path = "mesh/sr71_faceted_clean.stl"
    m = trimesh.load(stl_path)
    v = m.vertices
    f = m.faces
    edges = m.edges_unique
    e_pts = v[edges]

    fig = plt.figure(figsize=(24, 14), facecolor="#060913", dpi=180)

    # 1. 3D Perspective View (Top-Left)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.set_facecolor("#060913")
    ax1.set_title("3D Isometric Shaded Wireframe (Authentic Lockheed SR-71 Airframe)",
                  fontsize=13, color="#38bdf8", fontweight="bold", pad=12)

    normals = m.face_normals
    light = np.array([-0.4, 0.6, 0.7])
    light /= np.linalg.norm(light)
    diffuse = np.clip(np.abs(np.dot(normals, light)), 0.2, 1.0)
    
    poly = Poly3DCollection(v[f], alpha=0.92, edgecolor="#0284c7", linewidth=0.25)
    colors = np.outer(diffuse, np.array([0.06, 0.16, 0.32]))
    poly.set_facecolor(colors)
    ax1.add_collection3d(poly)

    # Tightly fit SR-71 bounds: X: [-17, 17], Y: [-2, 4], Z: [-9, 9]
    ax1.set_xlim(-17, 17)
    ax1.set_ylim(-9, 9)
    ax1.set_zlim(-5, 9)
    # Reorder coordinates so Z (wingspan) is horizontal and Y is vertical in 3D
    ax1.cla()
    # In matplotlib 3d: X is x, Y is z (span), Z is y (vertical)
    v_3d = np.column_stack([v[:, 0], v[:, 2], v[:, 1]])
    f_3d = v_3d[f]
    poly = Poly3DCollection(f_3d, alpha=0.92, edgecolor="#0284c7", linewidth=0.25)
    poly.set_facecolor(colors)
    ax1.add_collection3d(poly)
    ax1.set_xlim(-17, 17)
    ax1.set_ylim(-9, 9)
    ax1.set_zlim(-3, 5)
    ax1.view_init(elev=25, azim=-60)
    ax1.set_axis_off()

    # 2. Planform Top View (Top-Right)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#0a0f1e")
    ax2.set_title("Planform View (X-Z Plane) — Uniformly Subdivided Wings & Chines",
                  fontsize=13, color="#38bdf8", fontweight="bold", pad=12)
    lines_top = LineCollection(e_pts[:, :, [0, 2]], colors="#38bdf8", linewidths=0.35, alpha=0.7)
    ax2.add_collection(lines_top)
    ax2.set_xlim(-18, 18); ax2.set_ylim(-9.5, 9.5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Axial Station X [m] (Fuselage: 33.3 m)", color="#94a3b8", fontsize=10)
    ax2.set_ylabel("Spanwise Station Z [m] (Wingspan: 16.9 m)", color="#94a3b8", fontsize=10)
    ax2.tick_params(colors="#64748b")
    ax2.grid(True, color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.5)

    # 3. Side Elevation View (Bottom-Left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#0a0f1e")
    ax3.set_title("Side Elevation (X-Y Plane) — Chine Profile, Shock Spikes & Twin Canted Tails",
                  fontsize=13, color="#38bdf8", fontweight="bold", pad=12)
    lines_side = LineCollection(e_pts[:, :, [0, 1]], colors="#38bdf8", linewidths=0.35, alpha=0.7)
    ax3.add_collection(lines_side)
    ax3.set_xlim(-18, 18); ax3.set_ylim(-3.5, 4.5)
    ax3.set_aspect("equal")
    ax3.set_xlabel("Axial Station X [m]", color="#94a3b8", fontsize=10)
    ax3.set_ylabel("Vertical Station Y [m]", color="#94a3b8", fontsize=10)
    ax3.tick_params(colors="#64748b")
    ax3.grid(True, color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.5)

    # 4. Front View (Bottom-Right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#0a0f1e")
    ax4.set_title("Front View (Z-Y Plane) — Inward-Canted Vertical Rudders (15 deg) & Symmetry",
                  fontsize=13, color="#38bdf8", fontweight="bold", pad=12)
    lines_front = LineCollection(e_pts[:, :, [2, 1]], colors="#38bdf8", linewidths=0.45, alpha=0.75)
    ax4.add_collection(lines_front)
    ax4.set_xlim(-9.5, 9.5); ax4.set_ylim(-3.0, 4.0)
    ax4.set_aspect("equal")
    ax4.set_xlabel("Spanwise Station Z [m]", color="#94a3b8", fontsize=10)
    ax4.set_ylabel("Vertical Station Y [m]", color="#94a3b8", fontsize=10)
    ax4.tick_params(colors="#64748b")
    ax4.grid(True, color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.5)

    out_png = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sr71_multiview_wireframe.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved multiview wireframe to {out_png}")

if __name__ == "__main__":
    render_multiview()
