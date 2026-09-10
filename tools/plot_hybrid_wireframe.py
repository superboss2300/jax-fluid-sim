"""Render full-domain and cross-section wireframes of the hybrid 3D sphere mesh."""

import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import meshio


def plot_hybrid_wireframe(mesh_path: str, output_path: str):
    mesh = meshio.read(mesh_path)
    pts = np.asarray(mesh.points, dtype=np.float64)

    edges_3d = set()

    for cb in mesh.cells:
        if cb.type.startswith("wedge"):
            corners = cb.data[:, :6]
            wedge_edges = [
                (0, 1), (1, 2), (2, 0),
                (3, 4), (4, 5), (5, 3),
                (0, 3), (1, 4), (2, 5),
            ]
            for cell in corners:
                for u, v in wedge_edges:
                    e = tuple(sorted((int(cell[u]), int(cell[v]))))
                    edges_3d.add(e)
        elif cb.type.startswith("tetra"):
            corners = cb.data[:, :4]
            tet_edges = [
                (0, 1), (1, 2), (2, 0),
                (0, 3), (1, 3), (2, 3),
            ]
            for cell in corners:
                for u, v in tet_edges:
                    e = tuple(sorted((int(cell[u]), int(cell[v]))))
                    edges_3d.add(e)

    # Extract edges near the z = 0 midplane slice (-0.05 < z < 0.05)
    edges_midplane = []
    for u, v in edges_3d:
        p1, p2 = pts[u], pts[v]
        if abs(p1[2]) < 0.06 and abs(p2[2]) < 0.06:
            edges_midplane.append([(p1[0], p1[1]), (p2[0], p2[1])])

    fig = plt.figure(figsize=(18, 11), facecolor="#030712")

    # 1. Full Domain 3D Tunnel Midplane Cross-Section (Top & Bottom, -2 to +2)
    ax1 = fig.add_subplot(2, 2, 1, facecolor="#030712")
    lc1 = LineCollection(edges_midplane, colors="#38bdf8", linewidths=0.5, alpha=0.6)
    ax1.add_collection(lc1)
    # Draw full sphere outline
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), color="#f43f5e", linewidth=2.5, label="Sphere Surface (R=0.5m)")
    ax1.set_xlim(-2.2, 4.2)
    ax1.set_ylim(-2.2, 2.2)
    ax1.set_aspect("equal")
    ax1.set_title("Full Domain Cross-Section (Z=0 Midplane, Full Top & Bottom)", color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("x (m)", color="#94a3b8")
    ax1.set_ylabel("y (m)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#64748b")
    ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white")

    # 2. Zoomed Near-Sphere Boundary Layer Cross-Section
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#030712")
    lc2 = LineCollection(edges_midplane, colors="#38bdf8", linewidths=0.7, alpha=0.8)
    ax2.add_collection(lc2)
    ax2.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), color="#f43f5e", linewidth=3.0)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect("equal")
    ax2.set_title("Zoomed View: Full 360° Sphere & Prismatic Boundary Layer", color="white", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("x (m)", color="#94a3b8")
    ax2.set_ylabel("y (m)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#64748b")

    # 3. 3D View of the Full Sphere Surface and Boundary Elements
    ax3 = fig.add_subplot(2, 1, 2, projection="3d", facecolor="#030712")
    # Subsample 3D edges near the sphere (r < 1.0)
    sphere_edges_3d = []
    step = 1
    for u, v in edges_3d:
        p1, p2 = pts[u], pts[v]
        r1 = np.linalg.norm(p1)
        r2 = np.linalg.norm(p2)
        if r1 < 0.75 and r2 < 0.75:
            sphere_edges_3d.append([p1, p2])

    lc3 = Line3DCollection(sphere_edges_3d[::step], colors="#38bdf8", linewidths=0.5, alpha=0.7)
    ax3.add_collection3d(lc3)
    ax3.set_xlim(-0.8, 0.8)
    ax3.set_ylim(-0.8, 0.8)
    ax3.set_zlim(-0.8, 0.8)
    ax3.view_init(elev=20, azim=45)
    ax3.set_title("3D Full Sphere Surface (Complete 360° Surface, No Symmetry Cut)", color="white", fontsize=13, fontweight="bold", pad=15)
    ax3.set_xlabel("x (m)", color="#94a3b8")
    ax3.set_ylabel("y (m)", color="#94a3b8")
    ax3.set_zlabel("z (m)", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.xaxis.pane.set_edgecolor("#1e293b")
    ax3.yaxis.pane.set_edgecolor("#1e293b")
    ax3.zaxis.pane.set_edgecolor("#1e293b")

    plt.suptitle("Full 3D Sphere Wind Tunnel Grid (20,759 P=2 Elements, Full 360° Wake)", color="white", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved hybrid wireframe visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="mesh/sphere_tunnel_hybrid.msh")
    parser.add_argument("--output", default="/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sphere_hybrid_wireframe.png")
    args = parser.parse_args()
    plot_hybrid_wireframe(args.mesh, args.output)
