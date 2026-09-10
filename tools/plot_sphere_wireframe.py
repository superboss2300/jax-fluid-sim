"""Render 2D and 3D wireframe views of the structured sphere mesh."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import meshio


def plot_sphere_wireframe(mesh_path: str, output_path: str):
    mesh = meshio.read(mesh_path)
    pts = np.asarray(mesh.points, dtype=np.float64)

    # Collect unique edges from 2D and 3D cells
    edges_3d = set()

    for cb in mesh.cells:
        if cb.type in ("hexahedron", "hexahedron20", "hexahedron27"):
            corners = cb.data[:, :8]
            hex_edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            for cell in corners:
                for u, v in hex_edges:
                    e = tuple(sorted((int(cell[u]), int(cell[v]))))
                    edges_3d.add(e)
        elif cb.type in ("wedge", "wedge15", "wedge18"):
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

    # Extract edges lying on the z ~ 0 plane (front symmetry face)
    z_min = np.min(pts[:, 2])
    edges_midplane = []
    for u, v in edges_3d:
        p1, p2 = pts[u], pts[v]
        if abs(p1[2] - z_min) < 1e-4 and abs(p2[2] - z_min) < 1e-4:
            edges_midplane.append([(p1[0], p1[1]), (p2[0], p2[1])])

    if not edges_midplane:
        for u, v in edges_3d:
            p1, p2 = pts[u], pts[v]
            if abs(p1[2]) < 0.05 and abs(p2[2]) < 0.05:
                edges_midplane.append([(p1[0], p1[1]), (p2[0], p2[1])])

    fig = plt.figure(figsize=(18, 11), facecolor="#030712")

    # 1. 2D Full Domain Wireframe
    ax1 = fig.add_subplot(2, 2, 1, facecolor="#030712")
    lc1 = LineCollection(edges_midplane, colors="#38bdf8", linewidths=0.5, alpha=0.7)
    ax1.add_collection(lc1)
    # Sphere arc
    theta = np.linspace(0, np.pi, 200)
    ax1.plot(-0.5 * np.cos(theta), 0.5 * np.sin(theta), color="#f43f5e", linewidth=2.5, label="Sphere Surface (R=0.5m)")
    ax1.set_xlim(-3.2, 4.2)
    ax1.set_ylim(-0.1, 3.2)
    ax1.set_aspect("equal")
    ax1.set_title("Full Domain Structured O-Grid (Midplane Cross-Section)", color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("x (m)", color="#94a3b8")
    ax1.set_ylabel("y (m)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#64748b")
    ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white")

    # 2. 2D Zoomed Near-Body Wireframe
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#030712")
    lc2 = LineCollection(edges_midplane, colors="#38bdf8", linewidths=0.8, alpha=0.85)
    ax2.add_collection(lc2)
    ax2.plot(-0.5 * np.cos(theta), 0.5 * np.sin(theta), color="#f43f5e", linewidth=3.0)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-0.05, 1.3)
    ax2.set_aspect("equal")
    ax2.set_title("Zoomed Near-Body Flow Alignment & Boundary Layer", color="white", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("x (m)", color="#94a3b8")
    ax2.set_ylabel("y (m)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#64748b")

    # 3. 3D Perspective Wireframe of Sector
    ax3 = fig.add_subplot(2, 1, 2, projection="3d", facecolor="#030712")
    lines_3d = []
    step = 2
    all_edges_3d = list(edges_3d)
    for u, v in all_edges_3d[::step]:
        lines_3d.append([pts[u], pts[v]])

    lc3 = Line3DCollection(lines_3d, colors="#38bdf8", linewidths=0.4, alpha=0.6)
    ax3.add_collection3d(lc3)
    ax3.set_xlim(-1.5, 2.5)
    ax3.set_ylim(-0.2, 2.0)
    ax3.set_zlim(-0.2, 0.4)
    ax3.view_init(elev=22, azim=-60)
    ax3.set_title("3D Conformal Hexahedral Sector (Extruded Around Axis)", color="white", fontsize=13, fontweight="bold", pad=15)
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

    plt.suptitle("Curved P=2 Structured Sphere Mesh (100% Conformal Hex & Wedge, 0 Tets)", color="white", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved wireframe visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="mesh/sphere_quad_sector_balanced.msh")
    parser.add_argument("--output", default="/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sphere_grid_wireframe.png")
    args = parser.parse_args()
    plot_sphere_wireframe(args.mesh, args.output)
