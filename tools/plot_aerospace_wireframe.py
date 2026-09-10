"""Render full-domain and zoomed wireframes of the aerospace 3D sphere O-grid."""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import meshio


def plot_aerospace_ogrid_wireframe(mesh_path: str, output_path: str):
    mesh = meshio.read(mesh_path)
    pts = np.asarray(mesh.points, dtype=np.float64)

    edges_3d = set()

    for cb in mesh.cells:
        if cb.type.startswith("hexahedron"):
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

    # Midplane slice edges exactly on z = 0
    midplane_segments = []
    for u, v in edges_3d:
        p1, p2 = pts[u], pts[v]
        if abs(p1[2]) < 1e-5 and abs(p2[2]) < 1e-5:
            midplane_segments.append([(p1[0], p1[1]), (p2[0], p2[1])])

    fig = plt.figure(figsize=(19, 11), facecolor="#030712")

    # 1. Full Domain Midplane Cross-Section (Top & Bottom, -3 to +4)
    ax1 = fig.add_subplot(2, 2, 1, facecolor="#030712")
    lc1 = LineCollection(midplane_segments, colors="#38bdf8", linewidths=0.7, alpha=0.8)
    ax1.add_collection(lc1)
    # Sphere outline
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), color="#f43f5e", linewidth=2.5, label="Sphere Surface (R=0.5m)")
    ax1.set_xlim(-3.3, 4.3)
    ax1.set_ylim(-3.3, 3.3)
    ax1.set_aspect("equal")
    ax1.set_title("Full Domain Midplane (Z=0 Slice, Full 360° Inflow & Extended Wake)", color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("x (m)", color="#94a3b8")
    ax1.set_ylabel("y (m)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#64748b")
    ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white")

    # 2. Zoomed View of Near-Body Orthogonal O-Grid
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#030712")
    lc2 = LineCollection(midplane_segments, colors="#38bdf8", linewidths=1.0, alpha=0.9)
    ax2.add_collection(lc2)
    ax2.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), color="#f43f5e", linewidth=3.0)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect("equal")
    ax2.set_title("Zoomed View: Concentric O-Grid Shells & Wall Orthogonality", color="white", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("x (m)", color="#94a3b8")
    ax2.set_ylabel("y (m)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#64748b")

    # 3. 3D Perspective View of Sphere Shells
    ax3 = fig.add_subplot(2, 1, 2, projection="3d", facecolor="#030712")
    lines_3d = []
    for u, v in edges_3d:
        p1, p2 = pts[u], pts[v]
        r1, r2 = np.linalg.norm(p1), np.linalg.norm(p2)
        if r1 < 1.1 and r2 < 1.1:
            lines_3d.append([p1, p2])

    lc3 = Line3DCollection(lines_3d[::3], colors="#38bdf8", linewidths=0.4, alpha=0.6)
    ax3.add_collection3d(lc3)
    ax3.set_xlim(-1.0, 1.0)
    ax3.set_ylim(-1.0, 1.0)
    ax3.set_zlim(-1.0, 1.0)
    ax3.view_init(elev=22, azim=40)
    ax3.set_title("3D Conformal Spherical O-Grid (Curved P=2 Hexes, Full 360° Shells)", color="white", fontsize=13, fontweight="bold", pad=15)
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

    plt.suptitle("Pure Hexahedral 3D Sphere O-Grid (14,400 Conformal P=2 Hexes, Zero Tets, Zero Wedges)", color="white", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved wireframe visualization to {output_path}")


if __name__ == "__main__":
    plot_aerospace_ogrid_wireframe(
        "mesh/sphere_aerospace_ogrid.msh",
        "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sphere_aerospace_ogrid_final.png"
    )
