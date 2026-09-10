"""Comprehensive Multi-View Wireframe Visualizer for SR-71 Airframe and Cart3D Hex Mesh."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import trimesh
import meshio


def render_sr71_cad_wireframe(stl_path: str, output_path: str):
    """Render 5-panel orthographic and 3D wireframe views of the authentic SR-71 CAD surface."""
    print(f"Loading CAD STL: {stl_path}...")
    mesh = trimesh.load(stl_path)
    vertices = mesh.vertices
    faces = mesh.faces

    # Extract unique triangle edges
    edges = set()
    for tri in faces:
        for i in range(3):
            u, v = int(tri[i]), int(tri[(i + 1) % 3])
            edges.add(tuple(sorted((u, v))))
    edges = np.array(list(edges))

    p1 = vertices[edges[:, 0]]
    p2 = vertices[edges[:, 1]]

    fig = plt.figure(figsize=(22, 14), facecolor="#030712", dpi=150)
    plt.suptitle("Lockheed SR-71 Blackbird: Authentic Surface Triangulation & Wireframe",
                 color="white", fontsize=20, fontweight="bold", y=0.98)

    # 1. 3D Isometric View
    ax1 = fig.add_subplot(2, 3, 1, projection="3d", facecolor="#030712")
    step = 1
    segments_3d = np.stack([p1[::step], p2[::step]], axis=1)
    lc1 = Line3DCollection(segments_3d, colors="#38bdf8", linewidths=0.4, alpha=0.7)
    ax1.add_collection3d(lc1)
    ax1.set_xlim(-18, 18)
    ax1.set_ylim(-10, 10)
    ax1.set_zlim(-5, 5)
    ax1.view_init(elev=25, azim=-125)
    ax1.set_title("3D Isometric Wireframe", color="white", fontsize=14, fontweight="bold", pad=10)
    ax1.axis("off")

    # 2. Planform / Top-Down View (X-Z plane)
    ax2 = fig.add_subplot(2, 3, 2, facecolor="#030712")
    seg_top = np.stack([p1[:, [0, 2]], p2[:, [0, 2]]], axis=1)
    lc2 = LineCollection(seg_top, colors="#00f2fe", linewidths=0.5, alpha=0.7)
    ax2.add_collection(lc2)
    # Highlight centerline Z=0
    ax2.axhline(0, color="#f43f5e", linestyle="--", linewidth=1.2, alpha=0.8, label="Symmetry Centerline (Z=0)")
    ax2.set_xlim(-18, 18)
    ax2.set_ylim(-9.5, 9.5)
    ax2.set_aspect("equal")
    ax2.set_title("Planform View (Top-Down: X vs Z)", color="white", fontsize=14, fontweight="bold")
    ax2.set_xlabel("X (Longitudinal / Streamwise, m)", color="#94a3b8")
    ax2.set_ylabel("Z (Spanwise, m)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    for s in ax2.spines.values(): s.set_color("#334155")
    ax2.grid(True, linestyle="--", alpha=0.15, color="#64748b")
    ax2.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=10)

    # 3. Longitudinal Profile / Side View (X-Y plane)
    ax3 = fig.add_subplot(2, 3, 3, facecolor="#030712")
    seg_side = np.stack([p1[:, [0, 1]], p2[:, [0, 1]]], axis=1)
    lc3 = LineCollection(seg_side, colors="#38bdf8", linewidths=0.5, alpha=0.7)
    ax3.add_collection(lc3)
    ax3.set_xlim(-18, 18)
    ax3.set_ylim(-3.5, 4.0)
    ax3.set_aspect("equal")
    ax3.set_title("Elevation Profile (Side: X vs Y)", color="white", fontsize=14, fontweight="bold")
    ax3.set_xlabel("X (Longitudinal / Streamwise, m)", color="#94a3b8")
    ax3.set_ylabel("Y (Vertical, m)", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")
    for s in ax3.spines.values(): s.set_color("#334155")
    ax3.grid(True, linestyle="--", alpha=0.15, color="#64748b")

    # 4. Front View (Z-Y plane)
    ax4 = fig.add_subplot(2, 3, 4, facecolor="#030712")
    seg_front = np.stack([p1[:, [2, 1]], p2[:, [2, 1]]], axis=1)
    lc4 = LineCollection(seg_front, colors="#a78bfa", linewidths=0.5, alpha=0.7)
    ax4.add_collection(lc4)
    ax4.axvline(0, color="#f43f5e", linestyle="--", linewidth=1.2, alpha=0.8)
    ax4.set_xlim(-9.5, 9.5)
    ax4.set_ylim(-3.0, 3.5)
    ax4.set_aspect("equal")
    ax4.set_title("Front View (Looking Aft: Z vs Y)", color="white", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Z (Spanwise, m)", color="#94a3b8")
    ax4.set_ylabel("Y (Vertical, m)", color="#94a3b8")
    ax4.tick_params(colors="#94a3b8")
    for s in ax4.spines.values(): s.set_color("#334155")
    ax4.grid(True, linestyle="--", alpha=0.15, color="#64748b")

    # 5. Zoomed Dorsal Spine & Cockpit Ridge Detail (Top-Down X in [-16, 2])
    ax5 = fig.add_subplot(2, 3, 5, facecolor="#030712")
    mask_cockpit = ((p1[:, 0] >= -16.5) & (p1[:, 0] <= 2.0) & (np.abs(p1[:, 2]) <= 3.0)) | \
                   ((p2[:, 0] >= -16.5) & (p2[:, 0] <= 2.0) & (np.abs(p2[:, 2]) <= 3.0))
    seg_cockpit = np.stack([p1[mask_cockpit][:, [0, 2]], p2[mask_cockpit][:, [0, 2]]], axis=1)
    lc5 = LineCollection(seg_cockpit, colors="#34d399", linewidths=0.8, alpha=0.9)
    ax5.add_collection(lc5)
    ax5.axhline(0, color="#f43f5e", linestyle="--", linewidth=1.5, alpha=0.9, label="Centerline Z=0")
    ax5.set_xlim(-16.5, 2.0)
    ax5.set_ylim(-2.5, 2.5)
    ax5.set_aspect("equal")
    ax5.set_title("Cockpit & Dorsal Spine Alignment (Top Detail)", color="white", fontsize=14, fontweight="bold")
    ax5.set_xlabel("X (m)", color="#94a3b8")
    ax5.set_ylabel("Z (m)", color="#94a3b8")
    ax5.tick_params(colors="#94a3b8")
    for s in ax5.spines.values(): s.set_color("#334155")
    ax5.grid(True, linestyle="--", alpha=0.2, color="#64748b")
    ax5.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=10)

    # 6. Zoomed Side View of Cockpit Canopy & Nose Spike
    ax6 = fig.add_subplot(2, 3, 6, facecolor="#030712")
    mask_side_zoom = ((p1[:, 0] >= -16.5) & (p1[:, 0] <= 2.0)) | ((p2[:, 0] >= -16.5) & (p2[:, 0] <= 2.0))
    seg_side_zoom = np.stack([p1[mask_side_zoom][:, [0, 1]], p2[mask_side_zoom][:, [0, 1]]], axis=1)
    lc6 = LineCollection(seg_side_zoom, colors="#fbbf24", linewidths=0.8, alpha=0.9)
    ax6.add_collection(lc6)
    ax6.set_xlim(-16.5, 2.0)
    ax6.set_ylim(-1.5, 2.0)
    ax6.set_aspect("equal")
    ax6.set_title("Canopy & Nose Chine Profile (Side Detail)", color="white", fontsize=14, fontweight="bold")
    ax6.set_xlabel("X (m)", color="#94a3b8")
    ax6.set_ylabel("Y (m)", color="#94a3b8")
    ax6.tick_params(colors="#94a3b8")
    for s in ax6.spines.values(): s.set_color("#334155")
    ax6.grid(True, linestyle="--", alpha=0.2, color="#64748b")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=160, facecolor="#030712")
    plt.close(fig)
    print(f"Saved CAD wireframe visualization to {output_path}")


def render_sr71_cart3d_mesh_wireframe(mesh_path: str, stl_path: str, output_path: str):
    """Render 4-panel wireframe visualization of the NASA Cart3D computational hex mesh."""
    print(f"Loading Cart3D Mesh: {mesh_path}...")
    mesh = meshio.read(mesh_path)
    pts = mesh.points

    # Extract all hexahedral edges
    hex_edges_local = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    edges_3d = set()
    for cb in mesh.cells:
        if cb.type.startswith("hexahedron"):
            for cell in cb.data:
                for u, v in hex_edges_local:
                    e = tuple(sorted((int(cell[u]), int(cell[v]))))
                    edges_3d.add(e)

    edges_arr = np.array(list(edges_3d))
    p1 = pts[edges_arr[:, 0]]
    p2 = pts[edges_arr[:, 1]]

    # Load STL silhouette for overlays
    stl = trimesh.load(stl_path)

    fig = plt.figure(figsize=(22, 14), facecolor="#030712", dpi=150)
    plt.suptitle("NASA Cart3D Pure Hex Wind Tunnel Mesh: Conformal Orthogonal Discretization",
                 color="white", fontsize=20, fontweight="bold", y=0.98)

    # 1. Longitudinal Symmetry Plane (Z = 0 Cut) - Full Wind Tunnel Domain
    ax1 = fig.add_subplot(2, 2, 1, facecolor="#030712")
    mask_z0 = (np.abs(p1[:, 2]) < 1e-4) & (np.abs(p2[:, 2]) < 1e-4)
    seg_z0 = np.stack([p1[mask_z0][:, [0, 1]], p2[mask_z0][:, [0, 1]]], axis=1)
    lc1 = LineCollection(seg_z0, colors="#38bdf8", linewidths=0.6, alpha=0.7)
    ax1.add_collection(lc1)

    # Airframe slice at Z = 0
    lines_z0 = trimesh.intersections.mesh_plane(stl, plane_normal=[0, 0, 1], plane_origin=[0, 0, 0])
    if len(lines_z0) > 0:
        lc_air = LineCollection(lines_z0[:, :, :2], colors="#f43f5e", linewidths=2.2, label="SR-71 Airframe Wall (Tag 14)")
        ax1.add_collection(lc_air)

    ax1.set_xlim(-28, 42)
    ax1.set_ylim(-14, 16)
    ax1.set_aspect("equal")
    ax1.set_title("Longitudinal Symmetry Plane (Z=0 Slice) - Full Wind Tunnel", color="white", fontsize=14, fontweight="bold")
    ax1.set_xlabel("X (m)", color="#94a3b8")
    ax1.set_ylabel("Y (m)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    for s in ax1.spines.values(): s.set_color("#334155")
    ax1.grid(True, linestyle="--", alpha=0.15, color="#64748b")
    ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=11)

    # 2. Zoomed Near-Body Longitudinal Grid (Z = 0 Cut around Aircraft)
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#030712")
    lc2 = LineCollection(seg_z0, colors="#38bdf8", linewidths=0.9, alpha=0.85)
    ax2.add_collection(lc2)
    if len(lines_z0) > 0:
        path_2d = trimesh.load_path(lines_z0[:, :, :2])
        for curve in path_2d.discrete:
            if len(curve) >= 3:
                ax2.fill(curve[:, 0], curve[:, 1], facecolor="#0f172a", edgecolor="#f43f5e", linewidth=2.0, zorder=5)
    ax2.set_xlim(-18, 19)
    ax2.set_ylim(-4, 5)
    ax2.set_aspect("equal")
    ax2.set_title("Near-Body Mesh Detail (Z=0): Dense Core (dx=0.70m, dy=0.45m)", color="white", fontsize=14, fontweight="bold")
    ax2.set_xlabel("X (m)", color="#94a3b8")
    ax2.set_ylabel("Y (m)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    for s in ax2.spines.values(): s.set_color("#334155")
    ax2.grid(True, linestyle="--", alpha=0.15, color="#64748b")

    # 3. Planform Chine / Wing Plane Cut (Y = 0 Cut)
    ax3 = fig.add_subplot(2, 2, 3, facecolor="#030712")
    mask_y0 = (np.abs(p1[:, 1]) < 0.25) & (np.abs(p2[:, 1]) < 0.25)
    seg_y0 = np.stack([p1[mask_y0][:, [0, 2]], p2[mask_y0][:, [0, 2]]], axis=1)
    lc3 = LineCollection(seg_y0, colors="#00f2fe", linewidths=0.7, alpha=0.75)
    ax3.add_collection(lc3)

    lines_y0 = trimesh.intersections.mesh_plane(stl, plane_normal=[0, 1, 0], plane_origin=[0, 0, 0])
    if len(lines_y0) > 0:
        lc_wing = LineCollection(lines_y0[:, :, [0, 2]], colors="#f43f5e", linewidths=2.0, label="Airframe Wall (Y=0)")
        ax3.add_collection(lc_wing)

    ax3.set_xlim(-20, 22)
    ax3.set_ylim(-12, 12)
    ax3.set_aspect("equal")
    ax3.set_title("Planform Chine & Wing Grid (Y≈0 Cut): Spanwise Grid (dz=0.55m)", color="white", fontsize=14, fontweight="bold")
    ax3.set_xlabel("X (m)", color="#94a3b8")
    ax3.set_ylabel("Z (m)", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")
    for s in ax3.spines.values(): s.set_color("#334155")
    ax3.grid(True, linestyle="--", alpha=0.15, color="#64748b")
    ax3.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=11)

    # 4. Transverse Engine / Nacelle Station Cut (X = 6.0 m Cross-Section)
    ax4 = fig.add_subplot(2, 2, 4, facecolor="#030712")
    x_station = 6.0
    mask_x = (np.abs(p1[:, 0] - x_station) < 0.5) & (np.abs(p2[:, 0] - x_station) < 0.5)
    seg_x = np.stack([p1[mask_x][:, [2, 1]], p2[mask_x][:, [2, 1]]], axis=1)
    lc4 = LineCollection(seg_x, colors="#a78bfa", linewidths=0.9, alpha=0.85)
    ax4.add_collection(lc4)

    lines_x = trimesh.intersections.mesh_plane(stl, plane_normal=[1, 0, 0], plane_origin=[x_station, 0, 0])
    if len(lines_x) > 0:
        lc_x_wall = LineCollection(lines_x[:, :, [2, 1]], colors="#f43f5e", linewidths=2.2, label=f"Airframe Wall (X={x_station}m)")
        ax4.add_collection(lc_x_wall)

    ax4.set_xlim(-10, 10)
    ax4.set_ylim(-3, 4)
    ax4.set_aspect("equal")
    ax4.set_title(f"Engine Nacelle Cross-Section (X = {x_station}m Cut)", color="white", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Z (m)", color="#94a3b8")
    ax4.set_ylabel("Y (m)", color="#94a3b8")
    ax4.tick_params(colors="#94a3b8")
    for s in ax4.spines.values(): s.set_color("#334155")
    ax4.grid(True, linestyle="--", alpha=0.15, color="#64748b")
    ax4.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=160, facecolor="#030712")
    plt.close(fig)
    print(f"Saved Cart3D hex mesh wireframe visualization to {output_path}")


if __name__ == "__main__":
    out_dir = Path("output/sr71_cinematic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    render_sr71_cad_wireframe("mesh/sr71_perfect_clean.stl", str(out_dir / "sr71_cad_wireframe.png"))
    render_sr71_cart3d_mesh_wireframe("mesh/sr71_cart3d_hex.msh", "mesh/sr71_perfect_clean.stl", str(out_dir / "sr71_cart3d_mesh_wireframe.png"))

