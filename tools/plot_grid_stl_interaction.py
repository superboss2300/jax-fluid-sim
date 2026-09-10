"""Visualizes exactly how the snapped conforming hex grid interacts with and fits the authentic SR-71 STL airframe."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import trimesh
import meshio


def plot_grid_stl_interaction(
    mesh_path: str = "mesh/sr71_snapped_hex.msh",
    stl_path: str = "mesh/sr71_perfect_clean.stl",
    output_path: str = "output/sr71_cinematic/sr71_grid_stl_interaction.png",
):
    print(f"Loading mesh {mesh_path} and STL {stl_path}...")
    mesh = meshio.read(mesh_path)
    pts = mesh.points
    stl = trimesh.load(stl_path)

    # Extract quads and find airfoil wall quads (Tag 14)
    quad_cells = None
    quad_tags = None
    hex_cells = None
    for i, c in enumerate(mesh.cells):
        if c.type == "quad":
            quad_cells = c.data
            quad_tags = mesh.cell_data["gmsh:physical"][i]
        elif c.type.startswith("hexahedron"):
            hex_cells = c.data

    wall_quads = quad_cells[quad_tags == 14]
    print(f"Found {len(wall_quads)} airfoil wall quads (Tag 14)")

    # Build continuous triangular surface from wall quads
    wall_tris = []
    for q in wall_quads:
        wall_tris.append([q[0], q[1], q[2]])
        wall_tris.append([q[0], q[2], q[3]])
    wall_surf = trimesh.Trimesh(vertices=pts, faces=np.array(wall_tris), process=False)

    # Extract hex grid edges
    hex_edges_local = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    edges_3d = set()
    for cell in hex_cells:
        for u, v in hex_edges_local:
            e = tuple(sorted((int(cell[u]), int(cell[v]))))
            edges_3d.add(e)
    all_edges = np.array(list(edges_3d))
    h_p1 = pts[all_edges[:, 0]]
    h_p2 = pts[all_edges[:, 1]]

    fig = plt.figure(figsize=(22, 14), facecolor="#030712", dpi=160)
    plt.suptitle("SR-71 Airframe vs Snapped Conforming Hex Mesh: Zero-Gap Geometric Fit",
                 color="white", fontsize=20, fontweight="bold", y=0.98)

    # =========================================================================
    # Panel 1: Nose Spike & Cockpit High Zoom (Z = 0 Slice)
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1, facecolor="#030712")

    # Grid lines on Z = 0
    mask_z0 = (np.abs(h_p1[:, 2]) < 1e-4) & (np.abs(h_p2[:, 2]) < 1e-4)
    seg_grid_z0 = np.stack([h_p1[mask_z0][:, [0, 1]], h_p2[mask_z0][:, [0, 1]]], axis=1)
    lc_grid1 = LineCollection(seg_grid_z0, colors="#1e293b", linewidths=0.8, alpha=0.8, label="Hex Mesh Edges")
    ax1.add_collection(lc_grid1)

    # Continuous Wall Quads Slice at Z = 0
    wall_lines_z0 = trimesh.intersections.mesh_plane(wall_surf, plane_normal=[0, 0, 1], plane_origin=[0, 0, 0])
    if len(wall_lines_z0) > 0:
        lc_wall1 = LineCollection(wall_lines_z0[:, :, :2], colors="#facc15", linewidths=3.0, alpha=0.95, label="Snapped Wall (Tag 14 Quads)")
        ax1.add_collection(lc_wall1)

    # Authentic CAD STL Profile on Z = 0
    stl_lines_z0 = trimesh.intersections.mesh_plane(stl, plane_normal=[0, 0, 1], plane_origin=[0, 0, 0])
    if len(stl_lines_z0) > 0:
        lc_stl1 = LineCollection(stl_lines_z0[:, :, :2], colors="#00f2fe", linewidths=1.8, linestyle="--", label="Authentic CAD Geometry (STL)")
        ax1.add_collection(lc_stl1)

    ax1.set_xlim(-17.0, -6.0)
    ax1.set_ylim(-1.8, 1.2)
    ax1.set_aspect("equal")
    ax1.set_title("Forward Fuselage & Needle Nose Fit (Z = 0 Longitudinal Cut)", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlabel("X (Longitudinal, m)", color="#94a3b8")
    ax1.set_ylabel("Y (Vertical, m)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    for s in ax1.spines.values(): s.set_color("#334155")
    ax1.grid(True, linestyle=":", alpha=0.15, color="#64748b")
    ax1.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=10)

    # =========================================================================
    # Panel 2: Chine Leading Edge High Zoom (Y = 0 Planform Cut)
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#030712")

    mask_y0 = (np.abs(h_p1[:, 1]) < 0.25) & (np.abs(h_p2[:, 1]) < 0.25)
    seg_grid_y0 = np.stack([h_p1[mask_y0][:, [0, 2]], h_p2[mask_y0][:, [0, 2]]], axis=1)
    lc_grid2 = LineCollection(seg_grid_y0, colors="#1e293b", linewidths=0.8, alpha=0.8, label="Hex Mesh Edges")
    ax2.add_collection(lc_grid2)

    wall_lines_y0 = trimesh.intersections.mesh_plane(wall_surf, plane_normal=[0, 1, 0], plane_origin=[0, 0, 0])
    if len(wall_lines_y0) > 0:
        lc_wall2 = LineCollection(wall_lines_y0[:, :, [0, 2]], colors="#facc15", linewidths=3.0, alpha=0.95, label="Snapped Wall (Tag 14 Quads)")
        ax2.add_collection(lc_wall2)

    stl_lines_y0 = trimesh.intersections.mesh_plane(stl, plane_normal=[0, 1, 0], plane_origin=[0, 0, 0])
    if len(stl_lines_y0) > 0:
        lc_stl2 = LineCollection(stl_lines_y0[:, :, [0, 2]], colors="#00f2fe", linewidths=1.8, linestyle="--", label="Authentic CAD Geometry (STL)")
        ax2.add_collection(lc_stl2)

    ax2.set_xlim(-17.0, -2.0)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_aspect("equal")
    ax2.set_title("Forward Chine Sweep Fit (Y ≈ 0 Planform Cut)", color="white", fontsize=13, fontweight="bold")
    ax2.set_xlabel("X (Longitudinal, m)", color="#94a3b8")
    ax2.set_ylabel("Z (Spanwise, m)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    for s in ax2.spines.values(): s.set_color("#334155")
    ax2.grid(True, linestyle=":", alpha=0.15, color="#64748b")
    ax2.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=10)

    # =========================================================================
    # Panel 3: Engine Nacelle Cowl Cross-Section (X = 6.0 m Station)
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3, facecolor="#030712")
    x_station = 6.0

    mask_x = (np.abs(h_p1[:, 0] - x_station) < 0.35) & (np.abs(h_p2[:, 0] - x_station) < 0.35)
    seg_grid_x = np.stack([h_p1[mask_x][:, [2, 1]], h_p2[mask_x][:, [2, 1]]], axis=1)
    lc_grid3 = LineCollection(seg_grid_x, colors="#1e293b", linewidths=0.8, alpha=0.8, label="Hex Mesh Edges")
    ax3.add_collection(lc_grid3)

    wall_lines_x = trimesh.intersections.mesh_plane(wall_surf, plane_normal=[1, 0, 0], plane_origin=[x_station, 0, 0])
    if len(wall_lines_x) > 0:
        lc_wall3 = LineCollection(wall_lines_x[:, :, [2, 1]], colors="#facc15", linewidths=3.0, alpha=0.95, label="Snapped Wall (Tag 14 Quads)")
        ax3.add_collection(lc_wall3)

    stl_lines_x = trimesh.intersections.mesh_plane(stl, plane_normal=[1, 0, 0], plane_origin=[x_station, 0, 0])
    if len(stl_lines_x) > 0:
        lc_stl3 = LineCollection(stl_lines_x[:, :, [2, 1]], colors="#00f2fe", linewidths=1.8, linestyle="--", label="Authentic CAD Geometry (STL)")
        ax3.add_collection(lc_stl3)

    ax3.set_xlim(2.5, 6.5)
    ax3.set_ylim(-2.0, 1.5)
    ax3.set_aspect("equal")
    ax3.set_title(f"Starboard J58 Nacelle Cowl Fit (X = {x_station}m Station)", color="white", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Z (Spanwise, m)", color="#94a3b8")
    ax3.set_ylabel("Y (Vertical, m)", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")
    for s in ax3.spines.values(): s.set_color("#334155")
    ax3.grid(True, linestyle=":", alpha=0.15, color="#64748b")
    ax3.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=10)

    # =========================================================================
    # Panel 4: 3D Perspective Overlay of Snapped Tag 14 Quads vs STL Surface
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4, projection="3d", facecolor="#030712")

    # Draw CAD surface as translucent cyan mesh
    tri_verts = stl.vertices[stl.faces]
    stl_poly = Poly3DCollection(tri_verts[::2], facecolors="#0284c7", edgecolors="#00f2fe",
                                linewidths=0.2, alpha=0.35)
    ax4.add_collection3d(stl_poly)

    # Draw snapped boundary quads in gold wireframe
    quad_verts = pts[wall_quads]
    quad_poly = Poly3DCollection(quad_verts[::2], facecolors="none", edgecolors="#facc15",
                                 linewidths=0.6, alpha=0.8)
    ax4.add_collection3d(quad_poly)

    ax4.set_xlim(-18, 18)
    ax4.set_ylim(-9, 9)
    ax4.set_zlim(-3, 3)
    ax4.view_init(elev=22, azim=-120)
    ax4.set_title("3D Conformal Fit: Snapped Quads (Yellow) vs CAD Triangles (Cyan)", color="white", fontsize=13, fontweight="bold", pad=12)
    ax4.axis("off")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#facc15", lw=2.5, label="Snapped Conformal Hex Faces (Tag 14)"),
        Line2D([0], [0], color="#00f2fe", lw=1.8, linestyle="--", label="Authentic CAD Surface (STL)"),
    ]
    ax4.legend(handles=legend_elements, loc="lower right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="#030712")
    plt.close(fig)
    print(f"Saved snapped grid-STL interaction visualization to {output_path}")


if __name__ == "__main__":
    plot_grid_stl_interaction()
