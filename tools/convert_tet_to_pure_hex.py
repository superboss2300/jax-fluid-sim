"""Convert a tetrahedral 3D mesh into a pure hexahedral mesh via dual barycentric subdivision.

Each tetrahedron is subdivided into 4 topological hexahedra.
Each triangular boundary face is subdivided into 3 quadrilateral boundary faces.
Preserves all boundary tags and field data.
"""

import sys
import time
import numpy as np
import meshio

def tet_to_pure_hex(
    input_msh: str = "mesh/sr71_pristine.msh",
    output_msh: str = "mesh/sr71_pure_hex.msh"
):
    print("=" * 76)
    print("CONVERTING TETRAHEDRAL MESH TO PURE HEXAHEDRA (BARYCENTRIC SUBDIVISION)")
    print(f"  Input MSH:   {input_msh}")
    print(f"  Output MSH:  {output_msh}")
    print("=" * 76)

    t0 = time.perf_counter()
    raw = meshio.read(input_msh)
    pts = raw.points
    tets = raw.cells_dict["tetra"]
    tris = raw.cells_dict["triangle"]
    tri_tags = raw.cell_data_dict["gmsh:physical"]["triangle"]

    Nv = len(pts)
    print(f"Source: {Nv} vertices, {len(tets)} tetrahedra, {len(tris)} triangles.")

    # 1. Edge dictionary & midpoint coordinates
    edge_dict = {}
    edge_pts = []
    edges_all = np.vstack([
        tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
        tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]]
    ])
    for i, j in edges_all:
        key = (min(i, j), max(i, j))
        if key not in edge_dict:
            edge_dict[key] = Nv + len(edge_pts)
            edge_pts.append(0.5 * (pts[key[0]] + pts[key[1]]))

    Ne = len(edge_pts)
    print(f"Subdivided {Ne} unique edges.")

    # 2. Face dictionary & centroid coordinates
    face_dict = {}
    face_pts = []
    faces_all = np.vstack([
        tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
        tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]
    ])
    for i, j, k in faces_all:
        key = tuple(sorted([i, j, k]))
        if key not in face_dict:
            face_dict[key] = Nv + Ne + len(face_pts)
            face_pts.append((pts[key[0]] + pts[key[1]] + pts[key[2]]) / 3.0)

    Nf = len(face_pts)
    print(f"Subdivided {Nf} unique triangular faces.")

    # 3. Cell centroids
    cell_pts = [np.mean(pts[t], axis=0) for t in tets]
    cell_node_base = Nv + Ne + Nf

    all_pts = np.vstack([pts, np.array(edge_pts), np.array(face_pts), np.array(cell_pts)])
    print(f"Total pure conformal hex vertices: {len(all_pts)}")

    # 4. Construct 4 hexahedra per tetrahedron
    hex_cells = []
    for tet_idx, tet in enumerate(tets):
        v0, v1, v2, v3 = tet
        e01 = edge_dict[(min(v0, v1), max(v0, v1))]
        e02 = edge_dict[(min(v0, v2), max(v0, v2))]
        e03 = edge_dict[(min(v0, v3), max(v0, v3))]
        e12 = edge_dict[(min(v1, v2), max(v1, v2))]
        e13 = edge_dict[(min(v1, v3), max(v1, v3))]
        e23 = edge_dict[(min(v2, v3), max(v2, v3))]

        f012 = face_dict[tuple(sorted([v0, v1, v2]))]
        f013 = face_dict[tuple(sorted([v0, v1, v3]))]
        f023 = face_dict[tuple(sorted([v0, v2, v3]))]
        f123 = face_dict[tuple(sorted([v1, v2, v3]))]

        c = cell_node_base + tet_idx

        # Hex node ordering for positive determinant
        hex_cells.append([v0, e01, f012, e02, e03, f013, c, f023])
        hex_cells.append([v1, e12, f012, e01, e13, f123, c, f013])
        hex_cells.append([v2, e02, f012, e12, e23, f023, c, f123])
        hex_cells.append([v3, e03, f023, e23, e13, f013, c, f123])

    hex_cells = np.array(hex_cells)
    print(f"Generated {len(hex_cells)} pure hexahedra!")

    # 5. Construct 3 quadrilaterals per boundary triangle
    quad_cells = []
    quad_tags = []
    for tri, tag in zip(tris, tri_tags):
        v0, v1, v2 = tri
        e01 = edge_dict[(min(v0, v1), max(v0, v1))]
        e12 = edge_dict[(min(v1, v2), max(v1, v2))]
        e02 = edge_dict[(min(v0, v2), max(v0, v2))]
        f012 = face_dict[tuple(sorted([v0, v1, v2]))]

        quad_cells.append([v0, e01, f012, e02])
        quad_cells.append([v1, e12, f012, e01])
        quad_cells.append([v2, e02, f012, e12])
        quad_tags.extend([tag, tag, tag])

    quad_cells = np.array(quad_cells)
    quad_tags = np.array(quad_tags)
    print(f"Generated {len(quad_cells)} pure boundary quadrilaterals!")

    # Verify orientation / Jacobian sign
    p_h = all_pts[hex_cells]
    v1_vec = p_h[:, 1] - p_h[:, 0]
    v3_vec = p_h[:, 3] - p_h[:, 0]
    v4_vec = p_h[:, 4] - p_h[:, 0]
    triple = np.sum(np.cross(v1_vec, v3_vec) * v4_vec, axis=-1)
    neg_count = np.sum(triple <= 0)
    print(f"Orientation Check: min triple={triple.min():.6e}, max triple={triple.max():.6e}, inverted elements={neg_count}")

    # Write Gmsh 2.2 format
    out_mesh = meshio.Mesh(
        points=all_pts,
        cells=[("quad", quad_cells), ("hexahedron", hex_cells)],
        cell_data={"gmsh:physical": [quad_tags, np.full(len(hex_cells), 1, dtype=np.int32)]},
        field_data=raw.field_data,
    )
    out_mesh.write(output_msh, file_format="gmsh22")
    print(f"Mesh written successfully in {time.perf_counter() - t0:.2f}s to {output_msh}")

if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else "mesh/sr71_pristine.msh"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "mesh/sr71_pure_hex.msh"
    tet_to_pure_hex(in_file, out_file)

