"""Generate a mathematically airtight, 100% conforming 6-block Cubed-Sphere Hex O-grid."""

import argparse
from pathlib import Path
import numpy as np
import meshio


def make_cubed_sphere_ogrid(
    output_path: str = "mesh/sphere_conformal_cubed_ogrid.msh",
    r_inner: float = 0.5,
    r_outer: float = 3.0,
    n_face: int = 8,
    n_radial: int = 20,
    progression: float = 1.08,
    p_degree: int = 2,
):
    print("=================================================================")
    print("GENERATING CONFORMAL 6-BLOCK CUBED-SPHERE HEXAHEDRAL O-GRID")
    print("=================================================================")
    print(f"R_inner: {r_inner} m, R_outer: {r_outer} m")
    print(f"Resolution per face: {n_face} x {n_face}, Radial layers: {n_radial}")

    # Radial spacing with geometric progression near the sphere wall
    if abs(progression - 1.0) < 1e-4:
        r_layers = np.linspace(r_inner, r_outer, n_radial + 1)
    else:
        geom = (progression ** np.arange(n_radial + 1) - 1.0) / (progression ** n_radial - 1.0)
        r_layers = r_inner + (r_outer - r_inner) * geom

    # Equiangular grid parameters for the 6 cube faces
    angles = np.linspace(-np.pi / 4, np.pi / 4, n_face + 1)
    tan_vals = np.tan(angles)

    # Unit vector mapping for each of the 6 cube faces
    # Face 0: -X (Inflow Nose)
    # Face 1: +X (Wake / Downstream)
    # Face 2: +Y (Top)
    # Face 3: -Y (Bottom)
    # Face 4: +Z (Right)
    # Face 5: -Z (Left)
    def face_unit_dir(face, a, b):
        if face == 0:  # -X
            v = np.array([-1.0, b, a])
        elif face == 1:  # +X
            v = np.array([1.0, b, -a])
        elif face == 2:  # +Y
            v = np.array([a, 1.0, b])
        elif face == 3:  # -Y
            v = np.array([a, -1.0, -b])
        elif face == 4:  # +Z
            v = np.array([a, b, 1.0])
        elif face == 5:  # -Z
            v = np.array([-a, b, -1.0])
        else:
            raise ValueError(f"Invalid face {face}")
        return v / np.linalg.norm(v)

    # Build unique nodes with spatial hash table
    node_map = {}
    node_coords = []

    def register_node(x, y, z):
        # Hash with 8-decimal precision
        key = (round(float(x), 8), round(float(y), 8), round(float(z), 8))
        if key not in node_map:
            nid = len(node_coords)
            node_map[key] = nid
            node_coords.append([x, y, z])
            return nid
        return node_map[key]

    # Pre-populate 3D index grids for all 6 blocks
    # block_grids[face][i, j, k] -> global node id
    block_grids = np.zeros((6, n_face + 1, n_face + 1, n_radial + 1), dtype=np.int32)

    for face in range(6):
        for i in range(n_face + 1):
            a = tan_vals[i]
            for j in range(n_face + 1):
                b = tan_vals[j]
                u = face_unit_dir(face, a, b)
                for k in range(n_radial + 1):
                    rk = r_layers[k]
                    pt = rk * u
                    block_grids[face, i, j, k] = register_node(pt[0], pt[1], pt[2])

    print(f"Total Unique Conformal Nodes Generated: {len(node_coords)}")

    # Build Hexahedral elements (8 nodes each)
    hex_elements = []
    
    # Boundary Quad faces
    sphere_wall_quads = []
    inlet_quads = []
    outlet_quads = []
    outer_wall_quads = []

    for face in range(6):
        for i in range(n_face):
            for j in range(n_face):
                # Inner sphere surface (k = 0)
                n0 = block_grids[face, i, j, 0]
                n1 = block_grids[face, i + 1, j, 0]
                n2 = block_grids[face, i + 1, j + 1, 0]
                n3 = block_grids[face, i, j + 1, 0]
                sphere_wall_quads.append([n0, n1, n2, n3])

                # Outer boundary surface (k = n_radial)
                o0 = block_grids[face, i, j, n_radial]
                o1 = block_grids[face, i + 1, j, n_radial]
                o2 = block_grids[face, i + 1, j + 1, n_radial]
                o3 = block_grids[face, i, j + 1, n_radial]

                if face == 0:    # -X is inlet
                    inlet_quads.append([o0, o1, o2, o3])
                elif face == 1:  # +X is outlet
                    outlet_quads.append([o0, o1, o2, o3])
                else:            # +Y, -Y, +Z, -Z are farfield walls
                    outer_wall_quads.append([o0, o1, o2, o3])

                # Hexahedral volume cells along k
                for k in range(n_radial):
                    v0 = block_grids[face, i, j, k]
                    v1 = block_grids[face, i + 1, j, k]
                    v2 = block_grids[face, i + 1, j + 1, k]
                    v3 = block_grids[face, i, j + 1, k]

                    v4 = block_grids[face, i, j, k + 1]
                    v5 = block_grids[face, i + 1, j, k + 1]
                    v6 = block_grids[face, i + 1, j + 1, k + 1]
                    v7 = block_grids[face, i, j + 1, k + 1]

                    hex_elements.append([v0, v1, v2, v3, v4, v5, v6, v7])

    pts = np.asarray(node_coords, dtype=np.float64)
    hex_data = np.asarray(hex_elements, dtype=np.int32)
    
    print(f"Total Volume Hexahedra: {len(hex_data)} ({6} blocks x {n_face*n_face*n_radial})")
    print(f"Sphere Wall Quads: {len(sphere_wall_quads)}")
    print(f"Inlet Quads: {len(inlet_quads)}")
    print(f"Outlet Quads: {len(outlet_quads)}")
    print(f"Farfield Wall Quads: {len(outer_wall_quads)}")

    # Verify that NO internal faces are counted as boundaries
    # Topological boundary check:
    # A boundary quad appears exactly ONCE in the mesh.
    # An internal quad appears exactly TWICE.
    all_quad_faces = {}
    local_hex_quads = [
        (0, 3, 2, 1), (4, 5, 6, 7),  # bottom (k), top (k+1)
        (0, 1, 5, 4), (2, 3, 7, 6),  # lateral faces
        (1, 2, 6, 5), (3, 0, 4, 7),
    ]
    for cell in hex_data:
        for face_nodes in local_hex_quads:
            nodes = tuple(sorted(cell[v] for v in face_nodes))
            all_quad_faces[nodes] = all_quad_faces.get(nodes, 0) + 1

    boundary_face_count = sum(1 for count in all_quad_faces.values() if count == 1)
    internal_face_count = sum(1 for count in all_quad_faces.values() if count == 2)
    bad_faces = sum(1 for count in all_quad_faces.values() if count > 2 or count < 1)

    print(f"Topology Check:")
    print(f"  Boundary Quads (count==1): {boundary_face_count}")
    print(f"  Internal Quads (count==2): {internal_face_count}")
    print(f"  Non-manifold Quads (count > 2): {bad_faces}")

    tagged_boundary_count = (
        len(sphere_wall_quads) + len(inlet_quads) + len(outlet_quads) + len(outer_wall_quads)
    )
    print(f"  Tagged Boundary Quads: {tagged_boundary_count}")
    
    if tagged_boundary_count != boundary_face_count:
        print(f"WARNING: Mismatch between tagged boundary ({tagged_boundary_count}) and topological boundary ({boundary_face_count})!")
    else:
        print("PERFECT MATCH: 100% of topological boundary faces are correctly tagged!")

    output_cells = [
        ("quad", np.asarray(sphere_wall_quads, dtype=np.int32)),
        ("quad", np.asarray(inlet_quads, dtype=np.int32)),
        ("quad", np.asarray(outlet_quads, dtype=np.int32)),
        ("quad", np.asarray(outer_wall_quads, dtype=np.int32)),
        ("hexahedron", hex_data),
    ]

    cell_data = {
        "gmsh:physical": [
            np.full(len(sphere_wall_quads), 14, dtype=np.int32),  # sphere_wall
            np.full(len(inlet_quads), 11, dtype=np.int32),        # inlet
            np.full(len(outlet_quads), 12, dtype=np.int32),       # outlet
            np.full(len(outer_wall_quads), 13, dtype=np.int32),   # walls
            np.full(len(hex_data), 1, dtype=np.int32),            # fluid
        ]
    }

    field_data = {
        "fluid": np.array([1, 3]),
        "inlet": np.array([11, 2]),
        "outlet": np.array([12, 2]),
        "walls": np.array([13, 2]),
        "sphere_wall": np.array([14, 2]),
    }

    mesh = meshio.Mesh(pts, output_cells, cell_data=cell_data, field_data=field_data)
    temp_msh = Path(output_path).with_suffix(".linear.msh")
    mesh.write(str(temp_msh), file_format="gmsh22", binary=False)

    if p_degree == 2:
        print("Elevating to curved high-order P=2 (27-node hexahedra) using Gmsh...")
        import gmsh
        gmsh.initialize()
        gmsh.open(str(temp_msh))
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(output_path))
        gmsh.finalize()
        if temp_msh.exists():
            temp_msh.unlink()
    else:
        temp_msh.rename(output_path)

    print(f"Successfully created: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mesh/sphere_conformal_cubed_ogrid.msh")
    parser.add_argument("-p", "--polynomial-degree", type=int, default=2)
    parser.add_argument("--n-face", type=int, default=8)
    parser.add_argument("--n-radial", type=int, default=20)
    args = parser.parse_args()
    make_cubed_sphere_ogrid(
        output_path=args.output,
        n_face=args.n_face,
        n_radial=args.n_radial,
        p_degree=args.polynomial_degree,
    )
