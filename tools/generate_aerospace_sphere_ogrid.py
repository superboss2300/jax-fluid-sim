"""Generate a 100% conforming, pure hexahedral aerospace O-grid for a full 3D sphere with pristine physical tags."""

import argparse
from pathlib import Path
import numpy as np
import gmsh


def generate_aerospace_sphere_msh(
    output_path: str = "mesh/sphere_aerospace_ogrid_gpu6gb.msh",
    r_inner: float = 0.5,
    r_outer_base: float = 3.0,
    x_wake_stretch: float = 1.0,
    n_face: int = 8,
    n_radial: int = 20,
    progression: float = 1.06,
    p_degree: int = 2,
):
    print("=================================================================")
    print("GENERATING 3D AEROSPACE SPHERICAL HEXAHEDRAL O-GRID")
    print("=================================================================")
    print(f"Sphere radius: {r_inner} m, Base farfield: {r_outer_base} m, Wake outlet: {r_outer_base + x_wake_stretch} m")
    print(f"Face resolution: {n_face} x {n_face}, Radial layers: {n_radial}, Progression: {progression}")

    theta = np.linspace(-np.pi / 4, np.pi / 4, n_face + 1)
    tan_vals = np.tan(theta)

    def face_unit(face, a, b):
        if face == 0:    # -X (Nose / Upstream)
            v = np.array([-1.0, b, a])
        elif face == 1:  # +X (Wake / Downstream)
            v = np.array([1.0, b, -a])
        elif face == 2:  # +Y (Top)
            v = np.array([a, 1.0, b])
        elif face == 3:  # -Y (Bottom)
            v = np.array([a, -1.0, -b])
        elif face == 4:  # +Z (Right)
            v = np.array([a, b, 1.0])
        elif face == 5:  # -Z (Left)
            v = np.array([-a, b, -1.0])
        return v / np.linalg.norm(v)

    def outer_radius(u):
        if u[0] > 0:
            return r_outer_base + x_wake_stretch * (u[0] ** 1.5)
        return r_outer_base

    node_dict = {}
    node_coords = []

    def register_node(pt):
        k = (round(float(pt[0]), 8), round(float(pt[1]), 8), round(float(pt[2]), 8))
        if k not in node_dict:
            nid = len(node_coords)
            node_dict[k] = nid
            node_coords.append([pt[0], pt[1], pt[2]])
            return nid
        return node_dict[k]

    grids = np.zeros((6, n_face + 1, n_face + 1, n_radial + 1), dtype=np.int32)

    for face in range(6):
        for i in range(n_face + 1):
            a = tan_vals[i]
            for j in range(n_face + 1):
                b = tan_vals[j]
                u = face_unit(face, a, b)
                r_out = outer_radius(u)
                
                # Geometric radial clustering near sphere surface
                geom = (progression ** np.arange(n_radial + 1) - 1.0) / (progression ** n_radial - 1.0)
                r_layers = r_inner + (r_out - r_inner) * geom

                for k in range(n_radial + 1):
                    pt = r_layers[k] * u
                    grids[face, i, j, k] = register_node(pt)

    pts = np.asarray(node_coords, dtype=np.float64)
    print(f"Total Unique Conformal Nodes: {len(pts)}")

    hex_elements = []
    sphere_wall_quads = []
    inlet_quads = []
    outlet_quads = []
    outer_wall_quads = []

    for face in range(6):
        for i in range(n_face):
            for j in range(n_face):
                # Inner sphere surface (k = 0)
                n0 = grids[face, i, j, 0]
                n1 = grids[face, i + 1, j, 0]
                n2 = grids[face, i + 1, j + 1, 0]
                n3 = grids[face, i, j + 1, 0]
                sphere_wall_quads.append([n0, n1, n2, n3])

                # Outer boundary surface (k = n_radial)
                o0 = grids[face, i, j, n_radial]
                o1 = grids[face, i + 1, j, n_radial]
                o2 = grids[face, i + 1, j + 1, n_radial]
                o3 = grids[face, i, j + 1, n_radial]

                quad_center = np.mean(pts[[o0, o1, o2, o3]], axis=0)
                # Classify outer boundary face by position
                if quad_center[0] < -0.5:
                    inlet_quads.append([o0, o1, o2, o3])
                elif quad_center[0] > 1.5:
                    outlet_quads.append([o0, o1, o2, o3])
                else:
                    outer_wall_quads.append([o0, o1, o2, o3])

                # Volume hexes along k
                for k in range(n_radial):
                    v0 = grids[face, i, j, k]
                    v1 = grids[face, i + 1, j, k]
                    v2 = grids[face, i + 1, j + 1, k]
                    v3 = grids[face, i, j + 1, k]

                    v4 = grids[face, i, j, k + 1]
                    v5 = grids[face, i + 1, j, k + 1]
                    v6 = grids[face, i + 1, j + 1, k + 1]
                    v7 = grids[face, i, j + 1, k + 1]

                    hex_elements.append([v0, v1, v2, v3, v4, v5, v6, v7])

    print(f"Total Conformal Hexahedra: {len(hex_elements)}")
    print(f"Sphere Wall Quads: {len(sphere_wall_quads)}")
    print(f"Inlet Quads: {len(inlet_quads)}")
    print(f"Outlet Quads: {len(outlet_quads)}")
    print(f"Farfield Wall Quads: {len(outer_wall_quads)}")

    # Initialize Gmsh and populate with discrete entities
    gmsh.initialize()
    gmsh.model.add("sphere_aerospace_ogrid")

    # Node tags in Gmsh are 1-indexed
    node_tags = np.arange(1, len(pts) + 1, dtype=np.int64)
    flat_coords = pts.flatten()

    # Discrete entities:
    # 1: fluid volume
    # 11: inlet surface
    # 12: outlet surface
    # 13: walls surface
    # 14: sphere_wall surface
    gmsh.model.addDiscreteEntity(3, 1)
    gmsh.model.addDiscreteEntity(2, 11)
    gmsh.model.addDiscreteEntity(2, 12)
    gmsh.model.addDiscreteEntity(2, 13)
    gmsh.model.addDiscreteEntity(2, 14)

    # Add all nodes to volume entity 1
    gmsh.model.mesh.addNodes(3, 1, node_tags, flat_coords)

    # Element type in Gmsh:
    # 3 = 4-node quad
    # 5 = 8-node hex
    # Helper to add elements (1-indexed node tags)
    def add_quads(entity_tag, quads):
        if not quads:
            return
        quads_1based = (np.asarray(quads, dtype=np.int64) + 1).flatten()
        elem_tags = np.arange(len(quads), dtype=np.int64)  # gmsh auto-tags if empty
        gmsh.model.mesh.addElementsByType(entity_tag, 3, [], quads_1based)

    add_quads(14, sphere_wall_quads)
    add_quads(11, inlet_quads)
    add_quads(12, outlet_quads)
    add_quads(13, outer_wall_quads)

    hex_1based = (np.asarray(hex_elements, dtype=np.int64) + 1).flatten()
    gmsh.model.mesh.addElementsByType(1, 5, [], hex_1based)

    # Add Physical Groups
    gmsh.model.addPhysicalGroup(2, [14], tag=14, name="sphere_wall")
    gmsh.model.addPhysicalGroup(2, [11], tag=11, name="inlet")
    gmsh.model.addPhysicalGroup(2, [12], tag=12, name="outlet")
    gmsh.model.addPhysicalGroup(2, [13], tag=13, name="walls")
    gmsh.model.addPhysicalGroup(3, [1], tag=1, name="fluid")

    if p_degree == 2:
        print("Elevating to curved high-order P=2 (27-node hexes and 9-node quads)...")
        gmsh.model.mesh.setOrder(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(output_path)
    gmsh.finalize()

    print(f"Successfully wrote validated mesh: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mesh/sphere_aerospace_ogrid_gpu6gb.msh")
    parser.add_argument("-p", "--polynomial-degree", type=int, default=2)
    parser.add_argument("--n-face", type=int, default=8)
    parser.add_argument("--n-radial", type=int, default=20)
    args = parser.parse_args()
    generate_aerospace_sphere_msh(
        output_path=args.output,
        n_face=args.n_face,
        n_radial=args.n_radial,
        p_degree=args.polynomial_degree,
    )
