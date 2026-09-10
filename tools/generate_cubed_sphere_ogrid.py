"""Generate a 100% pure hexahedral Cubed-Sphere O-grid for supersonic flow past a sphere."""

import argparse
from pathlib import Path
import numpy as np
import meshio


def equiangular_grid(n):
    """Equiangular coordinates in [-1, 1] to minimize distortion at corners."""
    theta = np.linspace(-np.pi / 4, np.pi / 4, n + 1)
    return np.tan(theta)


def get_face_unit_vectors(face_id, xi, eta):
    """Map 2D local face coordinates (xi, eta) to 3D unit radial vectors on the unit sphere."""
    if face_id == 0:    # -X (Nose / Upstream)
        v = np.array([-1.0, eta, xi])
    elif face_id == 1:  # +X (Wake / Downstream)
        v = np.array([1.0, eta, -xi])
    elif face_id == 2:  # +Y (Top)
        v = np.array([xi, 1.0, eta])
    elif face_id == 3:  # -Y (Bottom)
        v = np.array([xi, -1.0, -eta])
    elif face_id == 4:  # +Z (Right)
        v = np.array([xi, eta, 1.0])
    elif face_id == 5:  # -Z (Left)
        v = np.array([-xi, eta, -1.0])
    else:
        raise ValueError(f"Invalid face_id {face_id}")
    
    # Normalize to lie on the sphere
    norm = np.linalg.norm(v)
    return v / norm


def generate_cubed_sphere_mesh(
    output_path: str = "mesh/sphere_full_cubed_ogrid.msh",
    r_sphere: float = 0.5,
    r_far: float = 3.0,
    x_outlet: float = 4.0,
    n_face: int = 10,
    n_radial: int = 24,
    progression: float = 1.08,
    p_degree: int = 2,
):
    print(f"Generating full 360° Cubed-Sphere Hex O-Grid...")
    print(f"  Sphere Radius: {r_sphere} m, Farfield Radius: {r_far} m, Outlet: {x_outlet} m")
    print(f"  Face grid: {n_face}x{n_face}, Radial layers: {n_radial}, Progression: {progression}")
    
    # Radial coordinates with smooth geometric clustering near the sphere wall
    if abs(progression - 1.0) < 1e-4:
        r_layers = np.linspace(r_sphere, r_far, n_radial + 1)
    else:
        geom = (progression ** np.arange(n_radial + 1) - 1.0) / (progression ** n_radial - 1.0)
        r_layers = r_sphere + (r_far - r_sphere) * geom

    xi_coords = equiangular_grid(n_face)
    eta_coords = equiangular_grid(n_face)

    # Node hash map to ensure 100% conforming shared vertices between adjacent faces
    node_map = {}
    node_list = []

    def get_node_id(pt):
        # Round to 7 decimals for topological stitching
        key = (round(pt[0], 7), round(pt[1], 7), round(pt[2], 7))
        if key not in node_map:
            node_id = len(node_list)
            node_map[key] = node_id
            node_list.append(pt)
            return node_id
        return node_map[key]

    # For P=1 linear hexes first, or P=2 quadratic hexes
    # We will build structured grid indices for each of the 6 faces
    # Face shape: (n_face + 1, n_face + 1, n_radial + 1)
    face_grids = []
    
    for face_id in range(6):
        grid = np.zeros((n_face + 1, n_face + 1, n_radial + 1), dtype=np.int32)
        for i in range(n_face + 1):
            xi = xi_coords[i]
            for j in range(n_face + 1):
                eta = eta_coords[j]
                u = get_face_unit_vectors(face_id, xi, eta)
                
                for k in range(n_radial + 1):
                    rk = r_layers[k]
                    pt = rk * u
                    
                    # For the downstream wake block (face_id == 1, +X), smoothly stretch the outer layers toward x_outlet
                    if face_id == 1 and k > 0:
                        frac_r = (rk - r_sphere) / (r_far - r_sphere)
                        pt[0] = pt[0] + (x_outlet - r_far) * (frac_r ** 1.5)
                        
                    grid[i, j, k] = get_node_id(pt)
        face_grids.append(grid)

    # Now construct hexahedral elements (8 corner nodes per hex)
    hex_elements = []
    
    # Also collect boundary quad faces
    sphere_wall_quads = []
    inlet_quads = []
    outlet_quads = []
    outer_wall_quads = []

    for face_id in range(6):
        grid = face_grids[face_id]
        for i in range(n_face):
            for j in range(n_face):
                # Inner sphere surface quad (k = 0)
                n00 = grid[i, j, 0]
                n10 = grid[i + 1, j, 0]
                n11 = grid[i + 1, j + 1, 0]
                n01 = grid[i, j + 1, 0]
                sphere_wall_quads.append([n00, n10, n11, n01])

                # Outer farfield quad (k = n_radial)
                o00 = grid[i, j, n_radial]
                o10 = grid[i + 1, j, n_radial]
                o11 = grid[i + 1, j + 1, n_radial]
                o01 = grid[i, j + 1, n_radial]
                
                if face_id == 0:    # -X is the inlet
                    inlet_quads.append([o00, o10, o11, o01])
                elif face_id == 1:  # +X is the outlet
                    outlet_quads.append([o00, o10, o11, o01])
                else:               # Top, bottom, left, right are farfield walls
                    outer_wall_quads.append([o00, o10, o11, o01])

                # Volume hexes along radial direction k
                for k in range(n_radial):
                    v0 = grid[i, j, k]
                    v1 = grid[i + 1, j, k]
                    v2 = grid[i + 1, j + 1, k]
                    v3 = grid[i, j + 1, k]
                    
                    v4 = grid[i, j, k + 1]
                    v5 = grid[i + 1, j, k + 1]
                    v6 = grid[i + 1, j + 1, k + 1]
                    v7 = grid[i, j + 1, k + 1]
                    
                    hex_elements.append([v0, v1, v2, v3, v4, v5, v6, v7])

    pts = np.asarray(node_list, dtype=np.float64)
    hex_data = np.asarray(hex_elements, dtype=np.int32)
    
    # If P=2, convert linear hex8 to quadratic hex27 (or gmsh high-order)
    # Let's save as gmsh format via meshio or Gmsh API
    print(f"Total Unique Nodes: {len(pts)}")
    print(f"Total Conformal Hexahedra: {len(hex_data)}")
    print(f"Sphere Wall Quads: {len(sphere_wall_quads)}")
    print(f"Inlet Quads: {len(inlet_quads)}")
    print(f"Outlet Quads: {len(outlet_quads)}")
    print(f"Farfield Wall Quads: {len(outer_wall_quads)}")

    output_cells = [
        ("quad", np.asarray(sphere_wall_quads, dtype=np.int32)),
        ("quad", np.asarray(inlet_quads, dtype=np.int32)),
        ("quad", np.asarray(outlet_quads, dtype=np.int32)),
        ("quad", np.asarray(outer_wall_quads, dtype=np.int32)),
        ("hexahedron", hex_data),
    ]
    
    cell_data = {
        "gmsh:physical": [
            np.full(len(sphere_wall_quads), 14, dtype=np.int32),  # sphere_wall = 14
            np.full(len(inlet_quads), 11, dtype=np.int32),        # inlet = 11
            np.full(len(outlet_quads), 12, dtype=np.int32),       # outlet = 12
            np.full(len(outer_wall_quads), 13, dtype=np.int32),   # walls = 13
            np.full(len(hex_data), 1, dtype=np.int32),            # fluid = 1
        ]
    }
    
    field_data = {
        "fluid": np.array([1, 3]),
        "inlet": np.array([11, 2]),
        "outlet": np.array([12, 2]),
        "walls": np.array([13, 2]),
        "sphere_wall": np.array([14, 2]),
    }

    mesh = meshio.Mesh(
        pts,
        output_cells,
        cell_data=cell_data,
        field_data=field_data,
    )
    
    # Write linear mesh first
    temp_msh = Path(output_path).with_suffix(".linear.msh")
    mesh.write(str(temp_msh), file_format="gmsh22", binary=False)
    
    if p_degree == 2:
        print("Upgrading to curved P=2 quadratic hexes using Gmsh...")
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

    print(f"Successfully generated 3D Cubed-Sphere Hex O-Grid: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mesh/sphere_cubed_ogrid.msh")
    parser.add_argument("-p", "--polynomial-degree", type=int, default=2)
    parser.add_argument("--n-face", type=int, default=8)
    parser.add_argument("--n-radial", type=int, default=20)
    args = parser.parse_args()
    
    generate_cubed_sphere_mesh(
        output_path=args.output,
        n_face=args.n_face,
        n_radial=args.n_radial,
        p_degree=args.polynomial_degree,
    )
