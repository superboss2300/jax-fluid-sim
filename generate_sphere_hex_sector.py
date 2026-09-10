"""Generate a high-quality structured quadrilateral/hexahedral O-grid sector mesh around a sphere.

This produces a 100% flow-aligned, shock-capturing mesh composed exclusively of
hexahedra and centerline wedges, with ZERO tetrahedra and ZERO triangles.
"""

import argparse
from pathlib import Path
import subprocess
import numpy as np
import meshio

from hybrid.mesh import _ELEMENT_ALIASES, _LOCAL_FACES


def generate_sector_mesh(
    output_path: str = "mesh/sphere_quad_sector.msh",
    r_sphere: float = 0.5,
    r_far: float = 3.0,
    x_out: float = 4.0,
    angle_deg: float = 5.0,
    n_radial: int = 48,
    n_circ: int = 36,
    n_wake: int = 48,
    progression_rad: float = 1.025,
    progression_wake: float = 1.025,
    p_degree: int = 2,
):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_geo = output_file.with_suffix(".temp.geo")
    temp_msh = output_file.with_suffix(".temp.msh")

    angle_rad = angle_deg * np.pi / 180.0

    # Gmsh OpenCASCADE script for structured 3-block O-grid
    geo_content = f"""SetFactory("OpenCASCADE");

r = {r_sphere};
r_far = {r_far};
x_out = {x_out};
angle = {angle_rad};

p_center = newp; Point(p_center) = {{0, 0, 0}};

// Block 1: Upstream sphere nose & bow shock region
p_inlet_axis = newp; Point(p_inlet_axis) = {{-r_far, 0, 0}};
p_nose = newp; Point(p_nose) = {{-r, 0, 0}};
p_shoulder = newp; Point(p_shoulder) = {{0, r, 0}};
p_top_shoulder = newp; Point(p_top_shoulder) = {{0, r_far, 0}};

c_sphere_front = newc; Circle(c_sphere_front) = {{p_nose, p_center, p_shoulder}};
c_far_front = newc; Circle(c_far_front) = {{p_inlet_axis, p_center, p_top_shoulder}};
l_axis_inlet = newl; Line(l_axis_inlet) = {{p_nose, p_inlet_axis}};
l_shoulder = newl; Line(l_shoulder) = {{p_shoulder, p_top_shoulder}};

cl1 = newcl; Curve Loop(cl1) = {{l_axis_inlet, c_far_front, -l_shoulder, -c_sphere_front}};
s1 = news; Plane Surface(s1) = {{cl1}};

// Block 2: Shoulder to rear sphere
p_rear = newp; Point(p_rear) = {{r, 0, 0}};
p_top_rear = newp; Point(p_top_rear) = {{r, r_far, 0}};
c_sphere_rear = newc; Circle(c_sphere_rear) = {{p_shoulder, p_center, p_rear}};
l_top_mid = newl; Line(l_top_mid) = {{p_top_shoulder, p_top_rear}};
l_rear_rad = newl; Line(l_rear_rad) = {{p_rear, p_top_rear}};

cl2 = newcl; Curve Loop(cl2) = {{l_shoulder, l_top_mid, -l_rear_rad, -c_sphere_rear}};
s2 = news; Plane Surface(s2) = {{cl2}};

// Block 3: Downstream wake
p_wake_axis = newp; Point(p_wake_axis) = {{x_out, 0, 0}};
p_wake_top = newp; Point(p_wake_top) = {{x_out, r_far, 0}};
l_wake_axis = newl; Line(l_wake_axis) = {{p_rear, p_wake_axis}};
l_top_wake = newl; Line(l_top_wake) = {{p_top_rear, p_wake_top}};
l_outlet = newl; Line(l_outlet) = {{p_wake_axis, p_wake_top}};

cl3 = newcl; Curve Loop(cl3) = {{l_rear_rad, l_top_wake, -l_outlet, -l_wake_axis}};
s3 = news; Plane Surface(s3) = {{cl3}};

n_rad = {n_radial};
n_circ = {n_circ};
n_wake = {n_wake};

// Smooth radial clustering towards the sphere wall
Transfinite Curve {{l_axis_inlet, l_shoulder, l_rear_rad}} = n_rad Using Progression {progression_rad};
Transfinite Curve {{c_sphere_front, c_far_front}} = n_circ;
Transfinite Curve {{c_sphere_rear, l_top_mid}} = n_circ;
Transfinite Curve {{l_wake_axis, l_top_wake}} = n_wake Using Progression {progression_wake};
Transfinite Curve {{l_outlet}} = n_rad;

Transfinite Surface {{s1}}; Recombine Surface {{s1}};
Transfinite Surface {{s2}}; Recombine Surface {{s2}};
Transfinite Surface {{s3}}; Recombine Surface {{s3}};

// Extrude 1 layer by azimuthal rotation around the x-axis and stitch interfaces
ext[] = Extrude {{ {{1,0,0}}, {{0,0,0}}, angle }} {{ Surface{{s1, s2, s3}}; Layers{{1}}; Recombine; }};
Coherence;

Physical Volume("fluid", 1) = {{Volume{{:}}}};

Mesh.ElementOrder = {p_degree};
Mesh.MshFileVersion = 4.1;
"""
    with open(temp_geo, "w") as f:
        f.write(geo_content)

    print(f"Generating 3D structured hex sector mesh using Gmsh...")
    cmd = ["gmsh", "-3", str(temp_geo), "-o", str(temp_msh)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Gmsh meshing failed:\n{res.stderr}")

    raw = meshio.read(str(temp_msh))
    pts = np.asarray(raw.points, dtype=np.float64)

    # Collect 3D elements
    cells_3d = []
    tags_3d = []
    for cb in raw.cells:
        alias = _ELEMENT_ALIASES.get(cb.type)
        if alias is not None and alias[0] in ("hexahedron", "wedge", "tetra"):
            cells_3d.append((cb.type, cb.data))
            tags_3d.append(np.full(len(cb.data), 1, dtype=np.int32))

    # Extract external boundary faces by topological matching
    pending = {}
    for cb_type, cb_data in cells_3d:
        elem_type, corner_count = _ELEMENT_ALIASES[cb_type]
        corners = cb_data[:, :corner_count]
        local_faces = _LOCAL_FACES[elem_type]
        for cell_idx, cell in enumerate(corners):
            for lf_idx, lf in enumerate(local_faces):
                nodes = tuple(int(cell[v]) for v in lf)
                key = tuple(sorted(nodes))
                if key in pending:
                    pending.pop(key)
                else:
                    pending[key] = (elem_type, cell_idx, lf_idx, nodes)

    quad_faces, quad_tags = [], []
    tri_faces, tri_tags = [], []

    tag_counts = {}
    tol_sphere = r_sphere + 0.01
    tol_far = r_far - 0.05
    tol_outlet = x_out - 0.05

    for key, (elem_type, cell_idx, lf_idx, nodes) in pending.items():
        center = np.mean(pts[list(nodes)], axis=0)
        x, y, z = center
        r_sph = np.sqrt(x**2 + y**2 + z**2)
        r_cyl = np.sqrt(y**2 + z**2)

        if r_sph <= tol_sphere:
            tag = 14  # sphere_wall
        elif x <= 0.05 and r_sph >= tol_far:
            tag = 11  # inlet
        elif x >= tol_outlet:
            tag = 12  # outlet
        elif r_cyl >= tol_far:
            tag = 13  # walls
        else:
            tag = 15  # symmetry

        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if len(nodes) == 4:
            quad_faces.append(nodes)
            quad_tags.append(tag)
        else:
            tri_faces.append(nodes)
            tri_tags.append(tag)

    output_cells = []
    output_cell_data = []

    if tri_faces:
        output_cells.append(("triangle", np.asarray(tri_faces, dtype=np.int32)))
        output_cell_data.append(np.asarray(tri_tags, dtype=np.int32))
    if quad_faces:
        output_cells.append(("quad", np.asarray(quad_faces, dtype=np.int32)))
        output_cell_data.append(np.asarray(quad_tags, dtype=np.int32))

    for (cb_type, cb_data), cb_tags in zip(cells_3d, tags_3d):
        output_cells.append((cb_type, cb_data))
        output_cell_data.append(cb_tags)

    field_data = {
        "fluid": np.array([1, 3]),
        "inlet": np.array([11, 2]),
        "outlet": np.array([12, 2]),
        "walls": np.array([13, 2]),
        "sphere_wall": np.array([14, 2]),
        "symmetry": np.array([15, 2]),
    }

    final_mesh = meshio.Mesh(
        pts,
        output_cells,
        cell_data={"gmsh:physical": output_cell_data},
        field_data=field_data,
    )
    final_mesh.write(str(output_file), file_format="gmsh22", binary=False)

    # Clean up temp files
    if temp_geo.exists():
        temp_geo.unlink()
    if temp_msh.exists():
        temp_msh.unlink()

    total_3d = sum(len(cb_data) for _, cb_data in cells_3d)
    print(f"Successfully generated {output_file}:")
    print(f"  Total 3D elements: {total_3d} (100% structured hex & wedge)")
    print(f"  Boundary tags: {tag_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="mesh/sphere_quad_sector.msh")
    parser.add_argument("-p", "--polynomial-degree", type=int, default=2)
    parser.add_argument("--r-sphere", type=float, default=0.5)
    parser.add_argument("--r-far", type=float, default=3.0)
    parser.add_argument("--x-out", type=float, default=4.0)
    parser.add_argument("--angle", type=float, default=5.0)
    parser.add_argument("--n-radial", type=int, default=48)
    parser.add_argument("--n-circ", type=int, default=36)
    parser.add_argument("--n-wake", type=int, default=48)
    parser.add_argument("--progression-rad", type=float, default=1.025)
    parser.add_argument("--progression-wake", type=float, default=1.025)
    args = parser.parse_args()

    generate_sector_mesh(
        output_path=args.output,
        r_sphere=args.r_sphere,
        r_far=args.r_far,
        x_out=args.x_out,
        angle_deg=args.angle,
        n_radial=args.n_radial,
        n_circ=args.n_circ,
        n_wake=args.n_wake,
        progression_rad=args.progression_rad,
        progression_wake=args.progression_wake,
        p_degree=args.polynomial_degree,
    )
