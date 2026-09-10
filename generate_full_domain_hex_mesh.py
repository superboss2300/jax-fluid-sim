"""Generate a 100% structured pure hexahedral full non-mirrored domain (Y in [-2.5, 2.5]).

Produces a 2D-extruded domain with 0 triangles, 0 tetrahedra, and 0 wedges.
Entire flow field (top and bottom) is simulated natively with no symmetry or mirroring.
"""

import subprocess
from pathlib import Path
import numpy as np
import meshio

from hybrid.mesh import _ELEMENT_ALIASES, _LOCAL_FACES


def generate_full_domain_mesh(
    output_path: str = "mesh/cylinder_full_hex.msh",
    r_cylinder: float = 0.5,
    r_far: float = 2.5,
    x_inlet: float = -2.5,
    x_outlet: float = 4.0,
    dz: float = 0.1,
    n_radial: int = 36,
    n_circ: int = 24,
    n_wake: int = 36,
    progression_rad: float = 1.05,
    p_degree: int = 2,
):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_geo = output_file.with_suffix(".temp.geo")
    temp_msh = output_file.with_suffix(".temp.msh")

    geo_content = f"""SetFactory("OpenCASCADE");

r = {r_cylinder};
r_far = {r_far};
x_in = {x_inlet};
x_out = {x_outlet};
dz = {dz};

// Center
p_c = newp; Point(p_c) = {{0, 0, 0}};

// Cylinder surface points: 0, 90, 180, 270 deg
p_cyl_r = newp; Point(p_cyl_r) = {{r, 0, 0}};
p_cyl_t = newp; Point(p_cyl_t) = {{0, r, 0}};
p_cyl_l = newp; Point(p_cyl_l) = {{-r, 0, 0}};
p_cyl_b = newp; Point(p_cyl_b) = {{0, -r, 0}};

// Farfield outer boundary points
p_far_t = newp; Point(p_far_t) = {{0, r_far, 0}};
p_far_l = newp; Point(p_far_l) = {{x_in, 0, 0}};
p_far_b = newp; Point(p_far_b) = {{0, -r_far, 0}};
p_far_tl = newp; Point(p_far_tl) = {{x_in, r_far, 0}};
p_far_bl = newp; Point(p_far_bl) = {{x_in, -r_far, 0}};

p_far_tr = newp; Point(p_far_tr) = {{r, r_far, 0}};
p_far_br = newp; Point(p_far_br) = {{r, -r_far, 0}};

p_out_t = newp; Point(p_out_t) = {{x_out, r_far, 0}};
p_out_b = newp; Point(p_out_b) = {{x_out, -r_far, 0}};
p_out_m = newp; Point(p_out_m) = {{x_out, 0, 0}};

// Cylinder arcs
c_cyl_tl = newc; Circle(c_cyl_tl) = {{p_cyl_t, p_c, p_cyl_l}};
c_cyl_bl = newc; Circle(c_cyl_bl) = {{p_cyl_l, p_c, p_cyl_b}};
c_cyl_tr = newc; Circle(c_cyl_tr) = {{p_cyl_r, p_c, p_cyl_t}};
c_cyl_br = newc; Circle(c_cyl_br) = {{p_cyl_b, p_c, p_cyl_r}};

// Farfield curves
l_in_t = newl; Line(l_in_t) = {{p_far_l, p_far_tl}};
l_top_in = newl; Line(l_top_in) = {{p_far_tl, p_far_t}};
l_in_b = newl; Line(l_in_b) = {{p_far_bl, p_far_l}};
l_bot_in = newl; Line(l_bot_in) = {{p_far_b, p_far_bl}};

// Radial separator lines
l_rad_l = newl; Line(l_rad_l) = {{p_cyl_l, p_far_l}};
l_rad_t = newl; Line(l_rad_t) = {{p_cyl_t, p_far_t}};
l_rad_b = newl; Line(l_rad_b) = {{p_cyl_b, p_far_b}};
l_rad_tr = newl; Line(l_rad_tr) = {{p_cyl_r, p_far_tr}};
l_rad_br = newl; Line(l_rad_br) = {{p_cyl_r, p_far_br}};

l_top_mid = newl; Line(l_top_mid) = {{p_far_t, p_far_tr}};
l_bot_mid = newl; Line(l_bot_mid) = {{p_far_br, p_far_b}};

// Block 1: Top-upstream
cl1 = newcl; Curve Loop(cl1) = {{l_rad_t, l_top_in, -l_in_t, -l_rad_l, -c_cyl_tl}};
s1 = news; Plane Surface(s1) = {{cl1}};

// Block 2: Bottom-upstream
cl2 = newcl; Curve Loop(cl2) = {{l_rad_l, -l_in_b, -l_bot_in, -l_rad_b, -c_cyl_bl}};
s2 = news; Plane Surface(s2) = {{cl2}};

// Block 3: Top-shoulder
cl3 = newcl; Curve Loop(cl3) = {{l_rad_tr, -l_top_mid, -l_rad_t, c_cyl_tr}};
s3 = news; Plane Surface(s3) = {{cl3}};

// Block 4: Bottom-shoulder
cl4 = newcl; Curve Loop(cl4) = {{c_cyl_br, l_rad_br, l_bot_mid, -l_rad_b}};
s4 = news; Plane Surface(s4) = {{cl4}};

// Wake: Top-wake and Bottom-wake
l_top_wake = newl; Line(l_top_wake) = {{p_far_tr, p_out_t}};
l_bot_wake = newl; Line(l_bot_wake) = {{p_far_br, p_out_b}};
l_wake_axis = newl; Line(l_wake_axis) = {{p_cyl_r, p_out_m}};
l_out_t = newl; Line(l_out_t) = {{p_out_m, p_out_t}};
l_out_b = newl; Line(l_out_b) = {{p_out_b, p_out_m}};

// Block 5: Top-wake
cl5 = newcl; Curve Loop(cl5) = {{l_rad_tr, l_top_wake, -l_out_t, -l_wake_axis}};
s5 = news; Plane Surface(s5) = {{cl5}};

// Block 6: Bottom-wake
cl6 = newcl; Curve Loop(cl6) = {{l_wake_axis, -l_out_b, -l_bot_wake, -l_rad_br}};
s6 = news; Plane Surface(s6) = {{cl6}};

n_rad = {n_radial};
n_circ = {n_circ};
n_wake = {n_wake};

// Radial progressions towards cylinder wall
Transfinite Curve {{l_rad_l, l_rad_t, l_rad_b, l_rad_tr, l_rad_br}} = n_rad Using Progression {progression_rad};
Transfinite Curve {{c_cyl_tl, l_in_t, l_top_in}} = n_circ;
Transfinite Curve {{c_cyl_bl, l_in_b, l_bot_in}} = n_circ;
Transfinite Curve {{c_cyl_tr, l_top_mid}} = n_circ;
Transfinite Curve {{c_cyl_br, l_bot_mid}} = n_circ;

Transfinite Curve {{l_wake_axis, l_top_wake, l_bot_wake}} = n_wake;
Transfinite Curve {{l_out_t, l_out_b}} = n_rad;

Transfinite Surface {{s1}}; Recombine Surface {{s1}};
Transfinite Surface {{s2}}; Recombine Surface {{s2}};
Transfinite Surface {{s3}}; Recombine Surface {{s3}};
Transfinite Surface {{s4}}; Recombine Surface {{s4}};
Transfinite Surface {{s5}}; Recombine Surface {{s5}};
Transfinite Surface {{s6}}; Recombine Surface {{s6}};

// Extrude 1 layer in z to form pure 3D hexahedra
ext[] = Extrude {{0, 0, dz}} {{ Surface{{s1, s2, s3, s4, s5, s6}}; Layers{{1}}; Recombine; }};
Coherence;

Physical Volume("fluid", 1) = {{Volume{{:}}}};

Mesh.ElementOrder = {p_degree};
Mesh.MshFileVersion = 4.1;
"""
    with open(temp_geo, "w") as f:
        f.write(geo_content)

    print("Generating full-domain pure hexahedral mesh with Gmsh...")
    cmd = ["gmsh", "-3", str(temp_geo), "-o", str(temp_msh)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Gmsh meshing failed:\n{res.stderr}")

    raw = meshio.read(str(temp_msh))
    pts = np.asarray(raw.points, dtype=np.float64)
    pts[:, 2] -= dz * 0.5

    cells_3d = []
    tags_3d = []
    for cb in raw.cells:
        alias = _ELEMENT_ALIASES.get(cb.type)
        if alias is not None and alias[0] in ("hexahedron", "wedge", "tetra"):
            cells_3d.append((cb.type, cb.data))
            tags_3d.append(np.full(len(cb.data), 1, dtype=np.int32))

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

    tol = 1.0e-3
    for elem_type, cell_idx, lf_idx, nodes in pending.values():
        face_pts = pts[list(nodes)]
        fx = face_pts[:, 0]
        fy = face_pts[:, 1]
        fz = face_pts[:, 2]
        r_sq = fx**2 + fy**2

        if np.all(np.abs(np.sqrt(r_sq) - r_cylinder) < 0.08):
            t = 14  # cylinder wall
        elif np.all(np.abs(fz - (-dz * 0.5)) < tol) or np.all(np.abs(fz - (dz * 0.5)) < tol):
            t = 13  # front/back symmetry planes
        elif np.all(np.abs(fx - x_inlet) < tol):
            t = 11  # inlet
        elif np.all(np.abs(fx - x_outlet) < tol):
            t = 12  # outlet
        elif np.all(np.abs(fy - r_far) < tol) or np.all(np.abs(fy - (-r_far)) < tol):
            t = 11  # top/bottom farfield
        else:
            t = 11

        tag_counts[t] = tag_counts.get(t, 0) + 1
        if len(nodes) == 4:
            quad_faces.append(nodes)
            quad_tags.append(t)
        else:
            tri_faces.append(nodes)
            tri_tags.append(t)

    tag_names = {11: "inlet/farfield", 12: "outlet", 13: "front/back symmetry", 14: "cylinder wall"}
    for t, cnt in sorted(tag_counts.items()):
        print(f"  Boundary Tag {t} ({tag_names.get(t, 'other')}): {cnt} faces")

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
    }

    out_mesh = meshio.Mesh(
        pts,
        output_cells,
        cell_data={"gmsh:physical": output_cell_data},
        field_data=field_data,
    )
    out_mesh.write(str(output_file), file_format="gmsh22", binary=False)
    print(f"Successfully generated full-domain mesh: {output_file}")
    print(f"  Total elements: {sum(len(b[1]) for b in cells_3d)} ({cells_3d[0][0]})")
    temp_geo.unlink(missing_ok=True)
    temp_msh.unlink(missing_ok=True)
    return output_file


if __name__ == "__main__":
    generate_full_domain_mesh()
