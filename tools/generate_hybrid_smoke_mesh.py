"""Generate three connected cells for hybrid-topology regression tests."""

from pathlib import Path

import meshio
import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "mesh" / "hybrid_smoke.msh"


def generate(output=OUTPUT):
    # A prism shares one triangular face with a tetrahedron and one quad face
    # with a hexahedron. This exercises both unlike-element interfaces.
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # 0 prism/hex
            [1.0, 0.0, 0.0],  # 1 prism/hex
            [0.0, 1.0, 0.0],  # 2 prism
            [0.0, 0.0, 1.0],  # 3 prism/hex/tet
            [1.0, 0.0, 1.0],  # 4 prism/hex/tet
            [0.0, 1.0, 1.0],  # 5 prism/tet
            [0.0, 0.0, 2.0],  # 6 tetra apex
            [1.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=float,
    )

    volume_cells = [
        ("hexahedron", np.array([[0, 1, 7, 8, 3, 4, 9, 10]])),
        ("wedge", np.array([[0, 1, 2, 3, 4, 5]])),
        ("tetra", np.array([[3, 4, 5, 6]])),
    ]

    face_templates = {
        "hexahedron": (
            (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        ),
        "wedge": (
            (0, 2, 1), (3, 4, 5), (0, 1, 4, 3),
            (1, 2, 5, 4), (2, 0, 3, 5),
        ),
        "tetra": ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
    }

    occurrences = {}
    for cell_type, cells in volume_cells:
        for cell in cells:
            for local_face in face_templates[cell_type]:
                nodes = tuple(int(cell[index]) for index in local_face)
                key = tuple(sorted(nodes))
                occurrences.setdefault(key, []).append(nodes)

    triangles = []
    quads = []
    for records in occurrences.values():
        if len(records) != 1:
            continue
        nodes = records[0]
        (triangles if len(nodes) == 3 else quads).append(nodes)

    cells = volume_cells + [
        ("triangle", np.asarray(triangles, dtype=np.int32)),
        ("quad", np.asarray(quads, dtype=np.int32)),
    ]
    physical = [
        np.full(len(block), 1 if index < 3 else 13, dtype=np.int32)
        for index, (_, block) in enumerate(cells)
    ]
    geometrical = [array.copy() for array in physical]
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data={
            "gmsh:physical": physical,
            "gmsh:geometrical": geometrical,
        },
        field_data={
            "fluid": np.array([1, 3]),
            "walls": np.array([13, 2]),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Gmsh 2.2 does not require CAD entity metadata for this synthetic mesh.
    meshio.write(output, mesh, file_format="gmsh22", binary=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    generate()
