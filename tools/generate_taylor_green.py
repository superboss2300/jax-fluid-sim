"""Generate a triply periodic tetrahedral box for Taylor--Green vortex tests."""

from pathlib import Path

import gmsh


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "mesh"
MESH_PATH = OUTPUT_DIR / "taylor_green.msh"

DOMAIN_LENGTH = 2.0 * 3.141592653589793
MESH_SIZE = DOMAIN_LENGTH / 50.0


def _translation(x=0.0, y=0.0, z=0.0):
    """Return Gmsh's row-major 4x4 affine translation matrix."""
    return [
        1.0, 0.0, 0.0, x,
        0.0, 1.0, 0.0, y,
        0.0, 0.0, 1.0, z,
        0.0, 0.0, 0.0, 1.0,
    ]


def _classify_surfaces(volume_tag):
    groups = {
        "periodic_x_minus": [],
        "periodic_x_plus": [],
        "periodic_y_minus": [],
        "periodic_y_plus": [],
        "periodic_z_minus": [],
        "periodic_z_plus": [],
    }

    tolerance = 1.0e-8
    for dimension, tag in gmsh.model.getBoundary(
        [(3, volume_tag)],
        oriented=False,
        recursive=False,
    ):
        if dimension != 2:
            continue

        x, y, z = gmsh.model.occ.getCenterOfMass(dimension, tag)
        if abs(x) < tolerance:
            groups["periodic_x_minus"].append(tag)
        elif abs(x - DOMAIN_LENGTH) < tolerance:
            groups["periodic_x_plus"].append(tag)
        elif abs(y) < tolerance:
            groups["periodic_y_minus"].append(tag)
        elif abs(y - DOMAIN_LENGTH) < tolerance:
            groups["periodic_y_plus"].append(tag)
        elif abs(z) < tolerance:
            groups["periodic_z_minus"].append(tag)
        elif abs(z - DOMAIN_LENGTH) < tolerance:
            groups["periodic_z_plus"].append(tag)

    for name, tags in groups.items():
        if len(tags) != 1:
            raise RuntimeError(f"Expected one surface for {name}, found {tags}")

    return groups


def generate():
    gmsh.initialize()
    try:
        gmsh.model.add("taylor_green_periodic_box")
        volume = gmsh.model.occ.addBox(
            0.0,
            0.0,
            0.0,
            DOMAIN_LENGTH,
            DOMAIN_LENGTH,
            DOMAIN_LENGTH,
        )
        gmsh.model.occ.synchronize()

        surfaces = _classify_surfaces(volume)

        gmsh.model.addPhysicalGroup(3, [volume], tag=1, name="fluid")
        for tag, name in enumerate(surfaces, start=21):
            gmsh.model.addPhysicalGroup(
                2,
                surfaces[name],
                tag=tag,
                name=name,
            )

        # Mesh each plus surface as a translated copy of its minus surface.
        for axis, translation in (
            ("x", _translation(x=DOMAIN_LENGTH)),
            ("y", _translation(y=DOMAIN_LENGTH)),
            ("z", _translation(z=DOMAIN_LENGTH)),
        ):
            gmsh.model.mesh.setPeriodic(
                2,
                surfaces[f"periodic_{axis}_plus"],
                surfaces[f"periodic_{axis}_minus"],
                translation,
            )

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), MESH_SIZE)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(MESH_PATH))

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        _, element_tags, _ = gmsh.model.mesh.getElements(3)
        element_count = sum(len(tags) for tags in element_tags)
        print(f"Wrote: {MESH_PATH}")
        print(f"Nodes: {len(node_tags)}")
        print(f"Tetrahedra: {element_count}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    generate()
