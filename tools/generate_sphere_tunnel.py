"""Generate a tagged tetrahedral wind-tunnel mesh around a sphere."""

from pathlib import Path

import gmsh


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "mesh"
MESH_PATH = OUTPUT_DIR / "sphere_tunnel.msh"

# Tunnel dimensions. The sphere is centered at the origin.
X_MIN = -2.0
X_MAX = 4.0
Y_MIN = -2.0
Y_MAX = 2.0
Z_MIN = -2.0
Z_MAX = 2.0
SPHERE_RADIUS = 0.5

# Mesh sizes. Keep this first mesh modest enough for rapid iteration.
SIZE_NEAR_SPHERE = 0.10
SIZE_FAR_FIELD = 0.40


def _surface_groups(volume_tag: int) -> dict[str, list[int]]:
    """Classify the fluid volume's boundary surfaces by location."""
    groups: dict[str, list[int]] = {
        "inlet": [],
        "outlet": [],
        "walls": [],
        "sphere_wall": [],
    }

    tolerance = 1.0e-6
    boundaries = gmsh.model.getBoundary(
        [(3, volume_tag)],
        oriented=False,
        recursive=False,
    )

    for dimension, surface_tag in boundaries:
        if dimension != 2:
            continue

        center_x, center_y, center_z = gmsh.model.occ.getCenterOfMass(
            dimension,
            surface_tag,
        )

        if abs(center_x - X_MIN) < tolerance:
            groups["inlet"].append(surface_tag)
        elif abs(center_x - X_MAX) < tolerance:
            groups["outlet"].append(surface_tag)
        elif (
            abs(center_y - Y_MIN) < tolerance
            or abs(center_y - Y_MAX) < tolerance
            or abs(center_z - Z_MIN) < tolerance
            or abs(center_z - Z_MAX) < tolerance
        ):
            groups["walls"].append(surface_tag)
        else:
            groups["sphere_wall"].append(surface_tag)

    return groups


def _add_physical_group(
    dimension: int,
    entity_tags: list[int],
    physical_tag: int,
    name: str,
) -> None:
    if not entity_tags:
        raise RuntimeError(f"No entities found for physical group {name!r}")

    gmsh.model.addPhysicalGroup(
        dimension,
        entity_tags,
        tag=physical_tag,
        name=name,
    )


def generate() -> None:
    gmsh.initialize()

    try:
        gmsh.model.add("sphere_wind_tunnel")

        box = gmsh.model.occ.addBox(
            X_MIN,
            Y_MIN,
            Z_MIN,
            X_MAX - X_MIN,
            Y_MAX - Y_MIN,
            Z_MAX - Z_MIN,
        )

        sphere = gmsh.model.occ.addSphere(
            0.0,
            0.0,
            0.0,
            SPHERE_RADIUS,
        )

        fluid_entities, _ = gmsh.model.occ.cut(
            [(3, box)],
            [(3, sphere)],
            removeObject=True,
            removeTool=True,
        )

        gmsh.model.occ.synchronize()

        fluid_volumes = [
            tag for dimension, tag in fluid_entities if dimension == 3
        ]
        if len(fluid_volumes) != 1:
            raise RuntimeError(
                f"Expected one fluid volume, found {len(fluid_volumes)}"
            )

        fluid_volume = fluid_volumes[0]
        surfaces = _surface_groups(fluid_volume)

        _add_physical_group(3, [fluid_volume], 1, "fluid")
        _add_physical_group(2, surfaces["inlet"], 11, "inlet")
        _add_physical_group(2, surfaces["outlet"], 12, "outlet")
        _add_physical_group(2, surfaces["walls"], 13, "walls")
        _add_physical_group(2, surfaces["sphere_wall"], 14, "sphere_wall")

        # Refine near the sphere and smoothly transition to the far-field size.
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            distance_field,
            "FacesList",
            surfaces["sphere_wall"],
        )

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "InField",
            distance_field,
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "SizeMin",
            SIZE_NEAR_SPHERE,
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "SizeMax",
            SIZE_FAR_FIELD,
        )
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.15)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 1.25)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT tetrahedra
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(MESH_PATH))

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        element_count = sum(len(tags) for tags in element_tags)

        print(f"Wrote: {MESH_PATH}")
        print(f"Nodes: {len(node_tags)}")
        print(f"Volume elements: {element_count}")
        print(f"Gmsh element types: {list(element_types)}")
        print(
            "Boundary surface counts: "
            + ", ".join(f"{name}={len(tags)}" for name, tags in surfaces.items())
        )
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    generate()
