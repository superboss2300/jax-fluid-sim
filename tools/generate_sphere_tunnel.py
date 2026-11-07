"""Generate a tagged, locally refined tetrahedral sphere wind tunnel."""

import argparse
from pathlib import Path

import gmsh


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "mesh"

# Tunnel dimensions. The sphere is centered at the origin.
X_MIN = -2.0
X_MAX = 4.0
Y_MIN = -2.0
Y_MAX = 2.0
Z_MIN = -2.0
Z_MAX = 2.0
SPHERE_RADIUS = 0.5

MESH_PRESETS = {
    # Useful for debugging solver changes.
    "coarse": dict(near=0.10, shock=0.14, wake=0.16, far=0.40),
    # Sized for P2 DG on a 6 GB GPU. It retains a substantially smoother
    # curved sphere but avoids the very large face tensors of `refined`.
    "gpu6gb": dict(near=0.070, shock=0.18, wake=0.20, far=0.40),
    # Default production-development mesh: substantially finer than the old
    # 0.10/0.40 mesh without making every far-field tetrahedron expensive.
    "refined": dict(near=0.050, shock=0.070, wake=0.085, far=0.30),
    # Expensive visualization/convergence mesh.
    "fine": dict(near=0.032, shock=0.045, wake=0.055, far=0.22),
    # Targets ~150k-200k tetra10 elements for P2 DG on 24 GB VRAM (RTX 3090 / 4090).
    "gpu24gb": dict(near=0.055, shock=0.075, wake=0.085, far=0.30),
    # High-resolution mesh for 48+ GB GPUs (A100 40GB/80GB, A6000).
    "gpu48gb": dict(near=0.036, shock=0.050, wake=0.060, far=0.24),
    # Maximum-resolution turbulence mesh for 80 GB GPUs (A100 80GB, H100).
    "ultra": dict(near=0.020, shock=0.030, wake=0.040, far=0.18),
}


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


def generate(preset: str = "refined", output_path: Path | None = None) -> None:
    sizes = MESH_PRESETS[preset]
    if output_path is None:
        output_path = OUTPUT_DIR / f"sphere_tunnel_{preset}.msh"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

        # Refine near the sphere and smoothly transition to the far field.
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
            sizes["near"],
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "SizeMax",
            sizes["far"],
        )
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.10)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 1.10)

        # Expected bow-shock region ahead of the sphere.
        shock_field = gmsh.model.mesh.field.add("Box")
        for name, value in (
            ("XMin", -1.35), ("XMax", 0.15),
            ("YMin", -1.15), ("YMax", 1.15),
            ("ZMin", -1.15), ("ZMax", 1.15),
            ("VIn", sizes["shock"]), ("VOut", sizes["far"]),
            ("Thickness", 0.35),
        ):
            gmsh.model.mesh.field.setNumber(shock_field, name, value)

        # Long downstream region where shear layers and the wake develop.
        wake_field = gmsh.model.mesh.field.add("Box")
        for name, value in (
            ("XMin", -0.15), ("XMax", 4.0),
            ("YMin", -0.85), ("YMax", 0.85),
            ("ZMin", -0.85), ("ZMax", 0.85),
            ("VIn", sizes["wake"]), ("VOut", sizes["far"]),
            ("Thickness", 0.40),
        ):
            gmsh.model.mesh.field.setNumber(wake_field, name, value)

        combined_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            combined_field,
            "FieldsList",
            [threshold_field, shock_field, wake_field],
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(combined_field)

        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT tetrahedra
        # Create complete quadratic elements and curve boundary nodes onto
        # the OpenCASCADE sphere instead of leaving them at linear midpoints.
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("HighOrder")
        gmsh.write(str(output_path))

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        element_count = sum(len(tags) for tags in element_tags)

        print(f"Preset: {preset} ({sizes})")
        print(f"Wrote: {output_path}")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=MESH_PRESETS, default="refined")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    generate(arguments.preset, arguments.output)
