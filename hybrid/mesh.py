"""Type-blocked mixed tetrahedron/prism/hexahedron mesh topology."""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import meshio
import numpy as np


_ELEMENT_ALIASES = {
    "tetra": ("tetra", 4),
    "tetra10": ("tetra", 4),
    "wedge": ("wedge", 6),
    "wedge15": ("wedge", 6),
    "wedge18": ("wedge", 6),
    "hexahedron": ("hexahedron", 8),
    "hexahedron20": ("hexahedron", 8),
    "hexahedron27": ("hexahedron", 8),
}

# Internal basis order <- meshio/VTK local node order. These were also checked
# against curved Gmsh 4.15 tetra10 and wedge18 sphere cells.
_COMPLETE_P2_PERMUTATIONS = {
    "tetra10": np.asarray([0,1,2,3,4,6,7,5,8,9], dtype=np.int32),
    "wedge18": np.asarray(
        [0,1,2,6,9,7,3,4,5,12,14,13,8,10,11,15,17,16],
        dtype=np.int32,
    ),
    "hexahedron27": np.asarray(
        [0,1,8,3,2,10,11,9,20,4,5,12,7,6,14,15,13,21,
         16,17,22,19,18,24,25,23,26],
        dtype=np.int32,
    ),
}

_LOCAL_FACES = {
    "tetra": (
        (0, 2, 1),
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
    ),
    "wedge": (
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    ),
    "hexahedron": (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
}

_TETRA_DECOMPOSITIONS = {
    "tetra": ((0, 1, 2, 3),),
    "wedge": (
        (0, 1, 2, 3),
        (1, 2, 4, 3),
        (2, 4, 5, 3),
    ),
    "hexahedron": (
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    ),
}


def _p2_reference_nodes_numpy(element_type):
    if element_type == "tetra":
        return np.asarray([
            [0,0,0], [1,0,0], [0,1,0], [0,0,1],
            [.5,0,0], [0,.5,0], [0,0,.5],
            [.5,.5,0], [.5,0,.5], [0,.5,.5],
        ], dtype=np.float64)
    if element_type == "wedge":
        triangle = np.asarray(
            [[0,0], [1,0], [0,1], [.5,0], [.5,.5], [0,.5]],
            dtype=np.float64,
        )
        return np.asarray(
            [[x,y,z] for z in (0.0,1.0,0.5) for x,y in triangle]
        )
    if element_type == "hexahedron":
        return np.asarray([
            [x,y,z]
            for z in (0.0,1.0,0.5)
            for y in (0.0,1.0,0.5)
            for x in (0.0,1.0,0.5)
        ])
    raise ValueError(element_type)


def _p1_basis_numpy(element_type, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if element_type == "tetra":
        return np.stack([1-x-y-z, x, y, z], axis=-1)
    if element_type == "wedge":
        triangle = (1-x-y, x, y)
        return np.stack([
            triangle[0]*(1-z), triangle[1]*(1-z), triangle[2]*(1-z),
            triangle[0]*z, triangle[1]*z, triangle[2]*z,
        ], axis=-1)
    if element_type == "hexahedron":
        return np.stack([
            (1-x)*(1-y)*(1-z), x*(1-y)*(1-z), x*y*(1-z), (1-x)*y*(1-z),
            (1-x)*(1-y)*z, x*(1-y)*z, x*y*z, (1-x)*y*z,
        ], axis=-1)
    raise ValueError(element_type)


@dataclass
class ElementBlock:
    element_type: str
    source_cell_type: str
    connectivity: np.ndarray
    geometry_connectivity: np.ndarray
    global_ids: np.ndarray
    centers: np.ndarray
    volumes: np.ndarray

    def jax_arrays(self) -> dict[str, jnp.ndarray]:
        return {
            "connectivity": jnp.asarray(self.connectivity),
            "geometry_connectivity": jnp.asarray(self.geometry_connectivity),
            "global_ids": jnp.asarray(self.global_ids),
            "centers": jnp.asarray(self.centers),
            "volumes": jnp.asarray(self.volumes),
        }


@dataclass
class FaceBlock:
    face_type: str
    nodes: np.ndarray
    owner_block: np.ndarray
    owner_local: np.ndarray
    owner_face: np.ndarray
    neighbor_block: np.ndarray
    neighbor_local: np.ndarray
    neighbor_face: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    boundary_tags: np.ndarray

    @property
    def internal(self) -> np.ndarray:
        return self.neighbor_block >= 0

    def jax_arrays(self) -> dict[str, jnp.ndarray]:
        return {
            "nodes": jnp.asarray(self.nodes),
            "owner_block": jnp.asarray(self.owner_block),
            "owner_local": jnp.asarray(self.owner_local),
            "owner_face": jnp.asarray(self.owner_face),
            "neighbor_block": jnp.asarray(self.neighbor_block),
            "neighbor_local": jnp.asarray(self.neighbor_local),
            "neighbor_face": jnp.asarray(self.neighbor_face),
            "centers": jnp.asarray(self.centers),
            "normals": jnp.asarray(self.normals),
            "areas": jnp.asarray(self.areas),
            "boundary_tags": jnp.asarray(self.boundary_tags),
            "internal": jnp.asarray(self.internal),
        }


class HybridMesh:
    """Read a mixed mesh and group topology by element and face type.

    Kernels operate on homogeneous blocks, avoiding padded ragged arrays on
    the GPU. Faces retain block/local indices so a triangle or quad flux can
    conservatively scatter into unlike neighboring element types.
    """

    def __init__(self, mesh_path):
        self.mesh_path = Path(mesh_path)
        source = meshio.read(self.mesh_path)
        self.points = np.asarray(source.points, dtype=np.float64)
        self.physical_names = {
            name: int(data[0]) for name, data in source.field_data.items()
        }
        self.element_blocks = self._extract_element_blocks(source)
        self.number_of_cells = sum(
            len(block.connectivity) for block in self.element_blocks
        )
        boundary_tags = self._extract_boundary_tags(source)
        self.face_blocks = self._build_faces(boundary_tags)
        self.number_of_internal_faces = sum(
            int(np.sum(block.internal)) for block in self.face_blocks.values()
        )
        self.number_of_boundary_faces = sum(
            int(np.sum(~block.internal)) for block in self.face_blocks.values()
        )
        self._validate()

    @staticmethod
    def _tetra_volume(vertices):
        return abs(
            np.linalg.det(
                np.stack(
                    [
                        vertices[1] - vertices[0],
                        vertices[2] - vertices[0],
                        vertices[3] - vertices[0],
                    ],
                    axis=1,
                )
            )
        ) / 6.0

    def _cell_geometry(self, element_type, connectivity):
        centers = []
        volumes = []
        decomposition = _TETRA_DECOMPOSITIONS[element_type]
        for cell_nodes in connectivity:
            vertices = self.points[cell_nodes]
            pieces = []
            weighted_centers = []
            for local_tetra in decomposition:
                tetra = vertices[np.asarray(local_tetra)]
                volume = self._tetra_volume(tetra)
                pieces.append(volume)
                weighted_centers.append(volume * np.mean(tetra, axis=0))
            total = float(np.sum(pieces))
            centers.append(np.sum(weighted_centers, axis=0) / total)
            volumes.append(total)
        return np.asarray(centers), np.asarray(volumes)

    def _extract_element_blocks(self, source):
        grouped: dict[tuple[str, str], list[np.ndarray]] = {}
        for cell_block in source.cells:
            if cell_block.type not in _ELEMENT_ALIASES:
                continue
            element_type, corner_count = _ELEMENT_ALIASES[cell_block.type]
            grouped.setdefault((element_type, cell_block.type), []).append(
                np.asarray(cell_block.data, dtype=np.int32)
            )

        if not grouped:
            raise ValueError("Mesh contains no tetrahedron, wedge, or hexahedron")

        blocks = []
        global_start = 0
        for element_type in ("hexahedron", "wedge", "tetra"):
            for (candidate, source_type), arrays in grouped.items():
                if candidate != element_type:
                    continue
                geometry = np.concatenate(arrays, axis=0)
                corner_count = _ELEMENT_ALIASES[source_type][1]
                connectivity = geometry[:, :corner_count]
                if source_type in {"tetra10", "wedge18", "hexahedron27"}:
                    geometry = geometry[
                        :, _COMPLETE_P2_PERMUTATIONS[source_type]
                    ]
                centers, volumes = self._cell_geometry(
                    element_type, connectivity
                )
                global_ids = np.arange(
                    global_start,
                    global_start + len(connectivity),
                    dtype=np.int32,
                )
                global_start += len(connectivity)
                blocks.append(
                    ElementBlock(
                        element_type=element_type,
                        source_cell_type=source_type,
                        connectivity=connectivity,
                        geometry_connectivity=geometry,
                        global_ids=global_ids,
                        centers=centers,
                        volumes=volumes,
                    )
                )
        return blocks

    def _order_complete_p2_geometry(
        self, element_type, connectivity, source_geometry
    ):
        """Map meshio's node order to this solver's tensor-product P2 order.

        Matching is done in physical space against the straight-sided P1
        location. This avoids format-specific Gmsh/VTK ordering tables and
        remains reliable for normally curved, valid second-order cells.
        """
        reference_nodes = _p2_reference_nodes_numpy(element_type)
        corner_basis = _p1_basis_numpy(element_type, reference_nodes)
        ordered = np.empty_like(source_geometry)

        for cell_index, (corners, source_nodes) in enumerate(
            zip(connectivity, source_geometry)
        ):
            expected = corner_basis @ self.points[corners]
            # Wedge18 and hex27 use tensor-product solution ordering, so their
            # corner nodes are intentionally not the first 6/8 local entries.
            remaining_internal = list(range(len(reference_nodes)))
            remaining_source = list(range(source_geometry.shape[1]))
            scale = np.max(np.linalg.norm(
                self.points[corners] - np.mean(self.points[corners], axis=0),
                axis=1,
            ))
            while remaining_internal:
                distances = np.asarray([
                    [
                        np.linalg.norm(
                            expected[internal] - self.points[source_nodes[source]]
                        )
                        for source in remaining_source
                    ]
                    for internal in remaining_internal
                ])
                flat = int(np.argmin(distances))
                row, column = np.unravel_index(flat, distances.shape)
                internal = remaining_internal.pop(row)
                source = remaining_source.pop(column)
                if distances[row, column] > 0.45 * max(scale, 1.0e-14):
                    raise ValueError(
                        f"Cannot identify P2 node ordering for {element_type} "
                        f"cell {cell_index}"
                    )
                ordered[cell_index, internal] = source_nodes[source]
        return ordered

    @staticmethod
    def _boundary_corner_count(cell_type):
        if cell_type in {"triangle", "triangle6"}:
            return 3
        if cell_type in {"quad", "quad8", "quad9"}:
            return 4
        return None

    def _extract_boundary_tags(self, source):
        physical = source.cell_data.get("gmsh:physical")
        tags = {}
        for block_index, cell_block in enumerate(source.cells):
            corner_count = self._boundary_corner_count(cell_block.type)
            if corner_count is None:
                continue
            if physical is None:
                block_tags = np.full(len(cell_block.data), -1, dtype=np.int32)
            else:
                block_tags = np.asarray(physical[block_index], dtype=np.int32)
            for nodes, tag in zip(cell_block.data[:, :corner_count], block_tags):
                key = tuple(sorted(int(node) for node in nodes))
                if key in tags and tags[key] != int(tag):
                    raise ValueError(f"Conflicting physical tags on face {key}")
                tags[key] = int(tag)
        return tags

    def _face_geometry(self, nodes, owner_center):
        vertices = self.points[nodes]
        center = np.mean(vertices, axis=0)
        if len(nodes) == 3:
            area_vector = 0.5 * np.cross(
                vertices[1] - vertices[0], vertices[2] - vertices[0]
            )
        else:
            area_vector = 0.5 * (
                np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                + np.cross(vertices[2] - vertices[0], vertices[3] - vertices[0])
            )
        if np.dot(area_vector, center - owner_center) < 0.0:
            area_vector = -area_vector
        area = np.linalg.norm(area_vector)
        return center, area_vector / area, area

    def _build_faces(self, boundary_tags):
        pending = {}
        finished = {"triangle": [], "quad": []}

        for block_id, block in enumerate(self.element_blocks):
            for local_cell, cell_nodes in enumerate(block.connectivity):
                for local_face, face_vertices in enumerate(
                    _LOCAL_FACES[block.element_type]
                ):
                    nodes = np.asarray(
                        [cell_nodes[index] for index in face_vertices],
                        dtype=np.int32,
                    )
                    key = tuple(sorted(int(node) for node in nodes))
                    record = (block_id, local_cell, local_face, nodes)
                    if key in pending:
                        owner = pending.pop(key)
                        finished["triangle" if len(nodes) == 3 else "quad"].append(
                            (owner, record, -1)
                        )
                    else:
                        pending[key] = record

        for key, owner in pending.items():
            tag = boundary_tags.get(key, -1)
            finished["triangle" if len(key) == 3 else "quad"].append(
                (owner, None, tag)
            )

        result = {}
        for face_type, records in finished.items():
            if not records:
                continue
            columns = {
                "nodes": [], "owner_block": [], "owner_local": [],
                "owner_face": [], "neighbor_block": [],
                "neighbor_local": [], "neighbor_face": [], "centers": [],
                "normals": [], "areas": [], "boundary_tags": [],
            }
            for owner, neighbor, tag in records:
                owner_block, owner_local, owner_face, nodes = owner
                center, normal, area = self._face_geometry(
                    nodes,
                    self.element_blocks[owner_block].centers[owner_local],
                )
                if neighbor is None:
                    neighbor_block = neighbor_local = neighbor_face = -1
                else:
                    neighbor_block, neighbor_local, neighbor_face, _ = neighbor
                    displacement = (
                        self.element_blocks[neighbor_block].centers[neighbor_local]
                        - self.element_blocks[owner_block].centers[owner_local]
                    )
                    if np.dot(normal, displacement) < 0.0:
                        normal = -normal
                columns["nodes"].append(nodes)
                columns["owner_block"].append(owner_block)
                columns["owner_local"].append(owner_local)
                columns["owner_face"].append(owner_face)
                columns["neighbor_block"].append(neighbor_block)
                columns["neighbor_local"].append(neighbor_local)
                columns["neighbor_face"].append(neighbor_face)
                columns["centers"].append(center)
                columns["normals"].append(normal)
                columns["areas"].append(area)
                columns["boundary_tags"].append(tag)

            result[face_type] = FaceBlock(
                face_type=face_type,
                nodes=np.asarray(columns["nodes"], dtype=np.int32),
                owner_block=np.asarray(columns["owner_block"], dtype=np.int32),
                owner_local=np.asarray(columns["owner_local"], dtype=np.int32),
                owner_face=np.asarray(columns["owner_face"], dtype=np.int32),
                neighbor_block=np.asarray(columns["neighbor_block"], dtype=np.int32),
                neighbor_local=np.asarray(columns["neighbor_local"], dtype=np.int32),
                neighbor_face=np.asarray(columns["neighbor_face"], dtype=np.int32),
                centers=np.asarray(columns["centers"], dtype=np.float64),
                normals=np.asarray(columns["normals"], dtype=np.float64),
                areas=np.asarray(columns["areas"], dtype=np.float64),
                boundary_tags=np.asarray(columns["boundary_tags"], dtype=np.int32),
            )
        return result

    def _validate(self):
        volumes = np.concatenate(
            [block.volumes for block in self.element_blocks]
        )
        if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
            raise ValueError("Mesh contains non-positive or invalid cell volumes")
        for face_type, block in self.face_blocks.items():
            if np.any(block.areas <= 0.0):
                raise ValueError(f"Mesh contains invalid {face_type} face areas")
            internal = block.internal
            if np.any(block.boundary_tags[internal] != -1):
                raise ValueError("Internal faces cannot have boundary tags")
            if np.any(block.boundary_tags[~internal] < 0):
                missing = int(np.sum(block.boundary_tags[~internal] < 0))
                raise ValueError(
                    f"{missing} boundary {face_type} faces lack physical tags"
                )

    def jax_arrays(self):
        return {
            "points": jnp.asarray(self.points),
            "elements": {
                block.element_type: block.jax_arrays()
                for block in self.element_blocks
            },
            "faces": {
                name: block.jax_arrays()
                for name, block in self.face_blocks.items()
            },
        }

    def summary(self):
        print(f"Hybrid mesh: {self.mesh_path}")
        print(f"Points: {len(self.points)}")
        for block_id, block in enumerate(self.element_blocks):
            print(
                f"Block {block_id}: {len(block.connectivity)} "
                f"{block.source_cell_type} cells; "
                f"volume=[{block.volumes.min():.6e}, {block.volumes.max():.6e}]"
            )
        print(f"Internal faces: {self.number_of_internal_faces}")
        print(f"Boundary faces: {self.number_of_boundary_faces}")
        for name, block in self.face_blocks.items():
            print(
                f"{name} faces: {len(block.nodes)} "
                f"({int(np.sum(block.internal))} internal)"
            )
