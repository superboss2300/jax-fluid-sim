"""Load linear or quadratic tetrahedral Gmsh meshes and build connectivity."""

from pathlib import Path

import meshio
import numpy as np


class MeshLoader:
    """Preprocess tagged tetrahedral topology and high-order geometry."""

    def __init__(self, mesh_path):
        self.mesh_path = Path(mesh_path)
        mesh = meshio.read(self.mesh_path)

        self.points = np.asarray(mesh.points, dtype=np.float64)
        self.physical_names = {
            name: int(data[0])
            for name, data in mesh.field_data.items()
        }

        (
            self.tetrahedra,
            self.boundary_triangles,
            gmsh_boundary_tags,
            self.tetrahedra_geometry,
            self.boundary_triangles_geometry,
            self.geometry_order,
        ) = self._extract_cells(mesh)

        self.number_of_cells = len(self.tetrahedra)
        self.cell_centers, self.cell_volumes = self._compute_cell_geometry()

        (
            self.internal_face_nodes,
            self.internal_owner,
            self.internal_neighbor,
            self.boundary_face_nodes,
            self.boundary_owner,
        ) = self._build_face_connectivity()

        self.internal_owner_face_geometry_nodes = np.asarray(
            [
                self._cell_face_geometry_nodes(cell, nodes)
                for cell, nodes in zip(
                    self.internal_owner,
                    self.internal_face_nodes,
                )
            ],
            dtype=np.int32,
        )
        self.internal_neighbor_face_geometry_nodes = np.asarray(
            [
                self._cell_face_geometry_nodes(cell, nodes)
                for cell, nodes in zip(
                    self.internal_neighbor,
                    self.internal_face_nodes,
                )
            ],
            dtype=np.int32,
        )
        self.boundary_face_geometry_nodes = np.asarray(
            [
                self._cell_face_geometry_nodes(cell, nodes)
                for cell, nodes in zip(
                    self.boundary_owner,
                    self.boundary_face_nodes,
                )
            ],
            dtype=np.int32,
        )

        self.boundary_tags = self._match_boundary_tags(gmsh_boundary_tags)

        (
            self.internal_face_centers,
            self.internal_face_areas,
            self.internal_face_normals,
        ) = self._compute_internal_face_geometry()

        (
            self.boundary_face_centers,
            self.boundary_face_areas,
            self.boundary_face_normals,
        ) = self._compute_boundary_face_geometry()

        # Ordinary internal faces have one physical location.  Periodic faces
        # are represented by two translated copies of the same face, so the
        # owner and neighbor reconstruction points must be stored separately.
        self.internal_owner_face_centers = self.internal_face_centers.copy()
        self.internal_neighbor_face_centers = self.internal_face_centers.copy()
        self.internal_owner_face_nodes = self.internal_face_nodes.copy()
        self.internal_neighbor_face_nodes = self.internal_face_nodes.copy()
        self.internal_face_displacements = (
            self.cell_centers[self.internal_neighbor]
            - self.cell_centers[self.internal_owner]
        )
        self.number_of_periodic_faces = 0

        self.number_of_internal_faces = len(self.internal_owner)
        self.number_of_boundary_faces = len(self.boundary_owner)

        self._validate()

    @staticmethod
    def _center_key(center, tolerance):
        """Create a stable integer key for matching translated face centers."""
        return tuple(np.rint(center / tolerance).astype(np.int64))

    def pair_periodic_boundaries(self, periodic_pairs, tolerance=1.0e-8):
        """Promote matching boundary faces to conservative internal faces.

        Each entry in ``periodic_pairs`` is
        ``(minus_name, plus_name, minus_to_plus_translation)``.  The normal
        stored for the resulting face is the outward normal on the minus side.
        The neighbor cell is translated back beside the owner when computing
        the wrapped center-to-center displacement.
        """
        remove_boundary = np.zeros(self.number_of_boundary_faces, dtype=bool)

        periodic_owner = []
        periodic_neighbor = []
        periodic_owner_centers = []
        periodic_neighbor_centers = []
        periodic_owner_nodes = []
        periodic_neighbor_nodes = []
        periodic_owner_geometry_nodes = []
        periodic_neighbor_geometry_nodes = []
        periodic_displacements = []
        periodic_areas = []
        periodic_normals = []

        for minus_name, plus_name, translation in periodic_pairs:
            minus_tag = self.physical_names[minus_name]
            plus_tag = self.physical_names[plus_name]
            translation = np.asarray(translation, dtype=np.float64)

            minus_indices = np.flatnonzero(self.boundary_tags == minus_tag)
            plus_indices = np.flatnonzero(self.boundary_tags == plus_tag)

            if len(minus_indices) != len(plus_indices):
                raise ValueError(
                    f"Periodic groups {minus_name!r} and {plus_name!r} have "
                    f"different face counts: {len(minus_indices)} and "
                    f"{len(plus_indices)}"
                )

            plus_lookup = {}
            for plus_index in plus_indices:
                key = self._center_key(
                    self.boundary_face_centers[plus_index],
                    tolerance,
                )
                if key in plus_lookup:
                    raise ValueError(
                        f"Duplicate periodic face-center key on {plus_name!r}"
                    )
                plus_lookup[key] = int(plus_index)

            for minus_index in minus_indices:
                target_center = (
                    self.boundary_face_centers[minus_index] + translation
                )
                key = self._center_key(target_center, tolerance)
                try:
                    plus_index = plus_lookup.pop(key)
                except KeyError as error:
                    raise ValueError(
                        f"Could not match a face from {minus_name!r} to "
                        f"{plus_name!r}; target center was {target_center}"
                    ) from error

                minus_area = self.boundary_face_areas[minus_index]
                plus_area = self.boundary_face_areas[plus_index]
                if not np.isclose(minus_area, plus_area, rtol=1.0e-7, atol=1.0e-12):
                    raise ValueError(
                        f"Periodic face areas do not match: {minus_area} and "
                        f"{plus_area}"
                    )

                owner = int(self.boundary_owner[minus_index])
                neighbor = int(self.boundary_owner[plus_index])
                wrapped_neighbor_center = self.cell_centers[neighbor] - translation
                displacement = wrapped_neighbor_center - self.cell_centers[owner]
                normal = self.boundary_face_normals[minus_index]

                if np.dot(normal, displacement) <= 0.0:
                    raise ValueError(
                        f"Periodic normal for {minus_name!r}/{plus_name!r} "
                        "does not point from owner to wrapped neighbor"
                    )

                periodic_owner.append(owner)
                periodic_neighbor.append(neighbor)
                periodic_owner_centers.append(
                    self.boundary_face_centers[minus_index]
                )
                periodic_neighbor_centers.append(
                    self.boundary_face_centers[plus_index]
                )

                minus_nodes = self.boundary_face_nodes[minus_index]
                plus_nodes = self.boundary_face_nodes[plus_index]
                ordered_plus_nodes = []
                available_plus_nodes = list(int(node) for node in plus_nodes)
                for minus_node in minus_nodes:
                    translated_point = self.points[int(minus_node)] + translation
                    distances = np.linalg.norm(
                        self.points[available_plus_nodes] - translated_point,
                        axis=1,
                    )
                    closest = int(np.argmin(distances))
                    if distances[closest] > tolerance:
                        raise ValueError(
                            "Could not match periodic face vertices within "
                            f"tolerance; distance was {distances[closest]}"
                        )
                    ordered_plus_nodes.append(available_plus_nodes.pop(closest))

                periodic_owner_nodes.append(minus_nodes)
                periodic_neighbor_nodes.append(ordered_plus_nodes)

                minus_geometry_nodes = self.boundary_face_geometry_nodes[
                    minus_index
                ]
                plus_geometry_nodes = self.boundary_face_geometry_nodes[
                    plus_index
                ]
                ordered_plus_geometry_nodes = []
                available_plus_geometry_nodes = list(
                    int(node) for node in plus_geometry_nodes
                )
                for minus_node in minus_geometry_nodes:
                    translated_point = self.points[int(minus_node)] + translation
                    distances = np.linalg.norm(
                        self.points[available_plus_geometry_nodes]
                        - translated_point,
                        axis=1,
                    )
                    closest = int(np.argmin(distances))
                    if distances[closest] > tolerance:
                        raise ValueError(
                            "Could not match periodic high-order face nodes; "
                            f"distance was {distances[closest]}"
                        )
                    ordered_plus_geometry_nodes.append(
                        available_plus_geometry_nodes.pop(closest)
                    )

                periodic_owner_geometry_nodes.append(minus_geometry_nodes)
                periodic_neighbor_geometry_nodes.append(
                    ordered_plus_geometry_nodes
                )
                periodic_displacements.append(displacement)
                periodic_areas.append(0.5 * (minus_area + plus_area))
                periodic_normals.append(normal)

                remove_boundary[minus_index] = True
                remove_boundary[plus_index] = True

            if plus_lookup:
                raise ValueError(
                    f"Some faces in periodic group {plus_name!r} were unmatched"
                )

        if periodic_owner:
            periodic_owner = np.asarray(periodic_owner, dtype=np.int32)
            periodic_neighbor = np.asarray(periodic_neighbor, dtype=np.int32)

            self.internal_owner = np.concatenate(
                [self.internal_owner, periodic_owner]
            )
            self.internal_neighbor = np.concatenate(
                [self.internal_neighbor, periodic_neighbor]
            )
            self.internal_owner_face_centers = np.concatenate(
                [
                    self.internal_owner_face_centers,
                    np.asarray(periodic_owner_centers),
                ],
                axis=0,
            )
            self.internal_neighbor_face_centers = np.concatenate(
                [
                    self.internal_neighbor_face_centers,
                    np.asarray(periodic_neighbor_centers),
                ],
                axis=0,
            )
            self.internal_owner_face_nodes = np.concatenate(
                [
                    self.internal_owner_face_nodes,
                    np.asarray(periodic_owner_nodes, dtype=np.int32),
                ],
                axis=0,
            )
            self.internal_neighbor_face_nodes = np.concatenate(
                [
                    self.internal_neighbor_face_nodes,
                    np.asarray(periodic_neighbor_nodes, dtype=np.int32),
                ],
                axis=0,
            )
            self.internal_owner_face_geometry_nodes = np.concatenate(
                [
                    self.internal_owner_face_geometry_nodes,
                    np.asarray(periodic_owner_geometry_nodes, dtype=np.int32),
                ],
                axis=0,
            )
            self.internal_neighbor_face_geometry_nodes = np.concatenate(
                [
                    self.internal_neighbor_face_geometry_nodes,
                    np.asarray(periodic_neighbor_geometry_nodes, dtype=np.int32),
                ],
                axis=0,
            )
            self.internal_face_displacements = np.concatenate(
                [
                    self.internal_face_displacements,
                    np.asarray(periodic_displacements),
                ],
                axis=0,
            )
            self.internal_face_centers = self.internal_owner_face_centers
            self.internal_face_areas = np.concatenate(
                [self.internal_face_areas, np.asarray(periodic_areas)]
            )
            self.internal_face_normals = np.concatenate(
                [self.internal_face_normals, np.asarray(periodic_normals)],
                axis=0,
            )

        keep_boundary = ~remove_boundary
        self.boundary_face_nodes = self.boundary_face_nodes[keep_boundary]
        self.boundary_face_geometry_nodes = (
            self.boundary_face_geometry_nodes[keep_boundary]
        )
        self.boundary_owner = self.boundary_owner[keep_boundary]
        self.boundary_tags = self.boundary_tags[keep_boundary]
        self.boundary_face_centers = self.boundary_face_centers[keep_boundary]
        self.boundary_face_areas = self.boundary_face_areas[keep_boundary]
        self.boundary_face_normals = self.boundary_face_normals[keep_boundary]

        self.number_of_periodic_faces = len(periodic_owner)
        self.number_of_internal_faces = len(self.internal_owner)
        self.number_of_boundary_faces = len(self.boundary_owner)

    @staticmethod
    def _concatenate_blocks(blocks, cell_type):
        if not blocks:
            raise ValueError(f"Mesh contains no {cell_type} cells")
        return np.concatenate(blocks, axis=0)

    def _extract_cells(self, mesh):
        tetra_blocks = []
        tetra10_blocks = []
        triangle_blocks = []
        triangle6_blocks = []
        triangle_tag_blocks = []

        try:
            physical_data = mesh.cell_data["gmsh:physical"]
        except KeyError as error:
            raise ValueError("Mesh has no Gmsh physical-group data") from error

        for cell_block, physical_tags in zip(mesh.cells, physical_data):
            if cell_block.type == "tetra":
                tetra_blocks.append(cell_block.data)
            elif cell_block.type == "tetra10":
                tetra10_blocks.append(cell_block.data)
            elif cell_block.type == "triangle":
                triangle_blocks.append(cell_block.data)
                triangle_tag_blocks.append(physical_tags)
            elif cell_block.type == "triangle6":
                triangle6_blocks.append(cell_block.data)
                triangle_tag_blocks.append(physical_tags)

        if tetra10_blocks:
            if tetra_blocks:
                raise ValueError(
                    "Mixed tetra/tetra10 meshes are not currently supported"
                )
            tetrahedra_geometry_gmsh = self._concatenate_blocks(
                tetra10_blocks,
                "quadratic tetrahedron",
            )
            # Gmsh/meshio: vertices, then edges 01, 12, 20, 03, 13, 23.
            # Solver geometry: vertices, then edges 01, 02, 03, 12, 13, 23.
            tetrahedra_geometry = tetrahedra_geometry_gmsh[:, [
                0, 1, 2, 3, 4, 6, 7, 5, 8, 9
            ]]
            tetrahedra = tetrahedra_geometry[:, :4]
            geometry_order = 2
        else:
            tetrahedra = self._concatenate_blocks(
                tetra_blocks,
                "tetrahedron",
            )
            tetrahedra_geometry = tetrahedra
            geometry_order = 1

        if triangle6_blocks:
            if triangle_blocks:
                raise ValueError(
                    "Mixed triangle/triangle6 boundaries are not currently supported"
                )
            boundary_triangles_geometry = self._concatenate_blocks(
                triangle6_blocks,
                "quadratic boundary triangle",
            )
            boundary_triangles = boundary_triangles_geometry[:, :3]
        else:
            boundary_triangles = self._concatenate_blocks(
                triangle_blocks,
                "boundary triangle",
            )
            boundary_triangles_geometry = boundary_triangles

        boundary_tags = self._concatenate_blocks(
            triangle_tag_blocks,
            "boundary triangle tag",
        )

        return (
            np.asarray(tetrahedra, dtype=np.int32),
            np.asarray(boundary_triangles, dtype=np.int32),
            np.asarray(boundary_tags, dtype=np.int32),
            np.asarray(tetrahedra_geometry, dtype=np.int32),
            np.asarray(boundary_triangles_geometry, dtype=np.int32),
            geometry_order,
        )

    def _cell_face_geometry_nodes(self, cell_id, face_vertices):
        """Return face nodes as vertices a,b,c then edges ab,bc,ca."""
        face_vertices = tuple(int(node) for node in face_vertices)
        if self.geometry_order == 1:
            return np.asarray(face_vertices, dtype=np.int32)

        geometry_nodes = self.tetrahedra_geometry[int(cell_id)]
        corners = geometry_nodes[:4]
        edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        edge_lookup = {
            tuple(sorted((int(corners[i]), int(corners[j])))): int(
                geometry_nodes[4 + edge_index]
            )
            for edge_index, (i, j) in enumerate(edge_pairs)
        }
        a, b, c = face_vertices
        return np.asarray(
            [
                a,
                b,
                c,
                edge_lookup[tuple(sorted((a, b)))],
                edge_lookup[tuple(sorted((b, c)))],
                edge_lookup[tuple(sorted((c, a)))],
            ],
            dtype=np.int32,
        )

    def _compute_cell_geometry(self):
        # Shape: (number of cells, 4 tetra vertices, xyz)
        vertices = self.points[self.tetrahedra]

        centers = np.mean(vertices, axis=1)

        edge_1 = vertices[:, 1] - vertices[:, 0]
        edge_2 = vertices[:, 2] - vertices[:, 0]
        edge_3 = vertices[:, 3] - vertices[:, 0]

        signed_six_volumes = np.einsum(
            "ij,ij->i",
            edge_1,
            np.cross(edge_2, edge_3),
        )
        volumes = np.abs(signed_six_volumes) / 6.0

        return centers, volumes

    def _build_face_connectivity(self):
        # A sorted node tuple identifies a face regardless of vertex order.
        face_table = {}

        for cell_id, (a, b, c, d) in enumerate(self.tetrahedra):
            local_faces = (
                (a, b, c),
                (a, b, d),
                (a, c, d),
                (b, c, d),
            )

            for nodes in local_faces:
                nodes = tuple(int(node) for node in nodes)
                key = tuple(sorted(nodes))

                if key not in face_table:
                    face_table[key] = [nodes, cell_id, -1]
                elif face_table[key][2] == -1:
                    face_table[key][2] = cell_id
                else:
                    raise ValueError(f"Non-manifold face shared by 3+ cells: {key}")

        internal_nodes = []
        internal_owner = []
        internal_neighbor = []
        boundary_nodes = []
        boundary_owner = []

        for nodes, owner, neighbor in face_table.values():
            if neighbor == -1:
                boundary_nodes.append(nodes)
                boundary_owner.append(owner)
            else:
                internal_nodes.append(nodes)
                internal_owner.append(owner)
                internal_neighbor.append(neighbor)

        return (
            np.asarray(internal_nodes, dtype=np.int32),
            np.asarray(internal_owner, dtype=np.int32),
            np.asarray(internal_neighbor, dtype=np.int32),
            np.asarray(boundary_nodes, dtype=np.int32),
            np.asarray(boundary_owner, dtype=np.int32),
        )

    def _match_boundary_tags(self, gmsh_boundary_tags):
        tag_lookup = {
            tuple(sorted(int(node) for node in nodes)): int(tag)
            for nodes, tag in zip(
                self.boundary_triangles,
                gmsh_boundary_tags,
            )
        }

        tags = []
        for nodes in self.boundary_face_nodes:
            key = tuple(sorted(int(node) for node in nodes))
            if key not in tag_lookup:
                raise ValueError(f"Boundary face has no physical tag: {key}")
            tags.append(tag_lookup[key])

        return np.asarray(tags, dtype=np.int32)

    def _raw_face_geometry(self, face_nodes):
        vertices = self.points[face_nodes]
        p0 = vertices[:, 0]
        p1 = vertices[:, 1]
        p2 = vertices[:, 2]

        centers = (p0 + p1 + p2) / 3.0
        area_vectors = 0.5 * np.cross(p1 - p0, p2 - p0)
        return centers, area_vectors

    @staticmethod
    def _areas_and_normals(area_vectors):
        areas = np.linalg.norm(area_vectors, axis=1)
        normals = area_vectors / areas[:, None]
        return areas, normals

    def _compute_internal_face_geometry(self):
        centers, area_vectors = self._raw_face_geometry(
            self.internal_face_nodes
        )

        # Make each normal point from its owner cell to its neighbor cell.
        owner_to_neighbor = (
            self.cell_centers[self.internal_neighbor]
            - self.cell_centers[self.internal_owner]
        )
        alignment = np.einsum("ij,ij->i", area_vectors, owner_to_neighbor)
        area_vectors[alignment < 0.0] *= -1.0

        areas, normals = self._areas_and_normals(area_vectors)
        return centers, areas, normals

    def _compute_boundary_face_geometry(self):
        centers, area_vectors = self._raw_face_geometry(
            self.boundary_face_nodes
        )

        # Make each boundary normal point outward from its owner cell.
        owner_to_face = centers - self.cell_centers[self.boundary_owner]
        alignment = np.einsum("ij,ij->i", area_vectors, owner_to_face)
        area_vectors[alignment < 0.0] *= -1.0

        areas, normals = self._areas_and_normals(area_vectors)
        return centers, areas, normals

    def _validate(self):
        if np.any(self.cell_volumes <= 0.0):
            raise ValueError("Mesh contains a zero-volume tetrahedron")
        if np.any(self.internal_face_areas <= 0.0):
            raise ValueError("Mesh contains a zero-area internal face")
        if np.any(self.boundary_face_areas <= 0.0):
            raise ValueError("Mesh contains a zero-area boundary face")

        internal_direction = (
            self.cell_centers[self.internal_neighbor]
            - self.cell_centers[self.internal_owner]
        )
        internal_alignment = np.einsum(
            "ij,ij->i",
            self.internal_face_normals,
            internal_direction,
        )
        if np.any(internal_alignment <= 0.0):
            raise ValueError("An internal face normal is incorrectly oriented")

        boundary_direction = (
            self.boundary_face_centers
            - self.cell_centers[self.boundary_owner]
        )
        boundary_alignment = np.einsum(
            "ij,ij->i",
            self.boundary_face_normals,
            boundary_direction,
        )
        if np.any(boundary_alignment <= 0.0):
            raise ValueError("A boundary face normal is not outward-facing")

    def boundary_mask(self, name):
        """Return a Boolean mask selecting a named physical boundary."""
        try:
            tag = self.physical_names[name]
        except KeyError as error:
            raise KeyError(f"Unknown physical boundary {name!r}") from error
        return self.boundary_tags == tag

    def jax_arrays(self):
        """Return the fixed solver arrays as JAX device arrays."""
        import jax.numpy as jnp

        return {
            "cell_centers": jnp.asarray(self.cell_centers),
            "cell_volumes": jnp.asarray(self.cell_volumes),
            "internal_owner": jnp.asarray(self.internal_owner),
            "internal_neighbor": jnp.asarray(self.internal_neighbor),
            "internal_face_centers": jnp.asarray(self.internal_face_centers),
            "internal_owner_face_centers": jnp.asarray(
                self.internal_owner_face_centers
            ),
            "internal_neighbor_face_centers": jnp.asarray(
                self.internal_neighbor_face_centers
            ),
            "internal_face_displacements": jnp.asarray(
                self.internal_face_displacements
            ),
            "internal_owner_face_nodes": jnp.asarray(
                self.internal_owner_face_nodes
            ),
            "internal_neighbor_face_nodes": jnp.asarray(
                self.internal_neighbor_face_nodes
            ),
            "internal_owner_face_geometry_nodes": jnp.asarray(
                self.internal_owner_face_geometry_nodes
            ),
            "internal_neighbor_face_geometry_nodes": jnp.asarray(
                self.internal_neighbor_face_geometry_nodes
            ),
            "internal_face_areas": jnp.asarray(self.internal_face_areas),
            "internal_face_normals": jnp.asarray(self.internal_face_normals),
            "boundary_owner": jnp.asarray(self.boundary_owner),
            "boundary_face_centers": jnp.asarray(self.boundary_face_centers),
            "boundary_face_areas": jnp.asarray(self.boundary_face_areas),
            "boundary_face_normals": jnp.asarray(self.boundary_face_normals),
            "boundary_face_geometry_nodes": jnp.asarray(
                self.boundary_face_geometry_nodes
            ),
            "boundary_tags": jnp.asarray(self.boundary_tags),
        }

    def summary(self):
        unique_tags, counts = np.unique(self.boundary_tags, return_counts=True)
        id_to_name = {tag: name for name, tag in self.physical_names.items()}

        print(f"Mesh: {self.mesh_path}")
        print(f"Points: {len(self.points)}")
        print(f"Tetrahedral cells: {self.number_of_cells}")
        print(f"Geometry order: {self.geometry_order}")
        print(f"Internal faces: {self.number_of_internal_faces}")
        if self.number_of_periodic_faces:
            print(f"  Periodic face pairs: {self.number_of_periodic_faces}")
        print(f"Boundary faces: {self.number_of_boundary_faces}")
        print(f"Minimum cell volume: {self.cell_volumes.min():.6e}")
        print(f"Total fluid volume: {self.cell_volumes.sum():.6f}")
        print("Boundary groups:")
        for tag, count in zip(unique_tags, counts):
            name = id_to_name.get(int(tag), "unknown")
            print(f"  {name} (tag {tag}): {count} faces")


if __name__ == "__main__":
    project_directory = Path(__file__).resolve().parent
    loader = MeshLoader(project_directory / "mesh/sphere_tunnel.msh")
    loader.summary()
