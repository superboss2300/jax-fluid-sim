"""Physical volume and face mappings for hybrid reference elements."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .elements import make_reference_element


LOCAL_FACES = {
    "tetra": (
        (0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3),
    ),
    "wedge": (
        (0, 2, 1), (3, 4, 5), (0, 1, 4, 3),
        (1, 2, 5, 4), (2, 0, 3, 5),
    ),
    "hexahedron": (
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ),
}


@dataclass(frozen=True)
class ElementMetrics:
    physical_points: jnp.ndarray
    jacobian: jnp.ndarray
    inverse_jacobian: jnp.ndarray
    determinant: jnp.ndarray
    integration_weights: jnp.ndarray
    physical_basis_gradients: jnp.ndarray
    mass_matrix: jnp.ndarray
    inverse_mass_matrix: jnp.ndarray


@dataclass(frozen=True)
class FaceMetrics:
    physical_points: jnp.ndarray
    normals: jnp.ndarray
    integration_weights: jnp.ndarray
    owner_reference_points: jnp.ndarray
    neighbor_reference_points: jnp.ndarray


def face_quadrature(face_type, polynomial_degree=1):
    """Return face shape weights and weights normalized to sum to one."""
    if face_type == "triangle" and polynomial_degree == 1:
        barycentric = jnp.asarray(
            [
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ]
        )
        return barycentric, jnp.full(3, 1.0 / 3.0)

    if face_type == "triangle":
        order = 4 if polynomial_degree <= 2 else 5
        raw, raw_weights = np.polynomial.legendre.leggauss(order)
        line = 0.5 * (raw + 1.0)
        weights_1d = 0.5 * raw_weights
        shapes = []
        weights = []
        for a, wa in zip(line, weights_1d):
            for b, wb in zip(line, weights_1d):
                x, y = a, (1.0-a)*b
                shapes.append([1.0-x-y, x, y])
                weights.append(2.0*wa*wb*(1.0-a))
        return jnp.asarray(shapes), jnp.asarray(weights)

    if face_type == "quad":
        order = 2 if polynomial_degree == 1 else (3 if polynomial_degree == 2 else 4)
        raw, raw_weights = np.polynomial.legendre.leggauss(order)
        line = 0.5 * (raw + 1.0)
        line_weights = 0.5 * raw_weights
        shapes = []
        weights = []
        for v, wv in zip(line, line_weights):
            for u, wu in zip(line, line_weights):
                shapes.append(
                    [(1-u)*(1-v), u*(1-v), u*v, (1-u)*v]
                )
                weights.append(wu*wv)
        return jnp.asarray(shapes), jnp.asarray(weights)

    raise ValueError(f"Unsupported face type {face_type!r}")


class HybridGeometry:
    """Precompute isoparametric metrics without padding unlike element types."""

    def __init__(self, mesh, polynomial_degree=1):
        self.mesh = mesh
        self.polynomial_degree = polynomial_degree
        self.reference_elements = {
            element_type: make_reference_element(element_type, polynomial_degree)
            for element_type in {block.element_type for block in mesh.element_blocks}
        }
        self.geometry_reference_elements = {}
        for block in mesh.element_blocks:
            geometry_degree = 2 if block.source_cell_type in {
                "tetra10", "wedge18", "hexahedron27"
            } else 1
            self.geometry_reference_elements[block.element_type] = (
                make_reference_element(block.element_type, geometry_degree)
            )
        self.element_metrics = self._build_element_metrics()
        self.solution_node_coordinates = self._build_solution_node_coordinates()
        self.face_metrics = self._build_face_metrics()
        self._validate()

    def _build_solution_node_coordinates(self):
        result = {}
        for block_id, block in enumerate(self.mesh.element_blocks):
            solution_reference = self.reference_elements[block.element_type]
            geometry_reference = self.geometry_reference_elements[
                block.element_type
            ]
            geometry_connectivity = (
                block.geometry_connectivity
                if geometry_reference.polynomial_degree == 2
                else block.connectivity
            )
            coordinates = jnp.asarray(
                self.mesh.points[geometry_connectivity]
            )
            geometry_basis = geometry_reference.basis(solution_reference.nodes)
            result[block.element_type] = jnp.einsum(
                "qi,eid->eqd",
                geometry_basis,
                coordinates,
                precision=jax.lax.Precision.HIGHEST,
            )
        return result

    def _build_element_metrics(self):
        result = {}
        for block_id, block in enumerate(self.mesh.element_blocks):
            reference = self.reference_elements[block.element_type]
            geometry_reference = self.geometry_reference_elements[
                block.element_type
            ]
            geometry_nodes = (
                block.geometry_connectivity
                if geometry_reference.polynomial_degree == 2
                else block.connectivity
            )
            coordinates = jnp.asarray(self.mesh.points[geometry_nodes])
            basis = geometry_reference.basis(reference.volume_points)
            solution_basis = reference.basis(reference.volume_points)
            gradients = reference.basis_gradients(reference.volume_points)
            geometry_gradients = geometry_reference.basis_gradients(
                reference.volume_points
            )
            physical_points = jnp.einsum(
                "qi,eid->eqd",
                basis,
                coordinates,
                precision=jax.lax.Precision.HIGHEST,
            )
            jacobian = jnp.einsum(
                "eid,qir->eqdr",
                coordinates,
                geometry_gradients,
                precision=jax.lax.Precision.HIGHEST,
            )
            determinant = jnp.linalg.det(jacobian)
            inverse = jnp.linalg.inv(jacobian)
            physical_gradients = jnp.einsum(
                "qir,eqrd->eqid",
                gradients,
                inverse,
                precision=jax.lax.Precision.HIGHEST,
            )
            integration_weights = (
                reference.volume_weights[None, :] * jnp.abs(determinant)
            )
            mass_matrix = jnp.einsum(
                "eq,qi,qj->eij",
                integration_weights,
                solution_basis,
                solution_basis,
                precision=jax.lax.Precision.HIGHEST,
            )
            result[block_id] = ElementMetrics(
                physical_points=physical_points,
                jacobian=jacobian,
                inverse_jacobian=inverse,
                determinant=determinant,
                integration_weights=integration_weights,
                physical_basis_gradients=physical_gradients,
                mass_matrix=mass_matrix,
                inverse_mass_matrix=jnp.linalg.inv(mass_matrix),
            )
        return result

    def _aligned_neighbor_reference_corners(
        self, face_block, face_index, owner_nodes
    ):
        neighbor_block_id = int(face_block.neighbor_block[face_index])
        if neighbor_block_id < 0:
            return np.full((len(owner_nodes), 3), np.nan)
        neighbor_local = int(face_block.neighbor_local[face_index])
        neighbor_face = int(face_block.neighbor_face[face_index])
        block = self.mesh.element_blocks[neighbor_block_id]
        reference = np.asarray(
            self.geometry_reference_elements[block.element_type].nodes
        )
        local_indices = LOCAL_FACES[block.element_type][neighbor_face]
        neighbor_nodes = block.connectivity[neighbor_local, list(local_indices)]
        neighbor_reference = reference[list(local_indices)]
        lookup = {
            int(node): neighbor_reference[position]
            for position, node in enumerate(neighbor_nodes)
        }
        return np.asarray([lookup[int(node)] for node in owner_nodes])

    def _build_face_metrics(self):
        result = {}
        for face_type, face_block in self.mesh.face_blocks.items():
            shape_values, normalized_weights = face_quadrature(
                face_type, self.polynomial_degree
            )
            shape_values_np = np.asarray(shape_values)
            owner_reference_points = []
            neighbor_reference_points = []
            tangent_u_reference = []
            tangent_v_reference = []

            for face_index, owner_nodes in enumerate(face_block.nodes):
                owner_block_id = int(face_block.owner_block[face_index])
                owner_local_face = int(face_block.owner_face[face_index])
                owner_block = self.mesh.element_blocks[owner_block_id]
                owner_reference = np.asarray(
                    self.geometry_reference_elements[
                        owner_block.element_type
                    ].nodes
                )
                owner_indices = LOCAL_FACES[owner_block.element_type][
                    owner_local_face
                ]
                owner_corners = owner_reference[list(owner_indices)]
                neighbor_corners = self._aligned_neighbor_reference_corners(
                    face_block, face_index, owner_nodes
                )
                owner_reference_points.append(shape_values_np @ owner_corners)
                neighbor_reference_points.append(
                    shape_values_np @ neighbor_corners
                )
                owner_points = shape_values_np @ owner_corners

                if face_type == "triangle":
                    tangent_u = np.broadcast_to(
                        owner_corners[1] - owner_corners[0], owner_points.shape
                    )
                    tangent_v = np.broadcast_to(
                        owner_corners[2] - owner_corners[0], owner_points.shape
                    )
                else:
                    u = shape_values_np[:, 1] + shape_values_np[:, 2]
                    v = shape_values_np[:, 2] + shape_values_np[:, 3]
                    c0, c1, c2, c3 = owner_corners
                    tangent_u = (
                        -(1-v)[:,None]*c0 + (1-v)[:,None]*c1
                        + v[:,None]*c2 - v[:,None]*c3
                    )
                    tangent_v = (
                        -(1-u)[:,None]*c0 - u[:,None]*c1
                        + u[:,None]*c2 + (1-u)[:,None]*c3
                    )
                tangent_u_reference.append(tangent_u)
                tangent_v_reference.append(tangent_v)

            owner_reference_points = jnp.asarray(
                np.asarray(owner_reference_points)
            )
            neighbor_reference_points = jnp.asarray(
                np.asarray(neighbor_reference_points)
            )
            tangent_u_reference = jnp.asarray(np.asarray(tangent_u_reference))
            tangent_v_reference = jnp.asarray(np.asarray(tangent_v_reference))
            face_count, quadrature_count = owner_reference_points.shape[:2]
            metric_dtype = jnp.asarray(self.mesh.points).dtype
            physical_points = jnp.zeros(
                (face_count, quadrature_count, 3), dtype=metric_dtype
            )
            normals = jnp.zeros_like(physical_points)
            integration_weights = jnp.zeros(
                (face_count, quadrature_count), dtype=metric_dtype
            )
            reference_measure = 0.5 if face_type == "triangle" else 1.0

            # One vectorized geometry calculation per owner block, rather than
            # one JAX dispatch per face.
            for owner_block_id in np.unique(face_block.owner_block):
                indices_np = np.flatnonzero(
                    face_block.owner_block == owner_block_id
                ).astype(np.int32)
                indices = jnp.asarray(indices_np)
                owner_block = self.mesh.element_blocks[int(owner_block_id)]
                geometry_reference = self.geometry_reference_elements[
                    owner_block.element_type
                ]
                local_cells = face_block.owner_local[indices_np]
                geometry_connectivity = (
                    owner_block.geometry_connectivity
                    if geometry_reference.polynomial_degree == 2
                    else owner_block.connectivity
                )
                coordinates = jnp.asarray(
                    self.mesh.points[geometry_connectivity[local_cells]]
                )
                points = owner_reference_points[indices]
                geometry_basis = geometry_reference.basis(points)
                geometry_gradients = geometry_reference.basis_gradients(points)
                mapped_points = jnp.einsum(
                    "fqi,fid->fqd", geometry_basis, coordinates,
                    precision=jax.lax.Precision.HIGHEST,
                )
                jacobian = jnp.einsum(
                    "fid,fqir->fqdr", coordinates, geometry_gradients,
                    precision=jax.lax.Precision.HIGHEST,
                )
                tangent_u = jnp.einsum(
                    "fqdr,fqr->fqd", jacobian, tangent_u_reference[indices],
                    precision=jax.lax.Precision.HIGHEST,
                )
                tangent_v = jnp.einsum(
                    "fqdr,fqr->fqd", jacobian, tangent_v_reference[indices],
                    precision=jax.lax.Precision.HIGHEST,
                )
                area_vectors = jnp.cross(tangent_u, tangent_v)
                outward = jnp.asarray(face_block.normals[indices_np])
                orientation = jnp.where(
                    jnp.sum(area_vectors * outward[:, None, :], axis=-1) >= 0,
                    1.0,
                    -1.0,
                )
                area_vectors = area_vectors * orientation[..., None]
                measures = jnp.linalg.norm(area_vectors, axis=-1)
                physical_points = physical_points.at[indices].set(mapped_points)
                normals = normals.at[indices].set(
                    area_vectors / measures[..., None]
                )
                integration_weights = integration_weights.at[indices].set(
                    normalized_weights[None, :] * reference_measure * measures
                )

            result[face_type] = FaceMetrics(
                physical_points=physical_points,
                normals=normals,
                integration_weights=integration_weights,
                owner_reference_points=owner_reference_points,
                neighbor_reference_points=neighbor_reference_points,
            )
        return result

    def _validate(self):
        for block_id, block in enumerate(self.mesh.element_blocks):
            metrics = self.element_metrics[block_id]
            computed_volume = np.asarray(
                jnp.sum(metrics.integration_weights, axis=1)
            )
            geometry_is_linear = (
                self.geometry_reference_elements[
                    block.element_type
                ].polynomial_degree == 1
            )
            if geometry_is_linear and not np.allclose(
                computed_volume, block.volumes, rtol=2.0e-6, atol=1.0e-12
            ):
                error = np.max(np.abs(computed_volume - block.volumes))
                raise ValueError(
                    f"{block.element_type} mapped-volume error is {error}"
                )
            if np.any(np.asarray(metrics.determinant) == 0.0):
                raise ValueError(f"Singular {block.element_type} mapping")

        for face_type, face_block in self.mesh.face_blocks.items():
            metrics = self.face_metrics[face_type]
            computed_area = np.asarray(
                jnp.sum(metrics.integration_weights, axis=1)
            )
            linear_geometry = all(
                self.geometry_reference_elements[block.element_type].polynomial_degree == 1
                for block in self.mesh.element_blocks
            )
            if linear_geometry and not np.allclose(computed_area, face_block.areas):
                raise ValueError(f"Incorrect {face_type} quadrature area")

            internal = face_block.internal
            if np.any(internal):
                owner_points = np.asarray(metrics.physical_points)[internal]
                # Both traces use the same physical face quadrature locations;
                # reference coordinates differ only by element mapping.
                if not np.all(np.isfinite(owner_points)):
                    raise ValueError("Invalid internal face quadrature points")
