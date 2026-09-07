"""Conservative mixed-element DG Euler spatial operator."""

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import HybridGeometry


class HybridEulerOperator:
    def __init__(self, solver):
        self.solver = solver
        self.mesh = solver.mesh
        self.geometry = HybridGeometry(
            self.mesh, solver.polynomial_degree
        )
        self.face_groups = self._group_faces()
        self.metric_closure = self._build_metric_closure()

    def _group_faces(self):
        groups = []
        for face_type, faces in self.mesh.face_blocks.items():
            keys = set()
            for face_index in range(len(faces.nodes)):
                keys.add(
                    (
                        int(faces.owner_block[face_index]),
                        int(faces.neighbor_block[face_index]),
                    )
                )
            for owner_block, neighbor_block in sorted(keys):
                indices = np.flatnonzero(
                    (faces.owner_block == owner_block)
                    & (faces.neighbor_block == neighbor_block)
                ).astype(np.int32)
                groups.append(
                    (face_type, owner_block, neighbor_block, jnp.asarray(indices))
                )
        return groups

    def _cell_mean_states(self, states):
        # A nodal anchor, not a mathematical cell average. Reconstructing the
        # deviation from this anchor makes constant states exactly constant in
        # float32 even when basis functions sum to 1 only within roundoff.
        return {name: U[:, 0, :] for name, U in states.items()}

    def _volume_terms(self, states, mean_states):
        residuals = {}
        for block_id, block in enumerate(self.mesh.element_blocks):
            reference = self.geometry.reference_elements[block.element_type]
            metrics = self.geometry.element_metrics[block_id]
            U = states[block.element_type]
            anchor = mean_states[block.element_type]
            U_at_quadrature = anchor[:, None, :] + jnp.einsum(
                "qi,eiv->eqv",
                reference.basis(reference.volume_points),
                U - anchor[:, None, :],
                precision=jax.lax.Precision.HIGHEST,
            )
            flux = self.solver.physical_flux_tensor(U_at_quadrature)
            mean_flux = self.solver.physical_flux_tensor(
                mean_states[block.element_type]
            )
            volume_term = jnp.einsum(
                "eqid,eqdv,eq->eiv",
                metrics.physical_basis_gradients,
                flux - mean_flux[:, None, :, :],
                metrics.integration_weights,
                precision=jax.lax.Precision.HIGHEST,
            )
            residuals[block.element_type] = volume_term
        return residuals

    def _build_metric_closure(self):
        surface_moments = {
            block.element_type: jnp.zeros(
                (
                    len(block.connectivity),
                    len(self.geometry.reference_elements[block.element_type].nodes),
                    3,
                )
            )
            for block in self.mesh.element_blocks
        }
        for face_type, owner_block_id, neighbor_block_id, indices in self.face_groups:
            faces = self.mesh.face_blocks[face_type]
            metrics = self.geometry.face_metrics[face_type]
            normals = metrics.normals[indices]
            weights = metrics.integration_weights[indices]
            owner_block = self.mesh.element_blocks[owner_block_id]
            owner_reference = self.geometry.reference_elements[
                owner_block.element_type
            ]
            owner_cells = jnp.asarray(faces.owner_local)[indices]
            owner_basis = owner_reference.basis(
                metrics.owner_reference_points[indices]
            )
            owner_moment = jnp.einsum(
                "fqi,fqd,fq->fid", owner_basis, normals, weights,
                precision=jax.lax.Precision.HIGHEST,
            )
            surface_moments[owner_block.element_type] = surface_moments[
                owner_block.element_type
            ].at[owner_cells].add(owner_moment)

            if neighbor_block_id >= 0:
                neighbor_block = self.mesh.element_blocks[neighbor_block_id]
                neighbor_reference = self.geometry.reference_elements[
                    neighbor_block.element_type
                ]
                neighbor_cells = jnp.asarray(faces.neighbor_local)[indices]
                neighbor_basis = neighbor_reference.basis(
                    metrics.neighbor_reference_points[indices]
                )
                neighbor_moment = -jnp.einsum(
                    "fqi,fqd,fq->fid", neighbor_basis, normals, weights,
                    precision=jax.lax.Precision.HIGHEST,
                )
                surface_moments[neighbor_block.element_type] = surface_moments[
                    neighbor_block.element_type
                ].at[neighbor_cells].add(neighbor_moment)

        closure = {}
        for block_id, block in enumerate(self.mesh.element_blocks):
            metrics = self.geometry.element_metrics[block_id]
            volume_moment = jnp.einsum(
                "eqid,eq->eid",
                metrics.physical_basis_gradients,
                metrics.integration_weights,
                precision=jax.lax.Precision.HIGHEST,
            )
            closure[block.element_type] = (
                surface_moments[block.element_type] - volume_moment
            )
        return closure

    def residual(self, states, boundary_state=None):
        """Return weak-form residual before multiplication by -M inverse."""
        mean_states = self._cell_mean_states(states)
        residuals = self._volume_terms(states, mean_states)

        for face_type, owner_block_id, neighbor_block_id, indices in self.face_groups:
            faces = self.mesh.face_blocks[face_type]
            metrics = self.geometry.face_metrics[face_type]
            owner_block = self.mesh.element_blocks[owner_block_id]
            owner_reference = self.geometry.reference_elements[
                owner_block.element_type
            ]
            owner_cells = jnp.asarray(faces.owner_local)[indices]
            owner_points = metrics.owner_reference_points[indices]
            owner_basis = owner_reference.basis(owner_points)
            owner_anchor = mean_states[owner_block.element_type][owner_cells]
            U_owner = owner_anchor[:, None, :] + jnp.einsum(
                "fqi,fiv->fqv",
                owner_basis,
                states[owner_block.element_type][owner_cells]
                - owner_anchor[:, None, :],
                precision=jax.lax.Precision.HIGHEST,
            )
            normals = metrics.normals[indices]

            if neighbor_block_id >= 0:
                neighbor_block = self.mesh.element_blocks[neighbor_block_id]
                neighbor_reference = self.geometry.reference_elements[
                    neighbor_block.element_type
                ]
                neighbor_cells = jnp.asarray(faces.neighbor_local)[indices]
                neighbor_points = metrics.neighbor_reference_points[indices]
                neighbor_basis = neighbor_reference.basis(neighbor_points)
                neighbor_anchor = mean_states[
                    neighbor_block.element_type
                ][neighbor_cells]
                U_neighbor = neighbor_anchor[:, None, :] + jnp.einsum(
                    "fqi,fiv->fqv",
                    neighbor_basis,
                    states[neighbor_block.element_type][neighbor_cells]
                    - neighbor_anchor[:, None, :],
                    precision=jax.lax.Precision.HIGHEST,
                )
            else:
                neighbor_cells = None
                neighbor_basis = None
                U_neighbor = (
                    U_owner
                    if boundary_state is None
                    else boundary_state(
                        U_owner,
                        normals,
                        jnp.asarray(faces.boundary_tags)[indices],
                    )
                )

            numerical_flux = self.solver.numerical_flux(
                U_owner, U_neighbor, normals
            )
            owner_mean = mean_states[owner_block.element_type][owner_cells]
            owner_baseline = self.solver.normal_physical_flux(
                owner_mean[:, None, :], normals
            )
            weighted_flux = (
                (numerical_flux - owner_baseline)
                * metrics.integration_weights[indices, :, None]
            )
            owner_surface = jnp.einsum(
                "fqi,fqv->fiv",
                owner_basis,
                weighted_flux,
                precision=jax.lax.Precision.HIGHEST,
            )
            residuals[owner_block.element_type] = residuals[
                owner_block.element_type
            ].at[owner_cells].add(-owner_surface)

            if neighbor_block_id >= 0:
                neighbor_mean = mean_states[
                    neighbor_block.element_type
                ][neighbor_cells]
                neighbor_baseline = self.solver.normal_physical_flux(
                    neighbor_mean[:, None, :], normals
                )
                neighbor_weighted_flux = (
                    numerical_flux - neighbor_baseline
                ) * metrics.integration_weights[indices, :, None]
                neighbor_surface = jnp.einsum(
                    "fqi,fqv->fiv",
                    neighbor_basis,
                    neighbor_weighted_flux,
                    precision=jax.lax.Precision.HIGHEST,
                )
                residuals[neighbor_block.element_type] = residuals[
                    neighbor_block.element_type
                ].at[neighbor_cells].add(neighbor_surface)

        return residuals

    def rhs(self, states, boundary_state=None):
        residuals = self.residual(states, boundary_state)
        result = {}
        for block_id, block in enumerate(self.mesh.element_blocks):
            inverse_mass = self.geometry.element_metrics[
                block_id
            ].inverse_mass_matrix
            result[block.element_type] = jnp.einsum(
                "eij,ejv->eiv",
                inverse_mass,
                residuals[block.element_type],
                precision=jax.lax.Precision.HIGHEST,
            )
        return result
