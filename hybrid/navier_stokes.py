"""BR2 Navier--Stokes operator for mixed P1/P2 DG element blocks."""

import jax
import jax.numpy as jnp

from .spatial import HybridEulerOperator


class HybridNavierStokesOperator(HybridEulerOperator):
    def __init__(self, solver, br2_stabilization=None):
        super().__init__(solver)
        self.br2_stabilization = float(
            polynomial_br2_penalty(solver.polynomial_degree)
            if br2_stabilization is None
            else br2_stabilization
        )
        self.filter_widths = {}
        for block_id, block in enumerate(self.mesh.element_blocks):
            volumes = jnp.sum(
                self.geometry.element_metrics[block_id].integration_weights,
                axis=1,
            )
            self.filter_widths[block.element_type] = (
                volumes ** (1.0 / 3.0) / (solver.polynomial_degree + 1)
            )

    def _trace_data(self, states, group, boundary_state):
        face_type, owner_block_id, neighbor_block_id, indices = group
        faces = self.mesh.face_blocks[face_type]
        metrics = self.geometry.face_metrics[face_type]
        owner_block = self.mesh.element_blocks[owner_block_id]
        owner_reference = self.geometry.reference_elements[
            owner_block.element_type
        ]
        owner_cells = jnp.asarray(faces.owner_local)[indices]
        owner_points = metrics.owner_reference_points[indices]
        owner_basis = owner_reference.basis(owner_points)
        U_owner_nodes = states[owner_block.element_type][owner_cells]
        U_owner = jnp.einsum(
            "fqi,fiv->fqv", owner_basis, U_owner_nodes,
            precision=jax.lax.Precision.HIGHEST,
        )
        variables_owner_nodes = self.solver.viscous_variables(U_owner_nodes)
        variables_owner = jnp.einsum(
            "fqi,fim->fqm", owner_basis, variables_owner_nodes,
            precision=jax.lax.Precision.HIGHEST,
        )

        if neighbor_block_id >= 0:
            neighbor_block = self.mesh.element_blocks[neighbor_block_id]
            neighbor_reference = self.geometry.reference_elements[
                neighbor_block.element_type
            ]
            neighbor_cells = jnp.asarray(faces.neighbor_local)[indices]
            neighbor_points = metrics.neighbor_reference_points[indices]
            neighbor_basis = neighbor_reference.basis(neighbor_points)
            U_neighbor_nodes = states[neighbor_block.element_type][neighbor_cells]
            U_neighbor = jnp.einsum(
                "fqi,fiv->fqv", neighbor_basis, U_neighbor_nodes,
                precision=jax.lax.Precision.HIGHEST,
            )
            variables_neighbor_nodes = self.solver.viscous_variables(
                U_neighbor_nodes
            )
            variables_neighbor = jnp.einsum(
                "fqi,fim->fqm", neighbor_basis, variables_neighbor_nodes,
                precision=jax.lax.Precision.HIGHEST,
            )
        else:
            neighbor_block = neighbor_cells = neighbor_basis = None
            variables_neighbor_nodes = None
            U_neighbor = (
                U_owner if boundary_state is None
                else boundary_state(
                    U_owner,
                    metrics.normals[indices],
                    jnp.asarray(faces.boundary_tags)[indices],
                )
            )
            variables_neighbor = (
                self.solver.viscous_variables(U_neighbor)
                if boundary_state is None
                or not hasattr(boundary_state, "viscous_ghost")
                else boundary_state.viscous_ghost(
                    U_owner,
                    metrics.normals[indices],
                    jnp.asarray(faces.boundary_tags)[indices],
                )
            )

        return {
            "face_type": face_type,
            "indices": indices,
            "faces": faces,
            "metrics": metrics,
            "owner_block_id": owner_block_id,
            "owner_block": owner_block,
            "owner_cells": owner_cells,
            "owner_points": owner_points,
            "owner_basis": owner_basis,
            "U_owner": U_owner,
            "variables_owner_nodes": variables_owner_nodes,
            "neighbor_block_id": neighbor_block_id,
            "neighbor_block": neighbor_block,
            "neighbor_cells": neighbor_cells,
            "neighbor_basis": neighbor_basis,
            "U_neighbor": U_neighbor,
            "variables_neighbor_nodes": variables_neighbor_nodes,
            "jump": variables_neighbor - variables_owner,
        }

    def _physical_basis_gradients(self, block_id, cells, points):
        block = self.mesh.element_blocks[block_id]
        solution_reference = self.geometry.reference_elements[block.element_type]
        geometry_reference = self.geometry.geometry_reference_elements[
            block.element_type
        ]
        geometry_connectivity = (
            block.geometry_connectivity
            if geometry_reference.polynomial_degree == 2
            else block.connectivity
        )
        coordinates = jnp.asarray(
            self.mesh.points[geometry_connectivity]
        )[cells]
        geometry_gradients = geometry_reference.basis_gradients(points)
        jacobian = jnp.einsum(
            "fid,fqir->fqdr", coordinates, geometry_gradients,
            precision=jax.lax.Precision.HIGHEST,
        )
        inverse = jnp.linalg.inv(jacobian)
        solution_gradients = solution_reference.basis_gradients(points)
        return jnp.einsum(
            "fqir,fqrd->fqid", solution_gradients, inverse,
            precision=jax.lax.Precision.HIGHEST,
        )

    def _br2_gradients(self, states, boundary_state):
        lift_nodes = {
            block.element_type: jnp.zeros(
                states[block.element_type].shape[:2] + (4, 3),
                dtype=states[block.element_type].dtype,
            )
            for block in self.mesh.element_blocks
        }
        face_data = []

        for group in self.face_groups:
            data = self._trace_data(states, group, boundary_state)
            weights = data["metrics"].integration_weights[data["indices"]]
            normals = data["metrics"].normals[data["indices"]]
            owner_rhs = 0.5 * jnp.einsum(
                "fq,fqm,fqd,fqi->fimd",
                weights, data["jump"], normals, data["owner_basis"],
                precision=jax.lax.Precision.HIGHEST,
            )
            owner_inverse_mass = self.geometry.element_metrics[
                data["owner_block_id"]
            ].inverse_mass_matrix[data["owner_cells"]]
            owner_lift = jnp.einsum(
                "fij,fjmd->fimd", owner_inverse_mass, owner_rhs,
                precision=jax.lax.Precision.HIGHEST,
            )
            element_type = data["owner_block"].element_type
            lift_nodes[element_type] = lift_nodes[element_type].at[
                data["owner_cells"]
            ].add(owner_lift)
            data["owner_lift"] = owner_lift

            if data["neighbor_block_id"] >= 0:
                neighbor_rhs = 0.5 * jnp.einsum(
                    "fq,fqm,fqd,fqi->fimd",
                    weights, data["jump"], normals, data["neighbor_basis"],
                    precision=jax.lax.Precision.HIGHEST,
                )
                neighbor_inverse_mass = self.geometry.element_metrics[
                    data["neighbor_block_id"]
                ].inverse_mass_matrix[data["neighbor_cells"]]
                neighbor_lift = jnp.einsum(
                    "fij,fjmd->fimd", neighbor_inverse_mass, neighbor_rhs,
                    precision=jax.lax.Precision.HIGHEST,
                )
                neighbor_type = data["neighbor_block"].element_type
                lift_nodes[neighbor_type] = lift_nodes[neighbor_type].at[
                    data["neighbor_cells"]
                ].add(neighbor_lift)
                data["neighbor_lift"] = neighbor_lift
            face_data.append(data)

        raw_volume_gradients = {}
        volume_gradients = {}
        for block_id, block in enumerate(self.mesh.element_blocks):
            reference = self.geometry.reference_elements[block.element_type]
            metrics = self.geometry.element_metrics[block_id]
            variables = self.solver.viscous_variables(states[block.element_type])
            raw = jnp.einsum(
                "eim,eqid->eqmd", variables, metrics.physical_basis_gradients,
                precision=jax.lax.Precision.HIGHEST,
            )
            raw_volume_gradients[block.element_type] = raw
            lifting = jnp.einsum(
                "qi,eimd->eqmd",
                reference.basis(reference.volume_points),
                lift_nodes[block.element_type],
                precision=jax.lax.Precision.HIGHEST,
            )
            volume_gradients[block.element_type] = raw + lifting
        return volume_gradients, raw_volume_gradients, face_data

    def viscous_residual(self, states, boundary_state=None):
        volume_gradients, raw_volume_gradients, face_data = self._br2_gradients(
            states, boundary_state
        )
        residuals = {}
        for block_id, block in enumerate(self.mesh.element_blocks):
            reference = self.geometry.reference_elements[block.element_type]
            metrics = self.geometry.element_metrics[block_id]
            U_quadrature = jnp.einsum(
                "qi,eiv->eqv", reference.basis(reference.volume_points),
                states[block.element_type],
                precision=jax.lax.Precision.HIGHEST,
            )
            width = self.filter_widths[block.element_type][:, None]
            flux = self.solver.viscous_physical_flux(
                U_quadrature,
                volume_gradients[block.element_type],
                width,
                turbulent_gradient=raw_volume_gradients[block.element_type],
            )
            residuals[block.element_type] = -jnp.einsum(
                "eqid,eqdv,eq->eiv", metrics.physical_basis_gradients,
                flux, metrics.integration_weights,
                precision=jax.lax.Precision.HIGHEST,
            )

        for data in face_data:
            metrics = data["metrics"]
            indices = data["indices"]
            normals = metrics.normals[indices]
            weights = metrics.integration_weights[indices]
            owner_grad_basis = self._physical_basis_gradients(
                data["owner_block_id"], data["owner_cells"],
                data["owner_points"],
            )
            raw_owner = jnp.einsum(
                "fim,fqid->fqmd", data["variables_owner_nodes"],
                owner_grad_basis,
                precision=jax.lax.Precision.HIGHEST,
            )
            owner_local_lift = jnp.einsum(
                "fqi,fimd->fqmd", data["owner_basis"], data["owner_lift"],
                precision=jax.lax.Precision.HIGHEST,
            )
            gradient_owner = raw_owner + self.br2_stabilization * owner_local_lift
            width_owner = self.filter_widths[
                data["owner_block"].element_type
            ][data["owner_cells"]][:, None]
            flux_owner = self.solver.viscous_physical_flux(
                data["U_owner"], gradient_owner, width_owner,
                turbulent_gradient=raw_owner,
            )

            if data["neighbor_block_id"] >= 0:
                neighbor_points = metrics.neighbor_reference_points[indices]
                neighbor_grad_basis = self._physical_basis_gradients(
                    data["neighbor_block_id"], data["neighbor_cells"],
                    neighbor_points,
                )
                raw_neighbor = jnp.einsum(
                    "fim,fqid->fqmd", data["variables_neighbor_nodes"],
                    neighbor_grad_basis,
                    precision=jax.lax.Precision.HIGHEST,
                )
                neighbor_local_lift = jnp.einsum(
                    "fqi,fimd->fqmd", data["neighbor_basis"],
                    data["neighbor_lift"],
                    precision=jax.lax.Precision.HIGHEST,
                )
                gradient_neighbor = (
                    raw_neighbor
                    + self.br2_stabilization * neighbor_local_lift
                )
                width_neighbor = self.filter_widths[
                    data["neighbor_block"].element_type
                ][data["neighbor_cells"]][:, None]
                flux_neighbor = self.solver.viscous_physical_flux(
                    data["U_neighbor"], gradient_neighbor, width_neighbor,
                    turbulent_gradient=raw_neighbor,
                )
                face_flux = 0.5 * (flux_owner + flux_neighbor)
            else:
                face_flux = flux_owner

            normal_flux = jnp.einsum(
                "fqdv,fqd->fqv", face_flux, normals,
                precision=jax.lax.Precision.HIGHEST,
            )
            if data["neighbor_block_id"] < 0 and boundary_state is not None:
                tags = jnp.asarray(data["faces"].boundary_tags)[indices]
                if hasattr(boundary_state, "wall_tags"):
                    wall = boundary_state._matches(tags, boundary_state.wall_tags)
                    normal_flux = normal_flux.at[..., 4].set(
                        jnp.where(wall[:, None], 0.0, normal_flux[..., 4])
                    )
            weighted = normal_flux * weights[..., None]
            owner_surface = jnp.einsum(
                "fqi,fqv->fiv", data["owner_basis"], weighted,
                precision=jax.lax.Precision.HIGHEST,
            )
            owner_type = data["owner_block"].element_type
            residuals[owner_type] = residuals[owner_type].at[
                data["owner_cells"]
            ].add(owner_surface)

            if data["neighbor_block_id"] >= 0:
                neighbor_surface = jnp.einsum(
                    "fqi,fqv->fiv", data["neighbor_basis"], weighted,
                    precision=jax.lax.Precision.HIGHEST,
                )
                neighbor_type = data["neighbor_block"].element_type
                residuals[neighbor_type] = residuals[neighbor_type].at[
                    data["neighbor_cells"]
                ].add(-neighbor_surface)
        return residuals

    def viscous_rhs(self, states, boundary_state=None):
        residuals = self.viscous_residual(states, boundary_state)
        result = {}
        for block_id, block in enumerate(self.mesh.element_blocks):
            result[block.element_type] = jnp.einsum(
                "eij,ejv->eiv",
                self.geometry.element_metrics[block_id].inverse_mass_matrix,
                residuals[block.element_type],
                precision=jax.lax.Precision.HIGHEST,
            )
        return result

    def rhs(self, states, boundary_state=None):
        euler = self.inviscid_rhs(states, boundary_state)
        if self.solver.dynamic_viscosity == 0.0 and not self.solver.use_wale:
            return euler
        viscous = self.viscous_rhs(states, boundary_state)
        return {name: euler[name] + viscous[name] for name in euler}

    def inviscid_rhs(self, states, boundary_state=None):
        """Euler portion exposed for conservative DG/subcell recomputation."""
        return super().rhs(states, boundary_state)


def polynomial_br2_penalty(degree):
    # A conservative default for mixed faces; tune only after convergence tests.
    return (degree + 1) * (degree + 2) / 2
