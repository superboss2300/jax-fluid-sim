import argparse
import os
import time
from pathlib import Path

import jax
# Verification remains float64 by default. Large production meshes can opt into
# float32 before import with JAX_ENABLE_X64=0, which roughly halves GPU memory.
if "JAX_ENABLE_X64" not in os.environ:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from hybrid.output import _robust_limits, shock_colormap
from mesh_loader import MeshLoader
from taylor_green import PERIODIC_PAIRS


class TetrahedralDG:
    def __init__(
        self,
        mesh_path,
        use_wale=False,
        inviscid_flux="hllc",
        polynomial_degree=2,
        use_shock_limiter=False,
        shock_sensor_threshold=0.1,
        farfield_density=1.0,
        farfield_pressure=1.0,
        farfield_mach=0.2,
        reynolds=1600.0,
        reynolds_reference_density=1.0,
        reynolds_reference_velocity=1.0,
        reynolds_reference_length=1.0,
    ):
        self.mesh = MeshLoader(mesh_path)
        self.use_wale = use_wale

        if inviscid_flux not in {"hllc", "rusanov"}:
            raise ValueError(
                "inviscid_flux must be 'hllc' or 'rusanov'"
            )
        self.inviscid_flux = inviscid_flux

        self.wale_constant = 0.5
        self.turbulent_prandtl = 0.9

        self.cfl = 0.2
        if polynomial_degree not in {1, 2}:
            raise ValueError("polynomial_degree must be 1 or 2")
        self.polynomial_degree = polynomial_degree
        self.use_shock_limiter = use_shock_limiter
        self.shock_sensor_threshold = shock_sensor_threshold

        self.gamma = 1.4
        self.number_of_variables = 5

        self.gas_constant = 1.0
        self.br2_stabilization = float(
            (self.polynomial_degree + 1) ** 2
        )

        self.reynolds = reynolds
        self.reynolds_reference_density = reynolds_reference_density
        self.reynolds_reference_velocity = reynolds_reference_velocity
        self.reynolds_reference_length = reynolds_reference_length
        self.prandtl = 0.72

        self.dynamic_viscosity = (
            self.reynolds_reference_density
            * self.reynolds_reference_velocity
            * self.reynolds_reference_length
            / self.reynolds
        )

        self.specific_heat_cp = (
            self.gamma
            * self.gas_constant
            / (self.gamma - 1.0)
        )

        self.thermal_conductivity = (
            self.dynamic_viscosity
            * self.specific_heat_cp
            / self.prandtl
        )

        periodic_names = {
            name
            for minus_name, plus_name, _ in PERIODIC_PAIRS
            for name in (minus_name, plus_name)
        }
        if periodic_names.issubset(self.mesh.physical_names):
            self.mesh.pair_periodic_boundaries(PERIODIC_PAIRS)

        self.points = jnp.asarray(
            self.mesh.points
        )

        farfield_sound_speed = jnp.sqrt(
            self.gamma * farfield_pressure / farfield_density
        )
        self.farfield_primitive = jnp.asarray(
            [
                farfield_density,
                farfield_mach * farfield_sound_speed,
                0.0,
                0.0,
                farfield_pressure,
            ],
            dtype=self.points.dtype,
        )
        self.inlet_tag = self.mesh.physical_names.get("inlet", -1)
        self.outlet_tag = self.mesh.physical_names.get("outlet", -1)
        self.walls_tag = self.mesh.physical_names.get("walls", -1)
        self.sphere_wall_tag = self.mesh.physical_names.get(
            "sphere_wall", -1
        )

        self.elements = jnp.asarray(
            self.mesh.tetrahedra
        )
        self.geometry_order = self.mesh.geometry_order
        self.geometry_connectivity = jnp.asarray(
            self.mesh.tetrahedra_geometry
        )
        self.geometry_nodes = self.points[
            self.geometry_connectivity
        ]

        self.number_of_elements = (
            self.elements.shape[0]
        )

        self.element_vertices = self.points[self.elements]

        vertex_reference_nodes = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=self.points.dtype)

        if self.polynomial_degree == 1:
            self.reference_nodes = vertex_reference_nodes
        else:
            # Vertices followed by edge midpoints:
            # 01, 02, 03, 12, 13, 23.
            self.reference_nodes = jnp.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ], dtype=self.points.dtype)

        self.nodes_per_element = self.reference_nodes.shape[0]
        self.solution_nodes = jnp.einsum(
            "ni,eid->end",
            self.lagrange_basis_at_reference_points(
                self.reference_nodes,
                self.geometry_order,
            ),
            self.geometry_nodes,
        )

        edge_vectors = (
            self.element_vertices[:, 1:, :]
            - self.element_vertices[:, :1, :]
        )

        self.jacobian = jnp.swapaxes(
            edge_vectors,
            1,
            2,
        )

        self.affine_inverse_jacobian = jnp.linalg.inv(
            self.jacobian
        )

        self.affine_determinant = jnp.linalg.det(
            self.jacobian
        )

        self.owner = jnp.asarray(
            self.mesh.internal_owner
        )

        self.neighbor = jnp.asarray(
            self.mesh.internal_neighbor
        )

        self.linear_face_normals = jnp.asarray(
            self.mesh.internal_face_normals
        )

        self.linear_face_areas = jnp.asarray(
            self.mesh.internal_face_areas
        )

        self.owner_face_nodes = jnp.asarray(
            self.mesh.internal_owner_face_nodes
        )

        self.neighbor_face_nodes = jnp.asarray(
            self.mesh.internal_neighbor_face_nodes
        )

        if self.geometry_order == 2:
            triangle_rs = jnp.array([
                [0.249286745170910, 0.249286745170910],
                [0.249286745170910, 0.501426509658179],
                [0.501426509658179, 0.249286745170910],
                [0.063089014491502, 0.063089014491502],
                [0.063089014491502, 0.873821971016996],
                [0.873821971016996, 0.063089014491502],
                [0.310352451033785, 0.636502499121399],
                [0.636502499121399, 0.053145049844816],
                [0.053145049844816, 0.310352451033785],
                [0.310352451033785, 0.053145049844816],
                [0.636502499121399, 0.310352451033785],
                [0.053145049844816, 0.636502499121399],
            ], dtype=self.points.dtype)
            self.triangle_quadrature_coordinates = jnp.concatenate(
                [
                    (1.0 - jnp.sum(triangle_rs, axis=-1))[:, None],
                    triangle_rs,
                ],
                axis=-1,
            )
            triangle_reference_weights = jnp.array([
                0.058393137863189,
                0.058393137863189,
                0.058393137863189,
                0.025422453185104,
                0.025422453185104,
                0.025422453185104,
                0.041425537809187,
                0.041425537809187,
                0.041425537809187,
                0.041425537809187,
                0.041425537809187,
                0.041425537809187,
            ], dtype=self.points.dtype)
        elif self.polynomial_degree == 1:
            self.triangle_quadrature_coordinates = jnp.array([
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ], dtype=self.points.dtype)
            triangle_reference_weights = jnp.full(
                3, 1.0 / 6.0, dtype=self.points.dtype
            )
        else:
            triangle_a = 0.816847572980459
            triangle_b = 0.091576213509771
            triangle_c = 0.108103018168070
            triangle_d = 0.445948490915965
            self.triangle_quadrature_coordinates = jnp.array([
                [triangle_a, triangle_b, triangle_b],
                [triangle_b, triangle_a, triangle_b],
                [triangle_b, triangle_b, triangle_a],
                [triangle_c, triangle_d, triangle_d],
                [triangle_d, triangle_c, triangle_d],
                [triangle_d, triangle_d, triangle_c],
            ], dtype=self.points.dtype)
            triangle_reference_weights = 0.5 * jnp.array([
                0.109951743655322,
                0.109951743655322,
                0.109951743655322,
                0.223381589678011,
                0.223381589678011,
                0.223381589678011,
            ], dtype=self.points.dtype)

        owner_face_vertices = self.points[
            self.owner_face_nodes
        ]

        neighbor_face_vertices = self.points[
            self.neighbor_face_nodes
        ]

        self.owner_face_quadrature_points = jnp.einsum(
            "qv,fvd->fqd",
            self.triangle_quadrature_coordinates,
            owner_face_vertices,
        )

        self.neighbor_face_quadrature_points = jnp.einsum(
            "qv,fvd->fqd",
            self.triangle_quadrature_coordinates,
            neighbor_face_vertices,
        )

        self.owner_trace_basis = (
            self.basis_at_physical_points(
                self.owner,
                self.owner_face_quadrature_points,
            )
        )

        self.neighbor_trace_basis = (
            self.basis_at_physical_points(
                self.neighbor,
                self.neighbor_face_quadrature_points,
            )
        )

        owner_reference_points = self.reference_coordinates_at_physical_points(
            self.owner,
            self.owner_face_quadrature_points,
        )
        neighbor_reference_points = self.reference_coordinates_at_physical_points(
            self.neighbor,
            self.neighbor_face_quadrature_points,
        )
        owner_geometry_jacobian = self.geometry_jacobian_at_reference_points(
            self.owner,
            owner_reference_points,
        )
        owner_geometry_inverse = jnp.linalg.inv(owner_geometry_jacobian)
        owner_geometry_determinant = jnp.linalg.det(owner_geometry_jacobian)

        self.owner_trace_physical_basis_gradients = jnp.einsum(
            "fqir,fqrx->fqix",
            self.basis_gradients_at_reference_points(owner_reference_points),
            owner_geometry_inverse,
        )
        neighbor_geometry_jacobian = self.geometry_jacobian_at_reference_points(
            self.neighbor,
            neighbor_reference_points,
        )
        neighbor_geometry_inverse = jnp.linalg.inv(neighbor_geometry_jacobian)
        self.neighbor_trace_physical_basis_gradients = jnp.einsum(
            "fqir,fqrx->fqix",
            self.basis_gradients_at_reference_points(neighbor_reference_points),
            neighbor_geometry_inverse,
        )

        linear_area_vectors = (
            2.0
            * self.linear_face_areas[:, None]
            * self.linear_face_normals
        )
        reference_area_vectors = jnp.einsum(
            "fdr,fd->fr",
            self.jacobian[self.owner],
            linear_area_vectors,
        ) / self.affine_determinant[self.owner, None]
        curved_area_vectors = (
            owner_geometry_determinant[..., None]
            * jnp.einsum(
                "fqrx,fr->fqx",
                owner_geometry_inverse,
                reference_area_vectors,
            )
        )
        face_metric = jnp.linalg.norm(curved_area_vectors, axis=-1)
        self.face_normals = curved_area_vectors / face_metric[..., None]
        self.face_quadrature_weights = (
            face_metric * triangle_reference_weights[None, :]
        )
        self.face_areas = jnp.sum(self.face_quadrature_weights, axis=1)

        owner_geometry_basis = self.lagrange_basis_at_reference_points(
            owner_reference_points,
            self.geometry_order,
        )
        neighbor_geometry_basis = self.lagrange_basis_at_reference_points(
            neighbor_reference_points,
            self.geometry_order,
        )
        self.owner_face_quadrature_points = jnp.einsum(
            "fqa,fad->fqd",
            owner_geometry_basis,
            self.geometry_nodes[self.owner],
        )
        self.neighbor_face_quadrature_points = jnp.einsum(
            "fqa,fad->fqd",
            neighbor_geometry_basis,
            self.geometry_nodes[self.neighbor],
        )

        self.boundary_owner = jnp.asarray(self.mesh.boundary_owner)
        self.boundary_face_nodes = jnp.asarray(self.mesh.boundary_face_nodes)
        self.boundary_tags = jnp.asarray(self.mesh.boundary_tags)
        number_of_boundary_faces = self.boundary_owner.shape[0]

        if number_of_boundary_faces:
            boundary_face_vertices = self.points[self.boundary_face_nodes]
            boundary_planar_points = jnp.einsum(
                "qv,fvd->fqd",
                self.triangle_quadrature_coordinates,
                boundary_face_vertices,
            )
            boundary_reference_points = (
                self.reference_coordinates_at_physical_points(
                    self.boundary_owner,
                    boundary_planar_points,
                )
            )
            self.boundary_trace_basis = self.basis_at_reference_points(
                boundary_reference_points
            )

            boundary_geometry_jacobian = (
                self.geometry_jacobian_at_reference_points(
                    self.boundary_owner,
                    boundary_reference_points,
                )
            )
            boundary_geometry_inverse = jnp.linalg.inv(
                boundary_geometry_jacobian
            )
            boundary_geometry_determinant = jnp.linalg.det(
                boundary_geometry_jacobian
            )
            self.boundary_trace_physical_basis_gradients = jnp.einsum(
                "fqir,fqrx->fqix",
                self.basis_gradients_at_reference_points(
                    boundary_reference_points
                ),
                boundary_geometry_inverse,
            )

            linear_boundary_normals = jnp.asarray(
                self.mesh.boundary_face_normals
            )
            linear_boundary_areas = jnp.asarray(
                self.mesh.boundary_face_areas
            )
            linear_boundary_area_vectors = (
                2.0
                * linear_boundary_areas[:, None]
                * linear_boundary_normals
            )
            boundary_reference_area_vectors = jnp.einsum(
                "fdr,fd->fr",
                self.jacobian[self.boundary_owner],
                linear_boundary_area_vectors,
            ) / self.affine_determinant[self.boundary_owner, None]
            curved_boundary_area_vectors = (
                boundary_geometry_determinant[..., None]
                * jnp.einsum(
                    "fqrx,fr->fqx",
                    boundary_geometry_inverse,
                    boundary_reference_area_vectors,
                )
            )
            boundary_metric = jnp.linalg.norm(
                curved_boundary_area_vectors,
                axis=-1,
            )
            self.boundary_normals = (
                curved_boundary_area_vectors
                / boundary_metric[..., None]
            )
            self.boundary_quadrature_weights = (
                boundary_metric
                * triangle_reference_weights[None, :]
            )
            self.boundary_areas = jnp.sum(
                self.boundary_quadrature_weights,
                axis=1,
            )
            boundary_geometry_basis = (
                self.lagrange_basis_at_reference_points(
                    boundary_reference_points,
                    self.geometry_order,
                )
            )
            self.boundary_quadrature_points = jnp.einsum(
                "fqa,fad->fqd",
                boundary_geometry_basis,
                self.geometry_nodes[self.boundary_owner],
            )
        else:
            number_of_face_points = self.triangle_quadrature_coordinates.shape[0]
            self.boundary_trace_basis = jnp.zeros(
                (0, number_of_face_points, self.nodes_per_element),
                dtype=self.points.dtype,
            )
            self.boundary_trace_physical_basis_gradients = jnp.zeros(
                (0, number_of_face_points, self.nodes_per_element, 3),
                dtype=self.points.dtype,
            )
            self.boundary_normals = jnp.zeros(
                (0, number_of_face_points, 3),
                dtype=self.points.dtype,
            )
            self.boundary_quadrature_weights = jnp.zeros(
                (0, number_of_face_points),
                dtype=self.points.dtype,
            )
            self.boundary_areas = jnp.zeros((0,), dtype=self.points.dtype)
            self.boundary_quadrature_points = jnp.zeros(
                (0, number_of_face_points, 3),
                dtype=self.points.dtype,
            )

        self.advection_velocity = jnp.array([
            1.0,
            0.5,
            0.25,
        ])

        if self.geometry_order == 2:
            (
                volume_reference_points,
                self.reference_volume_weights,
            ) = self.curved_volume_quadrature(self.points.dtype)
            volume_barycentric = jnp.concatenate(
                [
                    (1.0 - jnp.sum(volume_reference_points, axis=-1))[:, None],
                    volume_reference_points,
                ],
                axis=-1,
            )
        elif self.polynomial_degree == 1:
            a = 0.5854101966249685
            b = 0.1381966011250105
            volume_barycentric = jnp.array([
                [a, b, b, b],
                [b, a, b, b],
                [b, b, a, b],
                [b, b, b, a],
            ], dtype=self.points.dtype)
            self.reference_volume_weights = jnp.full(
                4, 1.0 / 24.0, dtype=self.points.dtype
            )
        else:
            # Positive 14-point, degree-five tetrahedron rule.
            group_1_a = 0.7217942490673264
            group_1_b = 0.0927352503108912
            group_2_a = 0.0673422422100982
            group_2_b = 0.3108859192633006
            pair_a = 0.4544962958743504
            pair_b = 0.04550370412564965
            volume_barycentric = jnp.array([
                [group_1_a, group_1_b, group_1_b, group_1_b],
                [group_1_b, group_1_a, group_1_b, group_1_b],
                [group_1_b, group_1_b, group_1_a, group_1_b],
                [group_1_b, group_1_b, group_1_b, group_1_a],
                [group_2_a, group_2_b, group_2_b, group_2_b],
                [group_2_b, group_2_a, group_2_b, group_2_b],
                [group_2_b, group_2_b, group_2_a, group_2_b],
                [group_2_b, group_2_b, group_2_b, group_2_a],
                [pair_a, pair_a, pair_b, pair_b],
                [pair_a, pair_b, pair_a, pair_b],
                [pair_a, pair_b, pair_b, pair_a],
                [pair_b, pair_a, pair_a, pair_b],
                [pair_b, pair_a, pair_b, pair_a],
                [pair_b, pair_b, pair_a, pair_a],
            ], dtype=self.points.dtype)
            self.reference_volume_weights = jnp.array([
                0.01224884051939366,
                0.01224884051939366,
                0.01224884051939366,
                0.01224884051939366,
                0.01878132095300264,
                0.01878132095300264,
                0.01878132095300264,
                0.01878132095300264,
                0.007091003462846911,
                0.007091003462846911,
                0.007091003462846911,
                0.007091003462846911,
                0.007091003462846911,
                0.007091003462846911,
            ], dtype=self.points.dtype)

        volume_reference_points = volume_barycentric[:, 1:]
        self.volume_basis = self.basis_at_reference_points(
            volume_reference_points
        )

        geometry_basis_at_volume = self.lagrange_basis_at_reference_points(
            volume_reference_points,
            self.geometry_order,
        )
        self.volume_quadrature_points = jnp.einsum(
            "qi,eid->eqd",
            geometry_basis_at_volume,
            self.geometry_nodes,
        )

        volume_jacobian = self.geometry_jacobian_at_reference_points(
            jnp.arange(self.number_of_elements),
            jnp.broadcast_to(
                volume_reference_points[None, :, :],
                (self.number_of_elements,) + volume_reference_points.shape,
            ),
        )
        volume_inverse_jacobian = jnp.linalg.inv(volume_jacobian)
        self.volume_determinant = jnp.linalg.det(volume_jacobian)
        self.volume_quadrature_weights = (
            jnp.abs(self.volume_determinant)
            * self.reference_volume_weights[None, :]
        )
        self.element_volumes = jnp.sum(
            self.volume_quadrature_weights,
            axis=1,
        )
        self.element_filter_width = (
            self.element_volumes ** (1.0 / 3.0)
            / (self.polynomial_degree + 1)
        )

        reference_volume_gradients = (
            self.basis_gradients_at_reference_points(
                volume_reference_points
            )
        )
        self.volume_physical_basis_gradients = jnp.einsum(
            "qir,eqrx->eqix",
            reference_volume_gradients,
            volume_inverse_jacobian,
        )

        center_reference_point = jnp.array(
            [[0.25, 0.25, 0.25]],
            dtype=self.points.dtype,
        )
        center_points = jnp.broadcast_to(
            center_reference_point[None, :, :],
            (self.number_of_elements, 1, 3),
        )
        center_inverse_jacobian = jnp.linalg.inv(
            self.geometry_jacobian_at_reference_points(
                jnp.arange(self.number_of_elements),
                center_points,
            )
        )
        self.center_physical_basis_gradients = jnp.einsum(
            "qir,eqrx->eqix",
            self.basis_gradients_at_reference_points(center_reference_point),
            center_inverse_jacobian,
        )[:, 0]

        self.reference_mass = jnp.einsum(
            "q,qi,qj->ij",
            self.reference_volume_weights,
            self.volume_basis,
            self.volume_basis,
        )
        self.inverse_reference_mass = jnp.linalg.inv(self.reference_mass)
        if self.geometry_order == 2:
            self.mass_matrices = jnp.einsum(
                "eq,qi,qj->eij",
                self.volume_quadrature_weights,
                self.volume_basis,
                self.volume_basis,
            )
            self.inverse_mass_matrices = jnp.linalg.inv(
                self.mass_matrices
            )
        else:
            self.mass_matrices = None
            self.inverse_mass_matrices = None

    @staticmethod
    def curved_volume_quadrature(dtype):
        """Degree-seven tetrahedron quadrature for curved P2 metrics."""
        points = jnp.array([
            [0.5, 0.5, 0.0], [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5], [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0], [0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0782131923303, 0.0782131923303, 0.0782131923303],
            [0.0782131923303, 0.0782131923303, 0.765360423009],
            [0.0782131923303, 0.765360423009, 0.0782131923303],
            [0.765360423009, 0.0782131923303, 0.0782131923303],
            [0.121843216664, 0.121843216664, 0.121843216664],
            [0.121843216664, 0.121843216664, 0.634470350008],
            [0.121843216664, 0.634470350008, 0.121843216664],
            [0.634470350008, 0.121843216664, 0.121843216664],
            [0.332539164446, 0.332539164446, 0.332539164446],
            [0.332539164446, 0.332539164446, 0.00238250666074],
            [0.332539164446, 0.00238250666074, 0.332539164446],
            [0.00238250666074, 0.332539164446, 0.332539164446],
            [0.1, 0.1, 0.2], [0.1, 0.2, 0.1],
            [0.1, 0.1, 0.6], [0.1, 0.6, 0.1],
            [0.1, 0.2, 0.6], [0.1, 0.6, 0.2],
            [0.2, 0.1, 0.1], [0.2, 0.1, 0.6],
            [0.2, 0.6, 0.1], [0.6, 0.1, 0.2],
            [0.6, 0.1, 0.1], [0.6, 0.2, 0.1],
        ], dtype=dtype)
        weights = jnp.array([
            0.000970017636685, 0.000970017636685,
            0.000970017636685, 0.000970017636685,
            0.000970017636685, 0.000970017636685,
            0.018264223466167,
            0.010599941524417, 0.010599941524417,
            0.010599941524417, 0.010599941524417,
            -0.062517740114333, -0.062517740114333,
            -0.062517740114333, -0.062517740114333,
            0.004891425263067, 0.004891425263067,
            0.004891425263067, 0.004891425263067,
            0.027557319224, 0.027557319224, 0.027557319224,
            0.027557319224, 0.027557319224, 0.027557319224,
            0.027557319224, 0.027557319224, 0.027557319224,
            0.027557319224, 0.027557319224, 0.027557319224,
        ], dtype=dtype)
        return points, weights

    @staticmethod
    def lagrange_basis_at_reference_points(
        reference_coordinates,
        degree,
    ):
        r = reference_coordinates[..., 0]
        s = reference_coordinates[..., 1]
        t = reference_coordinates[..., 2]
        barycentric = jnp.stack(
            [1.0 - r - s - t, r, s, t],
            axis=-1,
        )

        if degree == 1:
            return barycentric

        l0, l1, l2, l3 = [barycentric[..., i] for i in range(4)]
        return jnp.stack(
            [
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                l3 * (2.0 * l3 - 1.0),
                4.0 * l0 * l1,
                4.0 * l0 * l2,
                4.0 * l0 * l3,
                4.0 * l1 * l2,
                4.0 * l1 * l3,
                4.0 * l2 * l3,
            ],
            axis=-1,
        )

    @staticmethod
    def lagrange_basis_gradients_at_reference_points(
        reference_coordinates,
        degree,
    ):
        barycentric = jnp.concatenate(
            [
                (1.0 - jnp.sum(reference_coordinates, axis=-1))[..., None],
                reference_coordinates,
            ],
            axis=-1,
        )
        barycentric_gradients = jnp.array([
            [-1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0],
        ], dtype=reference_coordinates.dtype)

        if degree == 1:
            return jnp.broadcast_to(
                barycentric_gradients,
                reference_coordinates.shape[:-1] + (4, 3),
            )

        vertex_gradients = (
            (4.0 * barycentric - 1.0)[..., :, None]
            * barycentric_gradients
        )
        edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        edge_gradients = jnp.stack(
            [
                4.0
                * (
                    barycentric[..., i, None] * barycentric_gradients[j]
                    + barycentric[..., j, None] * barycentric_gradients[i]
                )
                for i, j in edge_pairs
            ],
            axis=-2,
        )
        return jnp.concatenate(
            [vertex_gradients, edge_gradients],
            axis=-2,
        )

    def basis_at_reference_points(self, reference_coordinates):
        return self.lagrange_basis_at_reference_points(
            reference_coordinates,
            self.polynomial_degree,
        )

    def basis_gradients_at_reference_points(self, reference_coordinates):
        return self.lagrange_basis_gradients_at_reference_points(
            reference_coordinates,
            self.polynomial_degree,
        )

    def geometry_jacobian_at_reference_points(
        self,
        element_ids,
        reference_coordinates,
    ):
        geometry_gradients = (
            self.lagrange_basis_gradients_at_reference_points(
                reference_coordinates,
                self.geometry_order,
            )
        )
        return jnp.einsum(
            "fai,fqar->fqir",
            self.geometry_nodes[element_ids],
            geometry_gradients,
        )

    def reference_coordinates_at_physical_points(
        self,
        element_ids,
        physical_points,
    ):
        element_origin = self.element_vertices[element_ids, 0]
        displacement = physical_points - element_origin[:, None, :]
        return jnp.einsum(
            "erx,eqx->eqr",
            self.affine_inverse_jacobian[element_ids],
            displacement,
        )

    def basis_at_physical_points(
        self,
        element_ids,
        physical_points,
    ):
        return self.basis_at_reference_points(
            self.reference_coordinates_at_physical_points(
                element_ids,
                physical_points,
            )
        )

    def interpolate_internal_traces(
        self,
        state,
    ):
        left_state = jnp.einsum(
            "fqi,fi...->fq...",
            self.owner_trace_basis,
            state[self.owner],
        )

        right_state = jnp.einsum(
            "fqi,fi...->fq...",
            self.neighbor_trace_basis,
            state[self.neighbor],
        )

        return left_state, right_state

    def interpolate_boundary_trace(self, state):
        return jnp.einsum(
            "fqi,fi...->fq...",
            self.boundary_trace_basis,
            state[self.boundary_owner],
        )

    def boundary_masks(self):
        """Return inlet, outlet, and solid-wall masks for physical faces."""
        inlet = self.boundary_tags == self.inlet_tag
        outlet = self.boundary_tags == self.outlet_tag
        wall = (
            (self.boundary_tags == self.walls_tag)
            | (self.boundary_tags == self.sphere_wall_tag)
        )
        return inlet, outlet, wall

    def make_euler_boundary_exterior(self, U):
        """Build inviscid ghost states at every physical boundary point."""
        U_inside = self.interpolate_boundary_trace(U)
        W_inside = self.conserved_to_primitive(U_inside)
        inlet, _, wall = self.boundary_masks()

        velocity_inside = W_inside[..., 1:4]
        normal_velocity = jnp.sum(
            velocity_inside * self.boundary_normals,
            axis=-1,
        )
        reflected_velocity = (
            velocity_inside
            - 2.0 * normal_velocity[..., None] * self.boundary_normals
        )
        W_wall = W_inside.at[..., 1:4].set(reflected_velocity)

        W_farfield = jnp.broadcast_to(
            self.farfield_primitive,
            W_inside.shape,
        )
        W_exterior = jnp.where(
            wall[:, None, None],
            W_wall,
            W_inside,
        )
        W_exterior = jnp.where(
            inlet[:, None, None],
            W_farfield,
            W_exterior,
        )
        return self.primitive_to_conserved(W_exterior)

    def make_viscous_boundary_data(self, U):
        """Return the wall/farfield state and BR2 ghost viscous variables."""
        U_inside = self.interpolate_boundary_trace(U)
        W_inside = self.conserved_to_primitive(U_inside)
        inlet, _, wall = self.boundary_masks()

        W_wall = W_inside.at[..., 1:4].set(0.0)
        W_farfield = jnp.broadcast_to(
            self.farfield_primitive,
            W_inside.shape,
        )
        W_face = jnp.where(
            wall[:, None, None],
            W_wall,
            W_inside,
        )
        W_face = jnp.where(
            inlet[:, None, None],
            W_farfield,
            W_face,
        )
        U_face = self.primitive_to_conserved(W_face)

        variables_inside = self.viscous_variables(U_inside)
        variables_face = self.viscous_variables(U_face)
        variables_ghost = 2.0 * variables_face - variables_inside
        return U_face, variables_ghost

    def apply_inverse_mass(self, weak_term):
        if self.geometry_order == 1:
            result = jnp.einsum(
                "ij,ej...->ei...",
                self.inverse_reference_mass,
                weak_term,
            )
            extra_dimensions = (1,) * (weak_term.ndim - 1)
            return result / jnp.abs(self.affine_determinant).reshape(
                (self.number_of_elements,) + extra_dimensions
            )
        assert self.inverse_mass_matrices is not None
        return jnp.einsum(
            "eij,ej...->ei...",
            self.inverse_mass_matrices,
            weak_term,
        )

    def apply_face_inverse_mass(self, weak_term, element_ids):
        if self.geometry_order == 1:
            result = jnp.einsum(
                "ij,fj...->fi...",
                self.inverse_reference_mass,
                weak_term,
            )
            extra_dimensions = (1,) * (weak_term.ndim - 1)
            return result / jnp.abs(
                self.affine_determinant[element_ids]
            ).reshape((element_ids.shape[0],) + extra_dimensions)
        assert self.inverse_mass_matrices is not None
        return jnp.einsum(
            "fij,fj...->fi...",
            self.inverse_mass_matrices[element_ids],
            weak_term,
        )

    def scalar_numerical_flux(self, state):
        left_state, right_state = (
            self.interpolate_internal_traces(
                state
            )
        )

        normal_velocity = jnp.sum(
            self.face_normals
            * self.advection_velocity[None, None, :],
            axis=-1,
        )

        upwind_state = jnp.where(
            normal_velocity >= 0.0,
            left_state,
            right_state,
        )

        numerical_flux = (
            normal_velocity
            * upwind_state
        )

        return numerical_flux

    def scalar_surface_term(self, state):
        numerical_flux = (
            self.scalar_numerical_flux(state)
        )

        owner_contribution = -jnp.einsum(
            "fq,fq,fqi->fi",
            numerical_flux,
            self.face_quadrature_weights,
            self.owner_trace_basis,
        )

        neighbor_contribution = jnp.einsum(
            "fq,fq,fqi->fi",
            numerical_flux,
            self.face_quadrature_weights,
            self.neighbor_trace_basis,
        )

        surface_term = jnp.zeros(
            (
                self.number_of_elements,
                self.nodes_per_element,
            ),
            dtype=state.dtype,
        )

        surface_term = surface_term.at[
            self.owner
        ].add(owner_contribution)

        surface_term = surface_term.at[
            self.neighbor
        ].add(neighbor_contribution)

        return surface_term

    def scalar_volume_term(self, state):
        # state shape: (elements, 4 nodes)

        # Interpolate nodal values to four quadrature points
        state_at_quadrature = jnp.einsum(
            "qi,ei->eq",
            self.volume_basis,
            state,
        )

        # F(u) = velocity_vector * u
        physical_flux = (
            state_at_quadrature[..., None]
            * self.advection_velocity[None, None, :]
        )

        # Integral of grad(phi_i) dot F over each element
        volume_term = jnp.einsum(
            "eqid,eqd,eq->ei",
            self.volume_physical_basis_gradients,
            physical_flux,
            self.volume_quadrature_weights,
        )

        return volume_term

    def scalar_rhs(self, state):
        volume_term = self.scalar_volume_term(state)
        surface_term = self.scalar_surface_term(state)

        weak_rhs = volume_term + surface_term

        return self.apply_inverse_mass(weak_rhs)
    def compute_scalar_dt(self):
        normal_speed = jnp.abs(
            jnp.sum(
                self.face_normals
                * self.advection_velocity[None, None, :],
                axis=-1,
            )
        )
        face_transport_rate = jnp.sum(
            normal_speed * self.face_quadrature_weights,
            axis=1,
        )
        element_transport_rate = jnp.zeros(
            self.number_of_elements,
            dtype=self.points.dtype,
        )
        # Each internal face belongs to both elements.
        element_transport_rate = element_transport_rate.at[
            self.owner
        ].add(face_transport_rate)
        element_transport_rate = element_transport_rate.at[
            self.neighbor
        ].add(face_transport_rate)
        # Approximate DG spectral rate for each element.
        spectral_rate = (
            (2 * self.polynomial_degree + 1)
            * element_transport_rate
            / self.element_volumes
        )
        maximum_rate = jnp.max(spectral_rate)
        return self.cfl / maximum_rate

    def scalar_ssp_rk3_step(self, state, dt):
        initial_state = state

        # Stage 1
        state_1 = (
            initial_state
            + dt * self.scalar_rhs(initial_state)
        )

        # Stage 2
        state_2 = (
            0.75 * initial_state
            + 0.25 * (
                state_1
                + dt * self.scalar_rhs(state_1)
            )
        )

        # Stage 3
        new_state = (
            (1.0 / 3.0) * initial_state
            + (2.0 / 3.0) * (
                state_2
                + dt * self.scalar_rhs(state_2)
            )
        )

        return new_state
    def primitive_to_conserved(self, W):
        density = W[..., 0]

        velocity_x = W[..., 1]
        velocity_y = W[..., 2]
        velocity_z = W[..., 3]

        pressure = W[..., 4]

        energy = (
            pressure / (self.gamma - 1.0)
            + 0.5
            * density
            * (
                velocity_x**2
                + velocity_y**2
                + velocity_z**2
            )
        )

        return jnp.stack(
            [
                density,
                density * velocity_x,
                density * velocity_y,
                density * velocity_z,
                energy,
            ],
            axis=-1,
        )
    def conserved_to_primitive(self, U):
        density = U[..., 0]

        momentum_x = U[..., 1]
        momentum_y = U[..., 2]
        momentum_z = U[..., 3]

        energy = U[..., 4]

        velocity_x = momentum_x / density
        velocity_y = momentum_y / density
        velocity_z = momentum_z / density

        kinetic_energy = (
            0.5
            * (
                momentum_x**2
                + momentum_y**2
                + momentum_z**2
            )
            / density
        )

        pressure = (
            self.gamma - 1.0
        ) * (
            energy - kinetic_energy
        )

        return jnp.stack(
            [
                density,
                velocity_x,
                velocity_y,
                velocity_z,
                pressure,
            ],
            axis=-1,
        )

    def euler_physical_flux(self, U):
        W = self.conserved_to_primitive(U)

        density = W[..., 0]
        velocity_x = W[..., 1]
        velocity_y = W[..., 2]
        velocity_z = W[..., 3]
        pressure = W[..., 4]

        energy = U[..., 4]

        flux_x = jnp.stack(
            [
                density * velocity_x,
                density * velocity_x**2 + pressure,
                density * velocity_x * velocity_y,
                density * velocity_x * velocity_z,
                velocity_x * (energy + pressure),
            ],
            axis=-1,
        )

        flux_y = jnp.stack(
            [
                density * velocity_y,
                density * velocity_y * velocity_x,
                density * velocity_y**2 + pressure,
                density * velocity_y * velocity_z,
                velocity_y * (energy + pressure),
            ],
            axis=-1,
        )

        flux_z = jnp.stack(
            [
                density * velocity_z,
                density * velocity_z * velocity_x,
                density * velocity_z * velocity_y,
                density * velocity_z**2 + pressure,
                velocity_z * (energy + pressure),
            ],
            axis=-1,
        )

        return jnp.stack(
            [
                flux_x,
                flux_y,
                flux_z,
            ],
            axis=-2,
        )

    def euler_volume_term(self, U):
        # U shape: (elements, 4 nodes, 5 variables)
    
        # Interpolate all five conserved variables from nodes
        # to the four volume quadrature points.
        U_at_quadrature = jnp.einsum(
            "qi,eiv->eqv",
            self.volume_basis,
            U,
        )
    
        # Shape: (elements, quadrature points, directions, variables)
        physical_flux = self.euler_physical_flux(
            U_at_quadrature
        )
    
        volume_term = jnp.einsum(
            "eqid,eqdv,eq->eiv",
            self.volume_physical_basis_gradients,
            physical_flux,
            self.volume_quadrature_weights,
        )
    
        return volume_term

    def euler_normal_flux(self, U, normals):
        if normals.ndim == 2:
            normals = normals[:, None, :]
        physical_flux = self.euler_physical_flux(U)

        normal_flux = jnp.einsum(
            "fqdv,fqd->fqv",
            physical_flux,
            normals,
        )

        return normal_flux

    def euler_rusanov_flux(
        self,
        U_left,
        U_right,
        normals,
    ):  
        if normals.ndim == 2:
            normals = normals[:, None, :]
        W_left = self.conserved_to_primitive(
            U_left
        )

        W_right = self.conserved_to_primitive(
            U_right
        )

        velocity_left = W_left[..., 1:4]
        velocity_right = W_right[..., 1:4]

        density_left = W_left[..., 0]
        density_right = W_right[..., 0]

        pressure_left = W_left[..., 4]
        pressure_right = W_right[..., 4]

        normal_velocity_left = jnp.einsum(
            "fqd,fqd->fq",
            velocity_left,
            normals,
        )

        normal_velocity_right = jnp.einsum(
            "fqd,fqd->fq",
            velocity_right,
            normals,
        )

        sound_speed_left = jnp.sqrt(
            self.gamma
            * pressure_left
            / density_left
        )

        sound_speed_right = jnp.sqrt(
            self.gamma
            * pressure_right
            / density_right
        )

        maximum_wave_speed = jnp.maximum(
            jnp.abs(normal_velocity_left)
            + sound_speed_left,
            jnp.abs(normal_velocity_right)
            + sound_speed_right,
        )

        flux_left = self.euler_normal_flux(
            U_left,
            normals,
        )

        flux_right = self.euler_normal_flux(
            U_right,
            normals,
        )

        numerical_flux = (
            0.5 * (flux_left + flux_right)
            - 0.5
            * maximum_wave_speed[..., None]
            * (U_right - U_left)
        )

        return numerical_flux

    def hllc_wave_speeds(
        self,
        U_left,
        U_right,
        normals,
    ):
        if normals.ndim == 2:
            normals = normals[:, None, :]
        W_left = self.conserved_to_primitive(U_left)
        W_right = self.conserved_to_primitive(U_right)

        density_left = W_left[..., 0]
        velocity_left = W_left[..., 1:4]
        pressure_left = W_left[..., 4]

        density_right = W_right[..., 0]
        velocity_right = W_right[..., 1:4]
        pressure_right = W_right[..., 4]

        normal_velocity_left = jnp.einsum(
            "fqd,fqd->fq",
            velocity_left,
            normals,
        )
        normal_velocity_right = jnp.einsum(
            "fqd,fqd->fq",
            velocity_right,
            normals,
        )

        sound_speed_left = jnp.sqrt(
            self.gamma * pressure_left / density_left
        )
        sound_speed_right = jnp.sqrt(
            self.gamma * pressure_right / density_right
        )

        S_left = jnp.minimum(
            normal_velocity_left - sound_speed_left,
            normal_velocity_right - sound_speed_right,
        )
        S_right = jnp.maximum(
            normal_velocity_left + sound_speed_left,
            normal_velocity_right + sound_speed_right,
        )

        numerator = (
            pressure_right
            - pressure_left
            + density_left
            * normal_velocity_left
            * (S_left - normal_velocity_left)
            - density_right
            * normal_velocity_right
            * (S_right - normal_velocity_right)
        )
        denominator = (
            density_left * (S_left - normal_velocity_left)
            - density_right * (S_right - normal_velocity_right)
        )
        safe_denominator = jnp.where(
            jnp.abs(denominator) > 1.0e-14,
            denominator,
            jnp.where(denominator >= 0.0, 1.0e-14, -1.0e-14),
        )
        S_middle = numerator / safe_denominator

        return S_left, S_middle, S_right

    def hllc_star_state(
        self,
        U,
        W,
        S,
        S_middle,
        normals,
    ):
        if normals.ndim == 2:
            normals = normals[:, None, :]
        density = W[..., 0]
        velocity = W[..., 1:4]
        pressure = W[..., 4]
        energy = U[..., 4]

        normal_velocity = jnp.einsum(
            "fqd,fqd->fq",
            velocity,
            normals,
        )
        denominator = S - S_middle
        safe_denominator = jnp.where(
            jnp.abs(denominator) > 1.0e-14,
            denominator,
            jnp.where(denominator >= 0.0, 1.0e-14, -1.0e-14),
        )

        density_star = (
            density
            * (S - normal_velocity)
            / safe_denominator
        )
        velocity_star = (
            velocity
            + (S_middle - normal_velocity)[..., None]
            * normals
        )
        momentum_star = density_star[..., None] * velocity_star

        pressure_star = (
            pressure
            + density
            * (S - normal_velocity)
            * (S_middle - normal_velocity)
        )
        energy_star = (
            (S - normal_velocity) * energy
            - pressure * normal_velocity
            + pressure_star * S_middle
        ) / safe_denominator

        return jnp.concatenate(
            [
                density_star[..., None],
                momentum_star,
                energy_star[..., None],
            ],
            axis=-1,
        )

    def euler_hllc_flux(
        self,
        U_left,
        U_right,
        normals,
    ):
        W_left = self.conserved_to_primitive(U_left)
        W_right = self.conserved_to_primitive(U_right)
        flux_left = self.euler_normal_flux(U_left, normals)
        flux_right = self.euler_normal_flux(U_right, normals)

        S_left, S_middle, S_right = self.hllc_wave_speeds(
            U_left,
            U_right,
            normals,
        )
        U_star_left = self.hllc_star_state(
            U_left,
            W_left,
            S_left,
            S_middle,
            normals,
        )
        U_star_right = self.hllc_star_state(
            U_right,
            W_right,
            S_right,
            S_middle,
            normals,
        )

        flux_star_left = (
            flux_left
            + S_left[..., None] * (U_star_left - U_left)
        )
        flux_star_right = (
            flux_right
            + S_right[..., None] * (U_star_right - U_right)
        )

        return jnp.where(
            (S_left >= 0.0)[..., None],
            flux_left,
            jnp.where(
                (S_middle >= 0.0)[..., None],
                flux_star_left,
                jnp.where(
                    (S_right > 0.0)[..., None],
                    flux_star_right,
                    flux_right,
                ),
            ),
        )

    def euler_numerical_flux(
        self,
        U_left,
        U_right,
        normals,
    ):
        if self.inviscid_flux == "hllc":
            return self.euler_hllc_flux(
                U_left,
                U_right,
                normals,
            )
        return self.euler_rusanov_flux(
            U_left,
            U_right,
            normals,
        )

    def euler_surface_term(self, U):
        U_left, U_right = (
            self.interpolate_internal_traces(U)
        )

        numerical_flux = self.euler_numerical_flux(
            U_left,
            U_right,
            self.face_normals,
        )

        owner_contribution = -jnp.einsum(
            "fqv,fq,fqi->fiv",
            numerical_flux,
            self.face_quadrature_weights,
            self.owner_trace_basis,
        )

        neighbor_contribution = jnp.einsum(
            "fqv,fq,fqi->fiv",
            numerical_flux,
            self.face_quadrature_weights,
            self.neighbor_trace_basis,
        )

        surface_term = jnp.zeros(
            (
                self.number_of_elements,
                self.nodes_per_element,
                5,
            ),
            dtype=U.dtype,
        )

        surface_term = surface_term.at[
            self.owner
        ].add(owner_contribution)

        surface_term = surface_term.at[
            self.neighbor
        ].add(neighbor_contribution)

        return surface_term

    def euler_boundary_surface_term(self, U, U_exterior):
        """Integrated boundary term using curved quadrature metrics."""
        U_inside = self.interpolate_boundary_trace(U)
        numerical_flux = self.euler_numerical_flux(
            U_inside,
            U_exterior,
            self.boundary_normals,
        )
        contribution = -jnp.einsum(
            "fqv,fq,fqi->fiv",
            numerical_flux,
            self.boundary_quadrature_weights,
            self.boundary_trace_basis,
        )
        surface_term = jnp.zeros_like(U)
        return surface_term.at[self.boundary_owner].add(contribution)

    def euler_rhs_with_boundary(self, U, U_exterior):
        weak_rhs = (
            self.euler_volume_term(U)
            + self.euler_surface_term(U)
            + self.euler_boundary_surface_term(U, U_exterior)
        )
        return self.apply_inverse_mass(weak_rhs)

    def euler_rhs(self, U):
        volume_term = self.euler_volume_term(U)
        surface_term = self.euler_surface_term(U)

        weak_rhs = volume_term + surface_term

        if self.boundary_owner.shape[0] > 0:
            U_exterior = self.make_euler_boundary_exterior(U)
            weak_rhs = (
                weak_rhs
                + self.euler_boundary_surface_term(U, U_exterior)
            )

        return self.apply_inverse_mass(weak_rhs)

    def compute_euler_dt(self, U):
        U_left, U_right = (
            self.interpolate_internal_traces(U)
        )

        W_left = self.conserved_to_primitive(
            U_left
        )

        W_right = self.conserved_to_primitive(
            U_right
        )

        density_left = W_left[..., 0]
        velocity_left = W_left[..., 1:4]
        pressure_left = W_left[..., 4]

        density_right = W_right[..., 0]
        velocity_right = W_right[..., 1:4]
        pressure_right = W_right[..., 4]

        normal_velocity_left = jnp.einsum(
            "fqd,fqd->fq",
            velocity_left,
            self.face_normals,
        )

        normal_velocity_right = jnp.einsum(
            "fqd,fqd->fq",
            velocity_right,
            self.face_normals,
        )

        sound_speed_left = jnp.sqrt(
            self.gamma
            * pressure_left
            / density_left
        )

        sound_speed_right = jnp.sqrt(
            self.gamma
            * pressure_right
            / density_right
        )

        face_wave_speed = jnp.maximum(
            jnp.abs(normal_velocity_left)
            + sound_speed_left,
            jnp.abs(normal_velocity_right)
            + sound_speed_right,
        )

        integrated_face_speed = jnp.sum(
            face_wave_speed * self.face_quadrature_weights,
            axis=1,
        )

        element_wave_rate = jnp.zeros(
            self.number_of_elements,
            dtype=U.dtype,
        )

        element_wave_rate = element_wave_rate.at[
            self.owner
        ].add(integrated_face_speed)

        element_wave_rate = element_wave_rate.at[
            self.neighbor
        ].add(integrated_face_speed)

        if self.boundary_owner.shape[0] > 0:
            U_boundary = self.interpolate_boundary_trace(U)
            W_boundary = self.conserved_to_primitive(U_boundary)
            boundary_velocity = W_boundary[..., 1:4]
            boundary_sound_speed = jnp.sqrt(
                self.gamma
                * W_boundary[..., 4]
                / W_boundary[..., 0]
            )
            boundary_normal_velocity = jnp.sum(
                boundary_velocity * self.boundary_normals,
                axis=-1,
            )
            boundary_speed = (
                jnp.abs(boundary_normal_velocity)
                + boundary_sound_speed
            )
            integrated_boundary_speed = jnp.sum(
                boundary_speed * self.boundary_quadrature_weights,
                axis=1,
            )
            element_wave_rate = element_wave_rate.at[
                self.boundary_owner
            ].add(integrated_boundary_speed)

        spectral_rate = (
            (2 * self.polynomial_degree + 1)
            * element_wave_rate
            / self.element_volumes
        )

        maximum_rate = jnp.max(spectral_rate)

        return self.cfl / maximum_rate

    def euler_ssp_rk3_step(self, U, dt):
        U_initial = U

        U_1 = self.stabilize_state(
            U_initial
            + dt * self.euler_rhs(U_initial)
        )

        U_2 = self.stabilize_state(
            0.75 * U_initial
            + 0.25 * (
                U_1
                + dt * self.euler_rhs(U_1)
            )
        )

        U_new = self.stabilize_state(
            (1.0 / 3.0) * U_initial
            + (2.0 / 3.0) * (
                U_2
                + dt * self.euler_rhs(U_2)
            )
        )

        return U_new
    def integrated_conserved_state(self, U):
        U_at_quadrature = jnp.einsum(
            "qi,eiv->eqv",
            self.volume_basis,
            U,
        )

        return jnp.sum(
            U_at_quadrature
            * self.volume_quadrature_weights[..., None],
            axis=(0, 1),
        )

    def element_average(self, state):
        """Metric-weighted element average for straight or curved cells."""
        state_at_quadrature = jnp.einsum(
            "qi,ei...->eq...",
            self.volume_basis,
            state,
        )
        integral = jnp.einsum(
            "eq,eq...->e...",
            self.volume_quadrature_weights,
            state_at_quadrature,
        )
        extra_dimensions = (1,) * (state.ndim - 2)
        return integral / self.element_volumes.reshape(
            (self.number_of_elements,) + extra_dimensions
        )

    def simulate_euler(
        self,
        U_initial,
        target_time,
        maximum_steps,
    ):
        initial_carry = (
            U_initial,
            jnp.asarray(0.0, dtype=U_initial.dtype),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def condition(carry):
            U, current_time, step = carry

            return (
                (current_time < target_time)
                & (step < maximum_steps)
            )

        def loop_body(carry):
            U, current_time, step = carry

            cfl_dt = self.compute_euler_dt(U)

            # Prevent the final step from passing target_time.
            dt = jnp.minimum(
                cfl_dt,
                target_time - current_time,
            )

            U_new = self.euler_ssp_rk3_step(
                U,
                dt,
            )

            return (
                U_new,
                current_time + dt,
                step + 1,
            )

        return jax.lax.while_loop(
            condition,
            loop_body,
            initial_carry,
        )

    def make_smooth_euler_initial_state(self):
        x = self.solution_nodes[..., 0]
        y = self.solution_nodes[..., 1]
        z = self.solution_nodes[..., 2]

        density = (
            1.0
            + 0.05 * jnp.sin(x) * jnp.cos(y)
        )

        velocity_x = (
            0.2
            + 0.02 * jnp.sin(y)
        )

        velocity_y = (
            0.1 * jnp.cos(x)
        )

        velocity_z = (
            0.05 * jnp.sin(z)
        )

        pressure = (
            1.0
            + 0.05 * jnp.cos(x) * jnp.cos(z)
        )

        W_initial = jnp.stack(
            [
                density,
                velocity_x,
                velocity_y,
                velocity_z,
                pressure,
            ],
            axis=-1,
        )

        return self.primitive_to_conserved(
            W_initial
        )

    def pressure_from_conserved(self, U):
        density = U[..., 0]
        momentum = U[..., 1:4]
        energy = U[..., 4]

        momentum_squared = jnp.sum(
            momentum**2,
            axis=-1,
        )

        kinetic_energy = (
            0.5
            * momentum_squared
            / density
        )

        return (
            self.gamma - 1.0
        ) * (
            energy - kinetic_energy
        )

    def enforce_euler_positivity(
        self,
        U,
        density_floor=1.0e-10,
        pressure_floor=1.0e-10,
    ):
        U_average = self.element_average(U)

        U_at_volume = jnp.einsum(
            "qi,eiv->eqv",
            self.volume_basis,
            U,
        )
        U_left, U_right = self.interpolate_internal_traces(U)
        U_boundary = self.interpolate_boundary_trace(U)

        # ---------- Density limiting ----------

        average_density = U_average[:, 0]
        minimum_density = jnp.minimum(
            jnp.min(U[..., 0], axis=1),
            jnp.min(U_at_volume[..., 0], axis=1),
        )
        minimum_density = minimum_density.at[self.owner].min(
            jnp.min(U_left[..., 0], axis=1)
        )
        minimum_density = minimum_density.at[self.neighbor].min(
            jnp.min(U_right[..., 0], axis=1)
        )
        minimum_density = minimum_density.at[self.boundary_owner].min(
            jnp.min(U_boundary[..., 0], axis=1)
        )

        density_denominator = jnp.maximum(
            average_density - minimum_density,
            1.0e-30,
        )

        density_theta = jnp.where(
            minimum_density < density_floor,
            (
                average_density - density_floor
            ) / density_denominator,
            1.0,
        )

        density_theta = jnp.clip(
            density_theta,
            0.0,
            1.0,
        )

        U_density_limited = (
            U_average[:, None, :]
            + density_theta[:, None, None]
            * (
                U - U_average[:, None, :]
            )
        )

        # ---------- Pressure limiting ----------

        volume_limited = jnp.einsum(
            "qi,eiv->eqv",
            self.volume_basis,
            U_density_limited,
        )
        left_limited, right_limited = self.interpolate_internal_traces(
            U_density_limited
        )
        boundary_limited = self.interpolate_boundary_trace(
            U_density_limited
        )

        def pressure_theta_for_points(states, averages):
            direction = states - averages[:, None, :]
            lower = jnp.zeros(states.shape[:-1], dtype=U.dtype)
            upper = jnp.ones(states.shape[:-1], dtype=U.dtype)

            def bisection_step(_, bounds):
                lower_value, upper_value = bounds
                middle = 0.5 * (lower_value + upper_value)
                middle_state = (
                    averages[:, None, :]
                    + middle[..., None] * direction
                )
                valid = (
                    self.pressure_from_conserved(middle_state)
                    >= pressure_floor
                )
                return (
                    jnp.where(valid, middle, lower_value),
                    jnp.where(valid, upper_value, middle),
                )

            lower, _ = jax.lax.fori_loop(
                0,
                50,
                bisection_step,
                (lower, upper),
            )
            return jnp.min(lower, axis=1)

        pressure_theta = jnp.minimum(
            pressure_theta_for_points(U_density_limited, U_average),
            pressure_theta_for_points(volume_limited, U_average),
        )
        owner_face_theta = pressure_theta_for_points(
            left_limited,
            U_average[self.owner],
        )
        neighbor_face_theta = pressure_theta_for_points(
            right_limited,
            U_average[self.neighbor],
        )
        pressure_theta = pressure_theta.at[self.owner].min(owner_face_theta)
        pressure_theta = pressure_theta.at[self.neighbor].min(
            neighbor_face_theta
        )
        boundary_face_theta = pressure_theta_for_points(
            boundary_limited,
            U_average[self.boundary_owner],
        )
        pressure_theta = pressure_theta.at[self.boundary_owner].min(
            boundary_face_theta
        )

        U_limited = (
            U_average[:, None, :]
            + pressure_theta[:, None, None]
            * (U_density_limited - U_average[:, None, :])
        )

        return U_limited

    def apply_positivity_if_needed(self, U):
        U_at_volume = jnp.einsum(
            "qi,eiv->eqv",
            self.volume_basis,
            U,
        )
        U_left, U_right = self.interpolate_internal_traces(U)
        U_boundary = self.interpolate_boundary_trace(U)

        density_valid = (
            jnp.all(U[..., 0] >= 1.0e-10)
            & jnp.all(U_at_volume[..., 0] >= 1.0e-10)
            & jnp.all(U_left[..., 0] >= 1.0e-10)
            & jnp.all(U_right[..., 0] >= 1.0e-10)
            & jnp.all(U_boundary[..., 0] >= 1.0e-10)
        )
        pressure_valid = (
            jnp.all(self.pressure_from_conserved(U) >= 1.0e-10)
            & jnp.all(self.pressure_from_conserved(U_at_volume) >= 1.0e-10)
            & jnp.all(self.pressure_from_conserved(U_left) >= 1.0e-10)
            & jnp.all(self.pressure_from_conserved(U_right) >= 1.0e-10)
            & jnp.all(
                self.pressure_from_conserved(U_boundary) >= 1.0e-10
            )
        )

        valid = (
            jnp.all(jnp.isfinite(U))
            & density_valid
            & pressure_valid
        )

        return jax.lax.cond(
            valid,
            lambda state: state,
            lambda state: self.enforce_euler_positivity(
                state
            ),
            U,
        )

    def shock_sensor(self, U):
        """Return the elementwise P2-curvature/face-jump shock indicator."""
        if self.polynomial_degree < 2:
            return jnp.zeros(
                self.number_of_elements,
                dtype=U.dtype,
            )

        W = self.conserved_to_primitive(U)
        edge_vertices = jnp.asarray(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        )
        linear_midpoint = 0.5 * (
            W[:, edge_vertices[:, 0], :]
            + W[:, edge_vertices[:, 1], :]
        )
        p2_midpoint = W[:, 4:10, :]

        density_scale = jnp.maximum(
            jnp.abs(linear_midpoint[..., 0]),
            1.0e-12,
        )
        pressure_scale = jnp.maximum(
            jnp.abs(linear_midpoint[..., 4]),
            1.0e-12,
        )
        curvature_sensor = jnp.maximum(
            jnp.max(
                jnp.abs(p2_midpoint[..., 0] - linear_midpoint[..., 0])
                / density_scale,
                axis=1,
            ),
            jnp.max(
                jnp.abs(p2_midpoint[..., 4] - linear_midpoint[..., 4])
                / pressure_scale,
                axis=1,
            ),
        )

        W_left, W_right = self.interpolate_internal_traces(W)
        density_face_scale = jnp.maximum(
            jnp.maximum(jnp.abs(W_left[..., 0]), jnp.abs(W_right[..., 0])),
            1.0e-12,
        )
        pressure_face_scale = jnp.maximum(
            jnp.maximum(jnp.abs(W_left[..., 4]), jnp.abs(W_right[..., 4])),
            1.0e-12,
        )
        face_sensor = jnp.maximum(
            jnp.max(
                jnp.abs(W_right[..., 0] - W_left[..., 0])
                / density_face_scale,
                axis=1,
            ),
            jnp.max(
                jnp.abs(W_right[..., 4] - W_left[..., 4])
                / pressure_face_scale,
                axis=1,
            ),
        )
        element_face_sensor = jnp.zeros(
            self.number_of_elements,
            dtype=U.dtype,
        )
        element_face_sensor = element_face_sensor.at[self.owner].max(
            face_sensor
        )
        element_face_sensor = element_face_sensor.at[self.neighbor].max(
            face_sensor
        )

        return jnp.maximum(curvature_sensor, element_face_sensor)

    def troubled_element_mask(self, U):
        return self.shock_sensor(U) > self.shock_sensor_threshold

    def apply_shock_limiter(self, U):
        """Conservatively damp high-order modes in troubled P2 elements.

        The previous limiter replaced every troubled polynomial with one
        constant cell average. That was robust but produced a conspicuous
        tetrahedral mosaic. This hierarchical limiter first damps the P2 edge
        curvature toward its P1 value, then gradually damps the remaining
        variation as the shock sensor grows. The metric-weighted conserved
        average is restored exactly after each operation.
        """
        if not self.use_shock_limiter:
            return U

        if self.polynomial_degree < 2:
            return U

        sensor = self.shock_sensor(U)
        threshold = jnp.asarray(
            self.shock_sensor_threshold,
            dtype=U.dtype,
        )
        troubled = sensor > threshold
        average = self.element_average(U)

        # P2 nodes 4:10 are the midpoints of edges
        # 01, 02, 03, 12, 13, 23.
        edge_vertices = jnp.asarray(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        )
        linear_midpoints = 0.5 * (
            U[:, edge_vertices[:, 0], :]
            + U[:, edge_vertices[:, 1], :]
        )

        sensor_ratio = threshold / jnp.maximum(sensor, threshold)
        quadratic_factor = jnp.where(
            troubled,
            jnp.clip(sensor_ratio, 0.0, 1.0),
            1.0,
        )
        limited_edges = (
            linear_midpoints
            + quadratic_factor[:, None, None]
            * (U[:, 4:10, :] - linear_midpoints)
        )
        curvature_limited = U.at[:, 4:10, :].set(limited_edges)

        # Changing edge modes can alter the integral on curved elements.
        # A constant conserved correction restores the original average.
        curvature_average = self.element_average(curvature_limited)
        curvature_limited = curvature_limited + (
            average - curvature_average
        )[:, None, :]

        # Retain P1 variation even in strong shocks instead of flattening the
        # whole element. The square root makes this less aggressive than the
        # quadratic damping while still approaching the mean smoothly.
        linear_factor = jnp.where(
            troubled,
            jnp.sqrt(jnp.clip(sensor_ratio, 0.0, 1.0)),
            1.0,
        )
        limited = (
            average[:, None, :]
            + linear_factor[:, None, None]
            * (curvature_limited - average[:, None, :])
        )

        # Floating-point roundoff aside, this is conservative by construction.
        limited_average = self.element_average(limited)
        limited = limited + (average - limited_average)[:, None, :]
        return jnp.where(troubled[:, None, None], limited, U)

    def stabilize_state(self, U):
        U = self.apply_shock_limiter(U)
        return self.apply_positivity_if_needed(U)

    def integrated_kinetic_energy(self, U):
        U_at_quadrature = jnp.einsum(
            "qi,eiv->eqv",
            self.volume_basis,
            U,
        )

        W_at_quadrature = (
            self.conserved_to_primitive(
                U_at_quadrature
            )
        )

        density = W_at_quadrature[..., 0]
        velocity = W_at_quadrature[..., 1:4]

        kinetic_energy_density = (
            0.5
            * density
            * jnp.sum(
                velocity**2,
                axis=-1,
            )
        )

        return jnp.sum(
            kinetic_energy_density
            * self.volume_quadrature_weights
        )

    def make_taylor_green_initial_state(
        self,
        reference_velocity=1.0,
        reference_density=1.0,
        reference_mach=0.1,
    ):
        x = self.solution_nodes[..., 0]
        y = self.solution_nodes[..., 1]
        z = self.solution_nodes[..., 2]

        velocity_x = (
            reference_velocity
            * jnp.sin(x)
            * jnp.cos(y)
            * jnp.cos(z)
        )

        velocity_y = (
            -reference_velocity
            * jnp.cos(x)
            * jnp.sin(y)
            * jnp.cos(z)
        )

        velocity_z = jnp.zeros_like(x)

        density = (
            reference_density
            * jnp.ones_like(x)
        )

        base_pressure = (
            reference_density
            * reference_velocity**2
            / (
                self.gamma
                * reference_mach**2
            )
        )

        pressure_perturbation = (
            reference_density
            * reference_velocity**2
            / 16.0
            * (
                jnp.cos(2.0 * x)
                + jnp.cos(2.0 * y)
            )
            * (
                jnp.cos(2.0 * z)
                + 2.0
            )
        )

        pressure = (
            base_pressure
            + pressure_perturbation
        )

        W_initial = jnp.stack(
            [
                density,
                velocity_x,
                velocity_y,
                velocity_z,
                pressure,
            ],
            axis=-1,
        )

        return self.primitive_to_conserved(
            W_initial
        )

    def viscous_variables(self, U):
        W = self.conserved_to_primitive(U)

        density = W[..., 0]
        velocity_x = W[..., 1]
        velocity_y = W[..., 2]
        velocity_z = W[..., 3]
        pressure = W[..., 4]

        temperature = (
            pressure
            / (
                density
                * self.gas_constant
            )
        )

        return jnp.stack(
            [
                velocity_x,
                velocity_y,
                velocity_z,
                temperature,
            ],
            axis=-1,
        )

    def raw_viscous_gradient(self, U):
        """Evaluate the uncorrected gradient at each element center."""
        variables = self.viscous_variables(U)
        return jnp.einsum(
            "eim,eid->emd",
            variables,
            self.center_physical_basis_gradients,
        )

    def raw_viscous_gradient_at_volume_points(self, U):
        variables = self.viscous_variables(U)
        return jnp.einsum(
            "eim,eqid->eqmd",
            variables,
            self.volume_physical_basis_gradients,
        )

    def raw_viscous_gradient_at_internal_faces(self, U):
        variables = self.viscous_variables(U)
        gradient_left = jnp.einsum(
            "fim,fqid->fqmd",
            variables[self.owner],
            self.owner_trace_physical_basis_gradients,
        )
        gradient_right = jnp.einsum(
            "fim,fqid->fqmd",
            variables[self.neighbor],
            self.neighbor_trace_physical_basis_gradients,
        )
        return gradient_left, gradient_right

    def br2_face_gradients(self, U):
        raw_volume_gradient = (
            self.raw_viscous_gradient_at_volume_points(U)
        )
        raw_gradient_left, raw_gradient_right = (
            self.raw_viscous_gradient_at_internal_faces(U)
        )

        variables = self.viscous_variables(U)

        variables_left, variables_right = (
            self.interpolate_internal_traces(
                variables
            )
        )

        # Shape: (faces, quadrature points, 4 variables)
        jump = (
            variables_right
            - variables_left
        )

        # The central trace gives one-half of the jump
        # to each adjacent element.
        owner_lifting_rhs = 0.5 * jnp.einsum(
            "fq,fqm,fqd,fqi->fimd",
            self.face_quadrature_weights,
            jump,
            self.face_normals,
            self.owner_trace_basis,
        )

        neighbor_lifting_rhs = 0.5 * jnp.einsum(
            "fq,fqm,fqd,fqi->fimd",
            self.face_quadrature_weights,
            jump,
            self.face_normals,
            self.neighbor_trace_basis,
        )

        # Apply each element's physical inverse mass matrix.
        owner_lifting_nodes = self.apply_face_inverse_mass(
            owner_lifting_rhs,
            self.owner,
        )

        neighbor_lifting_nodes = self.apply_face_inverse_mass(
            neighbor_lifting_rhs,
            self.neighbor,
        )

        raw_boundary_gradient = jnp.einsum(
            "fim,fqid->fqmd",
            variables[self.boundary_owner],
            self.boundary_trace_physical_basis_gradients,
        )
        variables_boundary_inside = self.interpolate_boundary_trace(
            variables
        )
        _, variables_boundary_ghost = self.make_viscous_boundary_data(U)
        boundary_jump = (
            variables_boundary_ghost - variables_boundary_inside
        )
        boundary_lifting_rhs = 0.5 * jnp.einsum(
            "fq,fqm,fqd,fqi->fimd",
            self.boundary_quadrature_weights,
            boundary_jump,
            self.boundary_normals,
            self.boundary_trace_basis,
        )
        boundary_lifting_nodes = self.apply_face_inverse_mass(
            boundary_lifting_rhs,
            self.boundary_owner,
        )

        global_lifting_nodes = jnp.zeros(
            (
                self.number_of_elements,
                self.nodes_per_element,
                4,
                3,
            ),
            dtype=U.dtype,
        )

        global_lifting_nodes = global_lifting_nodes.at[
            self.owner
        ].add(owner_lifting_nodes)

        global_lifting_nodes = global_lifting_nodes.at[
            self.neighbor
        ].add(neighbor_lifting_nodes)

        global_lifting_nodes = global_lifting_nodes.at[
            self.boundary_owner
        ].add(boundary_lifting_nodes)

        # Evaluate the summed lifting inside each element.
        volume_lifting = jnp.einsum(
            "qi,eimd->eqmd",
            self.volume_basis,
            global_lifting_nodes,
        )

        corrected_volume_gradient = (
            raw_volume_gradient
            + volume_lifting
        )

        # Evaluate each local lifting polynomial at its face.
        owner_lifting_at_face = jnp.einsum(
            "fqi,fimd->fqmd",
            self.owner_trace_basis,
            owner_lifting_nodes,
        )

        neighbor_lifting_at_face = jnp.einsum(
            "fqi,fimd->fqmd",
            self.neighbor_trace_basis,
            neighbor_lifting_nodes,
        )

        corrected_gradient_left = (
            raw_gradient_left
            + self.br2_stabilization
            * owner_lifting_at_face
        )

        corrected_gradient_right = (
            raw_gradient_right
            + self.br2_stabilization
            * neighbor_lifting_at_face
        )

        boundary_lifting_at_face = jnp.einsum(
            "fqi,fimd->fqmd",
            self.boundary_trace_basis,
            boundary_lifting_nodes,
        )
        corrected_boundary_gradient = (
            raw_boundary_gradient
            + self.br2_stabilization * boundary_lifting_at_face
        )

        return (
            corrected_volume_gradient,
            corrected_gradient_left,
            corrected_gradient_right,
            jump,
            corrected_boundary_gradient,
            boundary_jump,
        )

    def make_uniform_euler_state(
        self,
        density=1.0,
        velocity_x=0.2,
        velocity_y=0.1,
        velocity_z=-0.05,
        pressure=1.0,
    ):
        W_single = jnp.array(
            [
                density,
                velocity_x,
                velocity_y,
                velocity_z,
                pressure,
            ],
            dtype=self.points.dtype,
        )

        U_single = self.primitive_to_conserved(
            W_single
        )

        return jnp.broadcast_to(
            U_single,
            (
                self.number_of_elements,
                self.nodes_per_element,
                5,
            ),
        )

    def viscous_physical_flux(
        self,
        U,
        viscous_gradient,
        filter_width=None,
    ):
        W = self.conserved_to_primitive(U)

        velocity = W[..., 1:4]

        effective_viscosity = jnp.full(
            U.shape[:-1],
            self.dynamic_viscosity,
            dtype=U.dtype,
        )

        effective_conductivity = jnp.full(
            U.shape[:-1],
            self.thermal_conductivity,
            dtype=U.dtype,
        )

        if self.use_wale and filter_width is not None:
            turbulent_viscosity = (
                self.wale_dynamic_viscosity(
                    U,
                    viscous_gradient,
                    filter_width,
                )
            )

            effective_viscosity = (
                effective_viscosity
                + turbulent_viscosity
            )

            effective_conductivity = (
                effective_conductivity
                + turbulent_viscosity
                * self.specific_heat_cp
                / self.turbulent_prandtl
            )

        # Shape: (..., velocity component, derivative direction)
        velocity_gradient = (
            viscous_gradient[..., 0:3, :]
        )

        temperature_gradient = (
            viscous_gradient[..., 3, :]
        )

        divergence = (
            velocity_gradient[..., 0, 0]
            + velocity_gradient[..., 1, 1]
            + velocity_gradient[..., 2, 2]
        )

        identity = jnp.eye(
            3,
            dtype=U.dtype,
        )

        stress = effective_viscosity[..., None, None] * (
            velocity_gradient
            + jnp.swapaxes(
                velocity_gradient,
                -1,
                -2,
            )
            - (2.0 / 3.0)
            * divergence[..., None, None]
            * identity
        )

        # Change from [momentum component, direction]
        # to [flux direction, momentum component].
        momentum_flux = jnp.swapaxes(
            stress,
            -1,
            -2,
        )

        viscous_work = jnp.einsum(
            "...i,...id->...d",
            velocity,
            stress,
        )

        heat_conduction = (
            effective_conductivity[..., None]
            * temperature_gradient
        )

        energy_flux = (
            viscous_work
            + heat_conduction
        )

        zero_mass_flux = jnp.zeros_like(
            energy_flux
        )

        return jnp.stack(
            [
                zero_mass_flux,
                momentum_flux[..., 0],
                momentum_flux[..., 1],
                momentum_flux[..., 2],
                energy_flux,
            ],
            axis=-1,
        )

    def viscous_weak_term(self, U):
        (
            volume_gradient,
            gradient_left,
            gradient_right,
            jump,
            boundary_gradient,
            boundary_jump,
        ) = self.br2_face_gradients(U)

        # ---------- Volume flux ----------

        U_at_volume_points = jnp.einsum(
            "qi,eiv->eqv",
            self.volume_basis,
            U,
        )

        volume_filter_width = (
            self.element_filter_width[:, None]
            if self.use_wale
            else None
        )

        volume_flux = self.viscous_physical_flux(
            U_at_volume_points,
            volume_gradient,
            volume_filter_width,
        )

        # Viscous flux has the opposite weak-form sign
        # from the convective flux.
        volume_term = -jnp.einsum(
            "eqid,eqdv,eq->eiv",
            self.volume_physical_basis_gradients,
            volume_flux,
            self.volume_quadrature_weights,
        )

        # ---------- Face flux ----------

        U_left, U_right = (
            self.interpolate_internal_traces(U)
        )

        left_filter_width = (
            self.element_filter_width[
                self.owner
            ][:, None]
            if self.use_wale
            else None
        )

        right_filter_width = (
            self.element_filter_width[
                self.neighbor
            ][:, None]
            if self.use_wale
            else None
        )

        flux_left = self.viscous_physical_flux(
            U_left,
            gradient_left,
            left_filter_width,
        )

        flux_right = self.viscous_physical_flux(
            U_right,
            gradient_right,
            right_filter_width,
        )
        average_face_flux = (
            0.5 * (flux_left + flux_right)
        )

        normal_face_flux = jnp.einsum(
            "fqdv,fqd->fqv",
            average_face_flux,
            self.face_normals,
        )

        owner_contribution = jnp.einsum(
            "fqv,fq,fqi->fiv",
            normal_face_flux,
            self.face_quadrature_weights,
            self.owner_trace_basis,
        )

        neighbor_contribution = -jnp.einsum(
            "fqv,fq,fqi->fiv",
            normal_face_flux,
            self.face_quadrature_weights,
            self.neighbor_trace_basis,
        )

        surface_term = jnp.zeros_like(U)

        surface_term = surface_term.at[
            self.owner
        ].add(owner_contribution)

        surface_term = surface_term.at[
            self.neighbor
        ].add(neighbor_contribution)

        if self.boundary_owner.shape[0] > 0:
            U_boundary, _ = self.make_viscous_boundary_data(U)
            boundary_filter_width = (
                self.element_filter_width[self.boundary_owner][:, None]
                if self.use_wale
                else None
            )
            boundary_flux = self.viscous_physical_flux(
                U_boundary,
                boundary_gradient,
                boundary_filter_width,
            )
            normal_boundary_flux = jnp.einsum(
                "fqdv,fqd->fqv",
                boundary_flux,
                self.boundary_normals,
            )

            # Both solid groups are stationary adiabatic walls. Their
            # normal viscous energy flux is therefore zero.
            _, _, wall = self.boundary_masks()
            wall_energy_flux = jnp.where(
                wall[:, None],
                0.0,
                normal_boundary_flux[..., 4],
            )
            normal_boundary_flux = normal_boundary_flux.at[..., 4].set(
                wall_energy_flux
            )
            boundary_contribution = jnp.einsum(
                "fqv,fq,fqi->fiv",
                normal_boundary_flux,
                self.boundary_quadrature_weights,
                self.boundary_trace_basis,
            )
            surface_term = surface_term.at[
                self.boundary_owner
            ].add(boundary_contribution)

        return volume_term + surface_term

    def viscous_rhs(self, U):
        weak_term = self.viscous_weak_term(U)

        return self.apply_inverse_mass(weak_term)

    def navier_stokes_rhs(self, U):
        return (
            self.euler_rhs(U)
            + self.viscous_rhs(U)
        )

    def compute_navier_stokes_dt(self, U):
        # Existing convective CFL restriction
        euler_dt = self.compute_euler_dt(U)

        maximum_convective_rate = (
            self.cfl / euler_dt
        )

        # Sum face areas surrounding each element.
        element_surface_area = jnp.zeros(
            self.number_of_elements,
            dtype=U.dtype,
        )

        element_surface_area = element_surface_area.at[
            self.owner
        ].add(self.face_areas)

        element_surface_area = element_surface_area.at[
            self.neighbor
        ].add(self.face_areas)

        element_surface_area = element_surface_area.at[
            self.boundary_owner
        ].add(self.boundary_areas)

        density = U[..., 0]

        minimum_element_density = jnp.min(
            density,
            axis=1,
        )

        if self.use_wale:
            U_average = self.element_average(U)

            raw_gradient = (
                self.raw_viscous_gradient(U)
            )

            element_turbulent_viscosity = (
                self.wale_dynamic_viscosity(
                    U_average,
                    raw_gradient,
                    self.element_filter_width,
                )
            )
        else:
            element_turbulent_viscosity = (
                jnp.zeros(
                    self.number_of_elements,
                    dtype=U.dtype,
                )
            )

        effective_element_viscosity = (
            self.dynamic_viscosity
            + element_turbulent_viscosity
        )

        effective_element_conductivity = (
            self.thermal_conductivity
            + element_turbulent_viscosity
            * self.specific_heat_cp
            / self.turbulent_prandtl
        )

        momentum_diffusivity = (
            (4.0 / 3.0)
            * effective_element_viscosity
            / minimum_element_density
        )

        thermal_diffusivity = (
            effective_element_conductivity
            / (
                minimum_element_density
                * self.specific_heat_cp
            )
        )

        element_diffusivity = jnp.maximum(
            momentum_diffusivity,
            thermal_diffusivity,
        )

        inverse_length = (
            element_surface_area
            / self.element_volumes
        )

        diffusive_rate = (
            2.0
            * (2 * self.polynomial_degree + 1) ** 2
            * element_diffusivity
            * inverse_length**2
        )

        maximum_diffusive_rate = jnp.max(
            diffusive_rate
        )

        return (
            self.cfl
            / (
                maximum_convective_rate
                + maximum_diffusive_rate
            )
        )

    def navier_stokes_ssp_rk3_step(
        self,
        U,
        dt,
    ):
        U_initial = U

        U_1 = self.stabilize_state(
            U_initial
            + dt
            * self.navier_stokes_rhs(U_initial)
        )

        U_2 = self.stabilize_state(
            0.75 * U_initial
            + 0.25 * (
                U_1
                + dt
                * self.navier_stokes_rhs(U_1)
            )
        )

        U_new = self.stabilize_state(
            (1.0 / 3.0) * U_initial
            + (2.0 / 3.0) * (
                U_2
                + dt
                * self.navier_stokes_rhs(U_2)
            )
        )

        return U_new

    def simulate_navier_stokes(
        self,
        U_initial,
        target_time,
        maximum_steps,
    ):
        initial_carry = (
            U_initial,
            jnp.asarray(0.0, dtype=U_initial.dtype),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def condition(carry):
            U, current_time, step = carry

            return (
                (current_time < target_time)
                & (step < maximum_steps)
            )

        def loop_body(carry):
            U, current_time, step = carry

            cfl_dt = self.compute_navier_stokes_dt(
                U
            )

            dt = jnp.minimum(
                cfl_dt,
                target_time - current_time,
            )

            U_new = (
                self.navier_stokes_ssp_rk3_step(
                    U,
                    dt,
                )
            )

            return (
                U_new,
                current_time + dt,
                step + 1,
            )

        return jax.lax.while_loop(
            condition,
            loop_body,
            initial_carry,
        )

    def make_shear_wave_state(
        self,
        amplitude=0.01,
        wavenumber=1.0,
    ):
        coordinates = self.solution_nodes

        y = coordinates[..., 1]

        density = jnp.ones_like(y)

        velocity_x = (
            amplitude
            * jnp.sin(wavenumber * y)
        )

        velocity_y = jnp.zeros_like(y)
        velocity_z = jnp.zeros_like(y)

        pressure = jnp.ones_like(y)

        W = jnp.stack(
            [
                density,
                velocity_x,
                velocity_y,
                velocity_z,
                pressure,
            ],
            axis=-1,
        )

        return self.primitive_to_conserved(W)

    def wale_dynamic_viscosity(
        self,
        U,
        viscous_gradient,
        filter_width,
    ):
        density = U[..., 0]

        # G_ij = derivative of velocity_i with respect to x_j
        velocity_gradient = (
            viscous_gradient[..., 0:3, :]
        )

        strain = 0.5 * (
            velocity_gradient
            + jnp.swapaxes(
                velocity_gradient,
                -1,
                -2,
            )
        )

        gradient_squared = jnp.einsum(
            "...ik,...kj->...ij",
            velocity_gradient,
            velocity_gradient,
        )

        symmetric_gradient_squared = 0.5 * (
            gradient_squared
            + jnp.swapaxes(
                gradient_squared,
                -1,
                -2,
            )
        )

        trace_gradient_squared = jnp.trace(
            gradient_squared,
            axis1=-2,
            axis2=-1,
        )

        identity = jnp.eye(
            3,
            dtype=U.dtype,
        )

        traceless_squared_tensor = (
            symmetric_gradient_squared
            - (1.0 / 3.0)
            * trace_gradient_squared[..., None, None]
            * identity
        )

        strain_squared = jnp.sum(
            strain**2,
            axis=(-2, -1),
        )

        traceless_squared_norm = jnp.sum(
            traceless_squared_tensor**2,
            axis=(-2, -1),
        )

        numerator = (
            traceless_squared_norm ** 1.5
        )

        denominator = (
            strain_squared ** 2.5
            + traceless_squared_norm ** 1.25
            + 1.0e-30
        )

        kinematic_eddy_viscosity = (
            (
                self.wale_constant
                * filter_width
            ) ** 2
            * numerator
            / denominator
        )

        return (
            density
            * kinematic_eddy_viscosity
        )

    def element_wale_viscosity(self, U):
        """Return one inexpensive WALE viscosity estimate per element."""
        if not self.use_wale:
            return jnp.zeros(
                self.number_of_elements,
                dtype=U.dtype,
            )

        return self.wale_dynamic_viscosity(
            self.element_average(U),
            self.raw_viscous_gradient(U),
            self.element_filter_width,
        )

    def taylor_green_snapshot_fields(self, U):
        """Return reduced element fields instead of retaining the full DG state."""
        element_U = self.element_average(U)
        element_W = self.conserved_to_primitive(element_U)
        velocity = element_W[:, 1:4]
        speed = jnp.linalg.norm(velocity, axis=-1)
        pressure = element_W[:, 4]

        gradient = self.raw_viscous_gradient(U)
        velocity_gradient = gradient[:, 0:3, :]
        vorticity = jnp.stack(
            [
                velocity_gradient[:, 2, 1] - velocity_gradient[:, 1, 2],
                velocity_gradient[:, 0, 2] - velocity_gradient[:, 2, 0],
                velocity_gradient[:, 1, 0] - velocity_gradient[:, 0, 1],
            ],
            axis=-1,
        )
        vorticity_magnitude = jnp.linalg.norm(vorticity, axis=-1)
        wale_ratio = (
            self.element_wale_viscosity(U)
            / self.dynamic_viscosity
        )

        return jnp.stack(
            [
                speed,
                pressure,
                vorticity_magnitude,
                wale_ratio,
            ],
            axis=-1,
        )

    def advance_navier_stokes_to_time(
        self,
        U,
        current_time,
        current_step,
        requested_time,
        maximum_steps,
    ):
        """Advance one state to one output time with a reusable JIT kernel."""
        carry = (
            U,
            current_time,
            current_step,
        )

        def condition(inner_carry):
            _, time_value, step = inner_carry
            return (
                (time_value < requested_time)
                & (step < maximum_steps)
            )

        def loop_body(inner_carry):
            state, time_value, step = inner_carry
            dt = jnp.minimum(
                self.compute_navier_stokes_dt(state),
                requested_time - time_value,
            )
            state = self.navier_stokes_ssp_rk3_step(state, dt)
            return state, time_value + dt, step + 1

        return jax.lax.while_loop(
            condition,
            loop_body,
            carry,
        )


def run_taylor_green(
    mesh_path,
    target_time,
    maximum_steps,
    use_wale=True,
    inviscid_flux="hllc",
    polynomial_degree=2,
    show_plot=True,
    animation_frames=0,
    animation_interval=50,
    animation_fps=20,
    output_directory="outputs/taylor_green",
    save_outputs=True,
    resume_path=None,
):
    solver = TetrahedralDG(
        mesh_path,
        use_wale=use_wale,
        inviscid_flux=inviscid_flux,
        polynomial_degree=polynomial_degree,
    )

    reference_U = solver.make_taylor_green_initial_state()
    initial_totals = solver.integrated_conserved_state(reference_U)
    initial_kinetic_energy = solver.integrated_kinetic_energy(reference_U)
    if resume_path is not None:
        restart = np.load(resume_path)
        initial_U = jnp.asarray(restart["U"])
        starting_time = float(restart["time"])
        starting_step = int(restart["step"]) if "step" in restart else 0
        print(
            f"Resuming {resume_path} at t={starting_time:.8f}, "
            f"step={starting_step}",
            flush=True,
        )
    else:
        initial_U = reference_U
        starting_time = 0.0
        starting_step = 0

    start_time = time.perf_counter()
    snapshot_history = None
    output_times = None

    if animation_frames > 1:
        output_times = np.linspace(
            starting_time,
            target_time,
            animation_frames,
        )
        compiled_advance = jax.jit(
            solver.advance_navier_stokes_to_time,
            static_argnames=("maximum_steps",),
        )
        compiled_snapshot = jax.jit(
            solver.taylor_green_snapshot_fields,
        )

        final_U = initial_U
        final_time = jnp.asarray(starting_time, dtype=initial_U.dtype)
        completed_steps = jnp.asarray(starting_step, dtype=jnp.int32)
        snapshot_list = []
        checkpoint_directory = Path(output_directory)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)

        for output_index, requested_time in enumerate(output_times):
            if output_index > 0:
                final_U, final_time, completed_steps = compiled_advance(
                    final_U,
                    final_time,
                    completed_steps,
                    requested_time,
                    maximum_steps,
                )

            fields = compiled_snapshot(final_U)
            fields.block_until_ready()
            snapshot_list.append(np.asarray(fields))
            checkpoint_temporary = (
                checkpoint_directory / "checkpoint_latest.tmp.npz"
            )
            np.savez_compressed(
                checkpoint_temporary,
                U=np.asarray(final_U),
                time=np.asarray(final_time),
                step=np.asarray(completed_steps),
            )
            checkpoint_temporary.replace(
                checkpoint_directory / "checkpoint_latest.npz"
            )

            elapsed = time.perf_counter() - start_time
            fraction = (output_index + 1) / animation_frames
            eta = elapsed * (1.0 - fraction) / fraction
            print(
                f"progress={100.0 * fraction:6.2f}% "
                f"frame={output_index + 1}/{animation_frames} "
                f"time={float(final_time):.6f} "
                f"steps={int(completed_steps)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

        snapshot_history = np.stack(snapshot_list, axis=0)
    else:
        compiled_simulation = jax.jit(
            solver.advance_navier_stokes_to_time,
            static_argnames=("maximum_steps",),
        )
        final_U, final_time, completed_steps = compiled_simulation(
            initial_U,
            jnp.asarray(starting_time, dtype=initial_U.dtype),
            jnp.asarray(starting_step, dtype=jnp.int32),
            target_time,
            maximum_steps,
        )
    final_U.block_until_ready()
    execution_time = time.perf_counter() - start_time

    final_W = solver.conserved_to_primitive(final_U)
    final_totals = solver.integrated_conserved_state(final_U)
    final_kinetic_energy = solver.integrated_kinetic_energy(final_U)
    turbulent_viscosity = solver.element_wale_viscosity(final_U)

    print(f"Elements: {solver.number_of_elements}")
    print(f"Polynomial degree: {solver.polynomial_degree}")
    print(f"Geometry degree: {solver.geometry_order}")
    print(f"Nodes per element: {solver.nodes_per_element}")
    print(f"Completed steps: {int(completed_steps)}")
    print(f"Final time: {float(final_time):.8f}")
    print(f"Execution time: {execution_time:.3f} s")
    print(f"K/K0: {float(final_kinetic_energy / initial_kinetic_energy):.10f}")
    print("Conservation error:", final_totals - initial_totals)
    print(
        "Density range:",
        float(jnp.min(final_W[..., 0])),
        float(jnp.max(final_W[..., 0])),
    )
    print(
        "Pressure range:",
        float(jnp.min(final_W[..., 4])),
        float(jnp.max(final_W[..., 4])),
    )
    print(
        "WALE viscosity range:",
        float(jnp.min(turbulent_viscosity)),
        float(jnp.max(turbulent_viscosity)),
    )
    print("All finite:", bool(jnp.all(jnp.isfinite(final_U))))

    final_figure = None
    if show_plot or save_outputs:
        final_figure = plot_taylor_green_final(
            solver,
            final_U,
            final_time,
            final_kinetic_energy / initial_kinetic_energy,
            turbulent_viscosity,
        )

    animation = None
    if snapshot_history is not None:
        _, animation = animate_taylor_green(
            solver,
            output_times,
            snapshot_history,
            interval=animation_interval,
        )

    if save_outputs:
        assert final_figure is not None
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        final_state_path = output_path / "final_state.npz"
        np.savez_compressed(
            final_state_path,
            U=np.asarray(final_U),
            time=np.asarray(final_time),
            step=np.asarray(completed_steps),
        )
        print(f"Saved restart: {final_state_path.resolve()}")
        run_label = (
            f"{Path(mesh_path).stem}_p{solver.polynomial_degree}"
            f"_t{target_time:.6f}"
            .replace(".", "p")
        )

        final_image_path = output_path / f"{run_label}_final.png"
        final_figure.savefig(
            final_image_path,
            dpi=180,
            bbox_inches="tight",
        )
        print(f"Saved final figure: {final_image_path.resolve()}")

        if animation is not None:
            animation_path = output_path / f"{run_label}_animation.mp4"
            print(f"Saving animation: {animation_path.resolve()}", flush=True)
            try:
                animation.save(
                    animation_path,
                    writer=FFMpegWriter(
                        fps=animation_fps,
                        codec="mpeg4",
                        extra_args=["-q:v", "4", "-pix_fmt", "yuv420p"],
                    ),
                    dpi=120,
                )
                print(f"Saved animation: {animation_path.resolve()}")
            except Exception as error:
                gif_path = output_path / f"{run_label}_animation.gif"
                print(
                    f"MP4 save failed ({error}); saving GIF instead.",
                    flush=True,
                )
                animation.save(
                    gif_path,
                    writer=PillowWriter(fps=animation_fps),
                    dpi=100,
                )
                print(f"Saved animation: {gif_path.resolve()}")

    if show_plot:
        plt.show()
    else:
        plt.close("all")

    return solver, final_U


def plot_taylor_green_final(
    solver,
    final_U,
    final_time,
    kinetic_energy_ratio,
    turbulent_viscosity,
):
    """Plot one element-centered z=pi slice of the final DG state."""
    element_centers = jnp.mean(
        solver.element_vertices,
        axis=1,
    )
    element_U = solver.element_average(final_U)
    element_W = solver.conserved_to_primitive(element_U)

    velocity = element_W[:, 1:4]
    speed = jnp.linalg.norm(velocity, axis=-1)
    pressure = element_W[:, 4]

    gradient = solver.raw_viscous_gradient(final_U)
    velocity_gradient = gradient[:, 0:3, :]
    vorticity = jnp.stack(
        [
            velocity_gradient[:, 2, 1] - velocity_gradient[:, 1, 2],
            velocity_gradient[:, 0, 2] - velocity_gradient[:, 2, 0],
            velocity_gradient[:, 1, 0] - velocity_gradient[:, 0, 1],
        ],
        axis=-1,
    )
    vorticity_magnitude = jnp.linalg.norm(vorticity, axis=-1)

    characteristic_size = solver.element_volumes ** (1.0 / 3.0)
    slice_half_width = 1.5 * jnp.median(characteristic_size)
    slice_mask = jnp.abs(element_centers[:, 2] - jnp.pi) <= slice_half_width

    centers_plot = np.asarray(element_centers[slice_mask])
    speed_plot = np.asarray(speed[slice_mask])
    pressure_plot = np.asarray(pressure[slice_mask])
    vorticity_plot = np.asarray(vorticity_magnitude[slice_mask])
    wale_ratio_plot = np.asarray(
        turbulent_viscosity[slice_mask]
        / solver.dynamic_viscosity
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True,
        facecolor="#01051f",
    )

    fields = (
        (speed_plot, "Speed"),
        (pressure_plot, "Pressure"),
        (vorticity_plot, "Vorticity magnitude"),
        (wale_ratio_plot, r"WALE $\mu_t/\mu$"),
    )

    from scipy.interpolate import griddata
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 2.0*np.pi, 600),
        np.linspace(0.0, 2.0*np.pi, 600),
    )
    color_map = shock_colormap()
    for axis, (values, title) in zip(axes.flat, fields):
        raster = griddata(
            centers_plot[:, :2], values, (grid_x, grid_y), method="linear"
        )
        value_min, value_max = _robust_limits(raster)
        image = axis.imshow(
            raster,
            origin="lower",
            extent=(0.0, 2.0*np.pi, 0.0, 2.0*np.pi),
            interpolation="bilinear",
            cmap=color_map,
            vmin=value_min,
            vmax=value_max,
        )
        axis.set_facecolor("#01051f")
        axis.set_title(title, color="white")
        axis.set_xlabel("x", color="white")
        axis.set_ylabel("y", color="white")
        axis.tick_params(colors="white")
        for spine in axis.spines.values():
            spine.set_color("#4cc9f0")
        axis.set_aspect("equal")
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.ax.tick_params(colors="white")
        colorbar.outline.set_edgecolor("#4cc9f0")

    figure.suptitle(
        "DG Taylor--Green vortex "
        f"at t={float(final_time):.4f}, "
        f"K/K0={float(kinetic_energy_ratio):.6f}",
        color="white",
    )
    return figure


def animate_taylor_green(
    solver,
    output_times,
    snapshot_history,
    interval=50,
):
    """Animate reduced element fields on a fixed z=pi slice."""
    element_centers = jnp.mean(solver.element_vertices, axis=1)
    characteristic_size = solver.element_volumes ** (1.0 / 3.0)
    slice_half_width = 1.5 * jnp.median(characteristic_size)
    slice_mask = jnp.abs(element_centers[:, 2] - jnp.pi) <= slice_half_width

    centers_plot = np.asarray(element_centers[slice_mask])
    history_plot = np.asarray(snapshot_history[:, slice_mask, :])
    times_plot = np.asarray(output_times)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True,
        facecolor="#01051f",
    )
    definitions = (
        ("Speed", shock_colormap()),
        ("Pressure", shock_colormap()),
        ("Vorticity magnitude", shock_colormap()),
        (r"WALE $\mu_t/\mu$", shock_colormap()),
    )
    scatters = []

    for field_index, (axis, (title, color_map)) in enumerate(
        zip(axes.flat, definitions)
    ):
        values = history_plot[0, :, field_index]
        scatter = axis.scatter(
            centers_plot[:, 0],
            centers_plot[:, 1],
            c=values,
            s=8,
            cmap=color_map,
            linewidths=0.0,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        axis.set_facecolor("#01051f")
        axis.title.set_color("white")
        axis.xaxis.label.set_color("white")
        axis.yaxis.label.set_color("white")
        axis.tick_params(colors="white")
        figure.colorbar(scatter, ax=axis)
        scatters.append(scatter)

    def update(frame):
        for field_index, scatter in enumerate(scatters):
            values = history_plot[frame, :, field_index]
            scatter.set_array(values)
            value_min = float(np.min(values))
            value_max = float(np.max(values))
            if value_max <= value_min:
                value_max = value_min + 1.0e-14
            scatter.set_clim(value_min, value_max)

        figure.suptitle(
            f"DG Taylor--Green vortex at t={times_plot[frame]:.4f}"
        )
        return tuple(scatters)

    animation = FuncAnimation(
        figure,
        update,
        frames=history_plot.shape[0],
        interval=interval,
        blit=False,
        repeat=True,
    )
    return figure, animation


def main():
    parser = argparse.ArgumentParser(
        description="Run the periodic tetrahedral DG Taylor--Green vortex.",
    )
    parser.add_argument(
        "--mesh",
        default="mesh/taylor_green.msh",
    )
    parser.add_argument(
        "--target-time",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--maximum-steps",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--no-wale",
        action="store_true",
    )
    parser.add_argument(
        "--inviscid-flux",
        choices=("hllc", "rusanov"),
        default="hllc",
        help="Numerical flux at element faces (default: HLLC).",
    )
    parser.add_argument(
        "--polynomial-degree",
        type=int,
        choices=(1, 2),
        default=2,
        help="DG solution degree (default: 2).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the final Matplotlib figure.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Save reduced snapshots and show a final-state animation.",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--animation-interval",
        type=int,
        default=50,
        help="Milliseconds between animation frames.",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=20,
        help="Frames per second in the saved animation.",
    )
    parser.add_argument(
        "--output-directory",
        default="outputs/taylor_green",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the final PNG or animation file.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint_latest.npz or final_state.npz.",
    )
    args = parser.parse_args()

    run_taylor_green(
        mesh_path=args.mesh,
        target_time=args.target_time,
        maximum_steps=args.maximum_steps,
        use_wale=not args.no_wale,
        inviscid_flux=args.inviscid_flux,
        polynomial_degree=args.polynomial_degree,
        show_plot=not args.no_plot,
        animation_frames=(
            args.animation_frames
            if args.animate
            else 0
        ),
        animation_interval=args.animation_interval,
        animation_fps=args.animation_fps,
        output_directory=args.output_directory,
        save_outputs=not args.no_save,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
