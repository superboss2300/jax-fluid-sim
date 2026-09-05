import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import numpy as np
from mesh_loader import MeshLoader


class Solver:
    def __init__(
        self,
        mesh_path,
        case="sphere",
        periodic_pairs=None,
        reynolds=None,
    ):
        # Fluid/numerical constants
        self.cfl = 0.5
        self.gamma = 1.4

        # Load unstructured mesh
        self.mesh = MeshLoader(mesh_path)
        if periodic_pairs is not None:
            self.mesh.pair_periodic_boundaries(periodic_pairs)
        self.mesh.summary()

        geometry = self.mesh.jax_arrays()

        # Cell geometry
        self.cell_centers = geometry["cell_centers"]
        self.cell_volumes = geometry["cell_volumes"]
        self.number_of_cells = self.cell_centers.shape[0]

        # Internal-face connectivity and geometry
        self.owner = geometry["internal_owner"]
        self.neighbor = geometry["internal_neighbor"]
        self.normals = geometry["internal_face_normals"]
        self.face_areas = geometry["internal_face_areas"]
        self.face_displacements = geometry["internal_face_displacements"]
        self.owner_face_centers = geometry["internal_owner_face_centers"]
        self.neighbor_face_centers = geometry[
            "internal_neighbor_face_centers"
        ]

        # Boundary-face connectivity and geometry
        self.boundary_owner = geometry["boundary_owner"]
        self.boundary_normals = geometry["boundary_face_normals"]
        self.boundary_areas = geometry["boundary_face_areas"]
        self.boundary_tags = geometry["boundary_tags"]

        # Boundary tag numbers from Gmsh
        self.inlet_tag = self.mesh.physical_names.get("inlet", -1)
        self.outlet_tag = self.mesh.physical_names.get("outlet", -2)
        self.walls_tag = self.mesh.physical_names.get("walls", -3)
        self.sphere_wall_tag = self.mesh.physical_names.get(
            "sphere_wall",
            -4,
        )

        # Uniform freestream conditions
        density_inf = 1.0
        self.case = case

        if case == "sphere":
            pressure_inf = 1.0
            mach_inf = 0.2
            sound_speed_inf = jnp.sqrt(
                self.gamma * pressure_inf / density_inf
            )
            reference_velocity = mach_inf * sound_speed_inf
            self.W_farfield = jnp.array([
                density_inf,
                reference_velocity,
                0.0,
                0.0,
                pressure_inf,
            ])
            W_initial = jnp.broadcast_to(
                self.W_farfield,
                (self.number_of_cells, 5),
            )
            default_reynolds = 100.0
        elif case == "taylor_green":
            reference_velocity = 1.0
            mach_inf = 0.1
            sound_speed_inf = reference_velocity / mach_inf
            pressure_inf = (
                density_inf * sound_speed_inf**2 / self.gamma
            )

            x = self.cell_centers[:, 0]
            y = self.cell_centers[:, 1]
            z = self.cell_centers[:, 2]

            velocity_x = jnp.sin(x) * jnp.cos(y) * jnp.cos(z)
            velocity_y = -jnp.cos(x) * jnp.sin(y) * jnp.cos(z)
            velocity_z = jnp.zeros_like(x)

            # Cell-center sampling on an irregular mesh does not integrate the
            # analytic zero mean exactly.  Remove the tiny discrete mean so the
            # periodic calculation starts with exactly zero total momentum.
            total_volume = jnp.sum(self.cell_volumes)
            velocity_x = velocity_x - jnp.sum(
                velocity_x * self.cell_volumes
            ) / total_volume
            velocity_y = velocity_y - jnp.sum(
                velocity_y * self.cell_volumes
            ) / total_volume
            pressure = pressure_inf + (
                density_inf
                * reference_velocity**2
                / 16.0
                * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y))
                * (jnp.cos(2.0 * z) + 2.0)
            )

            self.W_farfield = jnp.array([
                density_inf,
                0.0,
                0.0,
                0.0,
                pressure_inf,
            ])
            W_initial = jnp.stack(
                [
                    jnp.full_like(x, density_inf),
                    velocity_x,
                    velocity_y,
                    velocity_z,
                    pressure,
                ],
                axis=-1,
            )
            default_reynolds = 1600.0
        else:
            raise ValueError(f"Unknown solver case: {case!r}")

        self.U_initial = self.primitive_to_conserved(
            W_initial
        )

        self.boundary_face_centers = (
            geometry["boundary_face_centers"]
        )

        self.gradient_matrix_inverse = (
            self.build_gradient_matrices()
        )

        self.gas_constant = 1.0
        self.prandtl = 0.72
        self.reynolds = (
            default_reynolds if reynolds is None else float(reynolds)
        )
        self.reference_length = 1.0

        self.dynamic_viscosity = (
            density_inf
            * reference_velocity
            * self.reference_length
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

        self.internal_face_centers = (
            geometry["internal_face_centers"]
        )

        self.wale_constant = 0.325
        self.turbulent_prandtl = 0.9

    def primitive_to_conserved(self, W):
        density = W[..., 0]

        velocity_x = W[..., 1]
        velocity_y = W[..., 2]
        velocity_z = W[..., 3]

        pressure = W[..., 4] 

        momentum_x = jnp.multiply(density, velocity_x)
        momentum_y = jnp.multiply(density, velocity_y)
        momentum_z = jnp.multiply(density, velocity_z)

        E = jnp.add(
            pressure/(self.gamma - 1.0) 
            , 0.5 
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
                momentum_x,
                momentum_y,
                momentum_z,
                E
            ],
            axis=-1
        )

    def compute_internal_residual(self, U):
        W = self.conserved_to_primitive(U)

        W_L, W_R = self.reconstruct_internal_primitives(U)

        W_L = self.enforce_positive_reconstruction(
            W_L,
            W[self.owner],
        )

        W_R = self.enforce_positive_reconstruction(
            W_R,
            W[self.neighbor],
        )

        U_L = self.primitive_to_conserved(W_L)
        U_R = self.primitive_to_conserved(W_R)

        face_flux = self.hllc_flux(
            U_L,
            U_R,
            self.normals,
        )

        integrated_flux = (
            face_flux * self.face_areas[:, None]
        )

        residual = jnp.zeros_like(U)

        residual = residual.at[
            self.owner
        ].add(integrated_flux)

        residual = residual.at[
            self.neighbor
        ].add(-integrated_flux)

        return residual
    
    def conserved_to_primitive(self, U):
        density = U[..., 0]
        momentum_x = U[..., 1]
        momentum_y = U[..., 2]
        momentum_z = U[..., 3]

        E = U[..., 4]

        velocity_x = momentum_x / density
        velocity_y = momentum_y / density
        velocity_z = momentum_z / density

        KE = (
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
        ) * (E - KE)

        return jnp.stack(
            [
                density,
                velocity_x,
                velocity_y,
                velocity_z,
                pressure
            ],
            axis=-1
        )

    def normal_physical_flux(self, U, normal):
        W = self.conserved_to_primitive(U)
        density = W[..., 0]
        velocity_x = W[..., 1]
        velocity_y = W[..., 2]
        velocity_z = W[..., 3]
        pressure = W[..., 4]
    
        energy = U[..., 4]

        normal_x = normal[..., 0]
        normal_y = normal[..., 1]
        normal_z = normal[..., 2]

        normal_velocity = (
            velocity_x * normal_x
            + velocity_y * normal_y
            + velocity_z * normal_z
        )

        mass_flux = density * normal_velocity

        momentum_x_flux = (
            density * velocity_x * normal_velocity
            + pressure * normal_x
        )

        momentum_y_flux = (
            density * velocity_y * normal_velocity
            + pressure * normal_y
        )

        momentum_z_flux = (
            density * velocity_z * normal_velocity
            + pressure * normal_z
        )

        energy_flux = normal_velocity * (
            energy + pressure
        )

        return jnp.stack(
            [
                mass_flux,
                momentum_x_flux,
                momentum_y_flux,
                momentum_z_flux,
                energy_flux,
            ],
            axis=-1,
        )
    def rusanov_flux(self, U_L, U_R, normal):
        # Physical normal flux on each side
        F_L = self.normal_physical_flux(
            U_L,
            normal,
        )

        F_R = self.normal_physical_flux(
            U_R,
            normal,
        )

        # Primitive variables on each side
        W_L = self.conserved_to_primitive(U_L)
        W_R = self.conserved_to_primitive(U_R)

        density_L = W_L[..., 0]
        velocity_x_L = W_L[..., 1]
        velocity_y_L = W_L[..., 2]
        velocity_z_L = W_L[..., 3]
        pressure_L = W_L[..., 4]

        density_R = W_R[..., 0]
        velocity_x_R = W_R[..., 1]
        velocity_y_R = W_R[..., 2]
        velocity_z_R = W_R[..., 3]
        pressure_R = W_R[..., 4]

        normal_x = normal[..., 0]
        normal_y = normal[..., 1]
        normal_z = normal[..., 2]

        normal_velocity_L = (
            velocity_x_L * normal_x
            + velocity_y_L * normal_y
            + velocity_z_L * normal_z
        )

        normal_velocity_R = (
            velocity_x_R * normal_x
            + velocity_y_R * normal_y
            + velocity_z_R * normal_z
        )

        sound_speed_L = jnp.sqrt(
            self.gamma * pressure_L / density_L
        )

        sound_speed_R = jnp.sqrt(
            self.gamma * pressure_R / density_R
        )

        alpha = jnp.maximum(
            jnp.abs(normal_velocity_L) + sound_speed_L,
            jnp.abs(normal_velocity_R) + sound_speed_R,
        )

        return (
            0.5 * (F_L + F_R)
            - 0.5
            * alpha[..., None]
            * (U_R - U_L)
        )
    def enforce_positive_reconstruction(self, W_face, W_cell):
        finite = jnp.all(
            jnp.isfinite(W_face),
            axis=-1,
        )

        valid = (
            finite
            & (W_face[..., 0] > 1e-8)
            & (W_face[..., 4] > 1e-8)
        )

        return jnp.where(
            valid[..., None],
            W_face,
            W_cell,
        )
    def compute_boundary_residual(self, U):
        W = self.conserved_to_primitive(U)

        W_inside = self.reconstruct_boundary_primitives(U)

        # Cell-centred fallback state for every boundary face
        W_owner = W[self.boundary_owner]

        W_inside = self.enforce_positive_reconstruction(
            W_inside,
            W_owner,
        )

        U_inside = self.primitive_to_conserved(W_inside)

        U_ghost = self.ghost_from_boundary_primitives(
            W_inside
        )

        face_flux = self.hllc_flux(
            U_inside,
            U_ghost,
            self.boundary_normals,
        )

        integrated_flux = (
            face_flux
            * self.boundary_areas[:, None]
        )

        residual = jnp.zeros_like(U)

        residual = residual.at[
            self.boundary_owner
        ].add(integrated_flux)

        return residual

    def wave_speed(self, U, normals):
        W = self.conserved_to_primitive(U)

        density = W[..., 0]
        velocity = W[..., 1:4]
        pressure = W[..., 4]

        normal_velocity = jnp.sum(
            velocity * normals,
            axis=-1,
        )

        sound_speed = jnp.sqrt(
            self.gamma * pressure / density
        )

        return jnp.abs(normal_velocity) + sound_speed

    def compute_dt(self, U):
        # Internal face states
        U_L = U[self.owner]
        U_R = U[self.neighbor]
    
        speed_L = self.wave_speed(
            U_L,
            self.normals,
        )
    
        speed_R = self.wave_speed(
            U_R,
            self.normals,
        )
    
        # Use the fastest side of every face
        internal_speed = jnp.maximum(
            speed_L,
            speed_R,
        )
    
        internal_contribution = (
            internal_speed * self.face_areas
        )
    
        # Start one accumulated value per cell
        spectral_sum = jnp.zeros(
            self.number_of_cells
        )
    
        # An internal face restricts both touching cells
        spectral_sum = spectral_sum.at[self.owner].add(
            internal_contribution
        )
    
        spectral_sum = spectral_sum.at[self.neighbor].add(
            internal_contribution
        )

        U_inside = U[self.boundary_owner]
        U_ghost = self.make_boundary_ghost_states(U)

        speed_inside = self.wave_speed(
            U_inside,
            self.boundary_normals,
        )

        speed_ghost = self.wave_speed(
            U_ghost,
            self.boundary_normals,
        )

        boundary_speed = jnp.maximum(
            speed_inside,
            speed_ghost,
        )

        boundary_contribution = (
            boundary_speed * self.boundary_areas
        )

        spectral_sum = spectral_sum.at[
            self.boundary_owner
        ].add(boundary_contribution)

        cell_diffusivity = (
            self.compute_cell_diffusivity(U)
        )

        owner_to_neighbor = self.face_displacements

        face_distance = jnp.linalg.norm(
            owner_to_neighbor,
            axis=1,
        )

        face_diffusivity = jnp.maximum(
            cell_diffusivity[self.owner],
            cell_diffusivity[self.neighbor],
        )

        internal_viscous_contribution = (
            2.0
            * face_diffusivity
            * self.face_areas
            / jnp.maximum(face_distance, 1e-12)
        )

        viscous_sum = jnp.zeros(
            self.number_of_cells
        )

        viscous_sum = viscous_sum.at[
            self.owner
        ].add(internal_viscous_contribution)

        viscous_sum = viscous_sum.at[
            self.neighbor
        ].add(internal_viscous_contribution)

        owner_to_boundary = (
            self.boundary_face_centers
            - self.cell_centers[self.boundary_owner]
        )

        boundary_distance = jnp.linalg.norm(
            owner_to_boundary,
            axis=1,
        )

        boundary_diffusivity = (
            cell_diffusivity[self.boundary_owner]
        )

        boundary_viscous_contribution = (
            2.0
            * boundary_diffusivity
            * self.boundary_areas
            / jnp.maximum(boundary_distance, 1e-12)
        )

        viscous_sum = viscous_sum.at[
            self.boundary_owner
        ].add(boundary_viscous_contribution)

        total_stability_sum = (
            spectral_sum
            + viscous_sum
        )

        cell_dt = (
            self.cfl
            * self.cell_volumes
            / jnp.maximum(
                total_stability_sum,
                1e-12,
            )
        )

        return jnp.min(cell_dt)

    def spatial_operator(self, U):
        residual = self.compute_residual(U)

        return (
            -residual
            / self.cell_volumes[:, None]
        )

    def update(self, U):
        dt = self.compute_dt(U)

        # Stage 1
        U_stage_1 = (
            U
            + dt * self.spatial_operator(U)
        )

        # Stage 2
        U_stage_2 = (
            0.75 * U
            + 0.25 * (
                U_stage_1
                + dt
                * self.spatial_operator(U_stage_1)
            )
        )

        # Stage 3
        U_new = (
            (1.0 / 3.0) * U
            + (2.0 / 3.0) * (
                U_stage_2
                + dt
                * self.spatial_operator(U_stage_2)
            )
        )

        return U_new, dt

    def simulate(self, U_initial, number_of_steps):
        initial_time = jnp.array(0.0)

        initial_carry = (
            U_initial,
            initial_time,
        )

        def loop_body(step, carry):
            U, current_time = carry

            U_new, dt = self.update(U)
            new_time = current_time + dt

            return U_new, new_time

        return jax.lax.fori_loop(
           0,
           number_of_steps,
           loop_body,
           initial_carry,
        )
    def build_gradient_matrices(self):
        # Vector from owner center to neighbor center
        displacement = self.face_displacements

        distance_squared = jnp.sum(
            displacement**2,
            axis=1,
        )

        weights = 1.0 / jnp.maximum(
            distance_squared,
            1e-12,
        )

        matrix_contribution = (
            weights[:, None, None]
            * displacement[:, :, None]
            * displacement[:, None, :]
        )

        matrices = jnp.zeros(
            (self.number_of_cells, 3, 3)
        )

        matrices = matrices.at[self.owner].add(
            matrix_contribution
        )

        matrices = matrices.at[self.neighbor].add(
            matrix_contribution
        )

        boundary_displacement = (
            self.boundary_face_centers
            - self.cell_centers[self.boundary_owner]
        )

        boundary_distance_squared = jnp.sum(
            boundary_displacement**2,
            axis=1,
        )

        boundary_weights = 1.0 / jnp.maximum(
            boundary_distance_squared,
            1e-12,
        )

        boundary_contribution = (
            boundary_weights[:, None, None]
            * boundary_displacement[:, :, None]
            * boundary_displacement[:, None, :]
        )

        matrices = matrices.at[
            self.boundary_owner
        ].add(boundary_contribution)

        return jnp.linalg.inv(matrices)

    def scalar_gradient(self, scalar, boundary_face_scalar,):  
        displacement = self.face_displacements

        weights = 1.0 / jnp.maximum(
            jnp.sum(displacement**2, axis=1),
            1e-12,
        )

        scalar_difference = (
            scalar[self.neighbor]
            - scalar[self.owner]
        )

        contribution = (
            weights[:, None]
            * displacement
            * scalar_difference[:, None]
        )

        right_hand_side = jnp.zeros(
            (self.number_of_cells, 3)
        )

        right_hand_side = right_hand_side.at[
            self.owner
        ].add(contribution)

        right_hand_side = right_hand_side.at[
            self.neighbor
        ].add(contribution)     
        boundary_displacement = (
            self.boundary_face_centers
            - self.cell_centers[self.boundary_owner]
        )

        boundary_weights = 1.0 / jnp.maximum(
            jnp.sum(boundary_displacement**2, axis=1),
            1e-12,
        )

        boundary_difference = (
            boundary_face_scalar
            - scalar[self.boundary_owner]
        )

        boundary_contribution = (
            boundary_weights[:, None]
            * boundary_displacement
            * boundary_difference[:, None]
        )

        right_hand_side = right_hand_side.at[
            self.boundary_owner
        ].add(boundary_contribution)

        return jnp.einsum(
            "nij,nj->ni",
            self.gradient_matrix_inverse,
            right_hand_side,
        )

    def make_boundary_face_primitives(self, U):
        U_inside = U[self.boundary_owner]
        W_inside = self.conserved_to_primitive(U_inside)

        # Outlet initially copies its owner-cell value
        W_face = W_inside

        # Fixed freestream inlet
        inlet_mask = (
            self.boundary_tags == self.inlet_tag
        )

        W_inlet = jnp.broadcast_to(
            self.W_farfield,
            W_inside.shape,
        )

        W_face = jnp.where(
            inlet_mask[:, None],
            W_inlet,
            W_face,
        )

        wall_mask = (
            (self.boundary_tags == self.walls_tag)
            | (self.boundary_tags == self.sphere_wall_tag)
        )
        W_wall = W_inside.at[:, 1:4].set(0.0)

        W_face = jnp.where(
            wall_mask[:, None],
            W_wall,
            W_face,
        )

        return W_face
    def temperature_from_primitive(self, W):
        density = W[..., 0]
        pressure = W[..., 4]

        return (
            pressure
            / (density * self.gas_constant)
        )
    def compute_primitive_gradients(self, U):
        W = self.conserved_to_primitive(U)
        W_boundary = self.make_boundary_face_primitives(U)

        velocity_x = W[:, 1]
        velocity_y = W[:, 2]
        velocity_z = W[:, 3]

        boundary_velocity_x = W_boundary[:, 1]
        boundary_velocity_y = W_boundary[:, 2]
        boundary_velocity_z = W_boundary[:, 3]

        grad_velocity_x = self.scalar_gradient(
            velocity_x,
            boundary_velocity_x,
        )

        grad_velocity_y = self.scalar_gradient(
            velocity_y,
            boundary_velocity_y,
        )

        grad_velocity_z = self.scalar_gradient(
            velocity_z,
            boundary_velocity_z,
        )
        velocity_gradient = jnp.stack(
            [
                grad_velocity_x,
                grad_velocity_y,
                grad_velocity_z,
            ],
            axis=1,
        )

        temperature = self.temperature_from_primitive(W)

        boundary_temperature = (
            self.temperature_from_primitive(
                W_boundary
            )
        )

        temperature_gradient = self.scalar_gradient(
            temperature,
            boundary_temperature,
        )

        return (
            velocity_gradient,
            temperature_gradient,
        )

    def compute_viscous_stress(
        self,
        velocity_gradient,
        viscosity,
    ):
        divergence = jnp.trace(
            velocity_gradient,
            axis1=1,
            axis2=2,
        )

        identity = jnp.eye(
            3,
            dtype=velocity_gradient.dtype,
        )

        strain_term = (
            velocity_gradient
            + jnp.swapaxes(
                velocity_gradient,
                1,
                2,
            )
            - (2.0 / 3.0)
            * divergence[:, None, None]
            * identity[None, :, :]
        )

        return (
            viscosity[:, None, None]
            * strain_term
        )

    def compute_heat_flux(
        self,
        temperature_gradient,
        conductivity,
    ):
        return (
            -conductivity[:, None]
            * temperature_gradient
        )

    def compute_cell_viscous_quantities(self, U):
        velocity_gradient, temperature_gradient = (
            self.compute_primitive_gradients(U)
        )
    
        viscosity, conductivity, _ = (
            self.compute_effective_transport(
                U,
                velocity_gradient,
            )
        )
    
        stress = self.compute_viscous_stress(
            velocity_gradient,
            viscosity,
        )
    
        heat_flux = self.compute_heat_flux(
            temperature_gradient,
            conductivity,
        )
    
        return stress, heat_flux
    def normal_viscous_flux(self, stress, heat_flux, velocity, normals):
        # Stress acting on a face with normal n
        traction = jnp.einsum(
            "fij,fj->fi",
            stress,
            normals,
        )

        # Physical heat flux through the face
        normal_heat_flux = jnp.sum(
            heat_flux * normals,
            axis=1,
        )

        viscous_energy_flux = (
            jnp.sum(
                velocity * traction,
                axis=1,
            )
            - normal_heat_flux
        )

        mass_flux = jnp.zeros_like(
            viscous_energy_flux
        )

        return jnp.concatenate(
            [
                mass_flux[:, None],
                traction,
                viscous_energy_flux[:, None],
            ],
            axis=1,
        )
    def compute_viscous_residual(self, U):
        W = self.conserved_to_primitive(U)

        (
            cell_velocity_gradient,
            cell_temperature_gradient,
        ) = self.compute_primitive_gradients(U)

        # Correct the internal-face velocity gradients
        face_velocity_gradient = (
            self.correct_internal_face_gradients(
                W[:, 1:4],
                cell_velocity_gradient,
            )
        )

        (
            cell_viscosity,
            cell_conductivity,
            cell_turbulent_viscosity,
        ) = self.compute_effective_transport(
            U,
            cell_velocity_gradient,
        )

        # Temperature is scalar, so temporarily add a variable axis
        temperature = self.temperature_from_primitive(W)

        face_temperature_gradient = (
            self.correct_internal_face_gradients(
                temperature[:, None],
                cell_temperature_gradient[:, None, :],
            )[:, 0, :]
        )

        face_viscosity = 0.5 * (
            cell_viscosity[self.owner]
            + cell_viscosity[self.neighbor]
        )

        face_conductivity = 0.5 * (
            cell_conductivity[self.owner]
            + cell_conductivity[self.neighbor]
        )

        # Compute stress and heat flux using corrected face gradients
        face_stress = self.compute_viscous_stress(
            face_velocity_gradient,
            face_viscosity,
        )

        face_heat_flux = self.compute_heat_flux(
            face_temperature_gradient,
            face_conductivity,
        )

        face_velocity = 0.5 * (
            W[self.owner, 1:4]
            + W[self.neighbor, 1:4]
        )

        internal_flux = self.normal_viscous_flux(
            face_stress,
            face_heat_flux,
            face_velocity,
            self.normals,
        )

        integrated_internal_flux = (
            internal_flux
            * self.face_areas[:, None]
        )

        residual = jnp.zeros_like(U)

        residual = residual.at[self.owner].add(
            integrated_internal_flux
        )

        residual = residual.at[self.neighbor].add(
            -integrated_internal_flux
        )

        W_boundary = self.make_boundary_face_primitives(U)

        # Correct velocity gradients using the boundary velocity
        boundary_velocity_gradient = (
            self.correct_boundary_face_gradients(
                W[:, 1:4],
                cell_velocity_gradient,
                W_boundary[:, 1:4],
            )
        )

        boundary_temperature = (
            self.temperature_from_primitive(
                W_boundary
            )
        )

        # Temperature needs a temporary one-variable axis
        boundary_temperature_gradient = (
            self.correct_boundary_face_gradients(
                temperature[:, None],
                cell_temperature_gradient[:, None, :],
                boundary_temperature[:, None],
            )[:, 0, :]
        )

        boundary_viscosity = (
            cell_viscosity[self.boundary_owner]
        )

        boundary_conductivity = (
            cell_conductivity[self.boundary_owner]
        )

        boundary_stress = self.compute_viscous_stress(
            boundary_velocity_gradient,
            boundary_viscosity,
        )

        boundary_heat_flux = self.compute_heat_flux(
            boundary_temperature_gradient,
            boundary_conductivity,
        )

        wall_mask = (
            (self.boundary_tags == self.walls_tag)
            | (self.boundary_tags == self.sphere_wall_tag)
        )

        normal_heat_flux = jnp.sum(
            boundary_heat_flux
            * self.boundary_normals,
            axis=1,
        )

        adiabatic_heat_flux = (
            boundary_heat_flux
            - normal_heat_flux[:, None]
            * self.boundary_normals
        )

        boundary_heat_flux = jnp.where(
            wall_mask[:, None],
            adiabatic_heat_flux,
            boundary_heat_flux,
        )
        boundary_velocity = W_boundary[:, 1:4]

        boundary_flux = self.normal_viscous_flux(
            boundary_stress,
            boundary_heat_flux,
            boundary_velocity,
            self.boundary_normals,
        )

        integrated_boundary_flux = (
            boundary_flux
            * self.boundary_areas[:, None]
        )

        residual = residual.at[
            self.boundary_owner
        ].add(integrated_boundary_flux)

        return residual

    def compute_inviscid_residual(self, U):
        internal_residual = (
            self.compute_internal_residual(U)
        )

        boundary_residual = (
            self.compute_boundary_residual(U)
        )

        return internal_residual + boundary_residual

    def compute_residual(self, U):
        inviscid_residual = (
            self.compute_inviscid_residual(U)
        )
    
        viscous_residual = (
            self.compute_viscous_residual(U)
        )
    
        return (
            inviscid_residual
            - viscous_residual
    )

    def compute_cell_diffusivity(self, U):
        W = self.conserved_to_primitive(U)
        density = W[:, 0]

        velocity_gradient, _ = (
            self.compute_primitive_gradients(U)
        )

        (
            effective_viscosity,
            effective_conductivity,
            _,
        ) = self.compute_effective_transport(
            U,
            velocity_gradient,
        )

        effective_kinematic_viscosity = (
            effective_viscosity / density
        )

        effective_thermal_diffusivity = (
            effective_conductivity
            / (
                density
                * self.specific_heat_cp
            )
        )

        return jnp.maximum(
            (4.0 / 3.0)
            * effective_kinematic_viscosity,
            effective_thermal_diffusivity,
        )

    def compute_all_primitive_gradients(self, U):
        W = self.conserved_to_primitive(U)

        W_boundary = (
            self.make_boundary_face_primitives(U)
        )

        gradients = jax.vmap(
            self.scalar_gradient,
            in_axes=(1, 1),
            out_axes=1,
        )(
            W,
            W_boundary,
        )
        return gradients

    def reconstruct_internal_primitives(self, U):
        W = self.conserved_to_primitive(U)

        gradients = (
            self.compute_all_primitive_gradients(U)
        )

        limiters = self.compute_primitive_limiters(
            U,
            gradients,
        )

        limited_gradients = (
            limiters[:, :, None]
            * gradients
        )

        owner_offset = (
            self.owner_face_centers
            - self.cell_centers[self.owner]
        )

        neighbor_offset = (
            self.neighbor_face_centers
            - self.cell_centers[self.neighbor]
        )

        owner_change = jnp.einsum(
            "fvd,fd->fv",
            limited_gradients[self.owner],
            owner_offset,
        )

        neighbor_change = jnp.einsum(
            "fvd,fd->fv",
            limited_gradients[self.neighbor],
            neighbor_offset,
        )

        W_L = W[self.owner] + owner_change
        W_R = W[self.neighbor] + neighbor_change

        return W_L, W_R

    def compute_primitive_bounds(
        self,
        W,
        W_boundary,
):  
        W_min = W
        W_max = W

        # Each owner considers its neighbor's values
        W_min = W_min.at[self.owner].min(
            W[self.neighbor]
        )

        W_max = W_max.at[self.owner].max(
            W[self.neighbor]
        )

        # Each neighbor considers its owner's values
        W_min = W_min.at[self.neighbor].min(
            W[self.owner]
        )

        W_max = W_max.at[self.neighbor].max(
            W[self.owner]
        )

        # Boundary cells also consider boundary-face values
        W_min = W_min.at[self.boundary_owner].min(
            W_boundary
        )

        W_max = W_max.at[self.boundary_owner].max(
            W_boundary
        )

        return W_min, W_max

    def venkatakrishnan_function(
        self,
        allowed_change,
        proposed_change,
):  
        small = 1e-12

        ratio = jnp.where(
            jnp.abs(proposed_change) > small,
            allowed_change / proposed_change,
            1e6,
        )

        ratio = jnp.maximum(ratio, 0.0)

        limiter = (
            ratio**2 + 2.0 * ratio
        ) / (
            ratio**2 + ratio + 2.0
        )

        return jnp.clip(
            limiter,
            0.0,
            1.0,
        )

    def compute_primitive_limiters(
        self,
        U,
        gradients,
    ):
        W = self.conserved_to_primitive(U)

        W_boundary = (
            self.make_boundary_face_primitives(U)
        )

        W_min, W_max = (
            self.compute_primitive_bounds(
                W,
                W_boundary,
            )
        )

        owner_offset = (
            self.owner_face_centers
            - self.cell_centers[self.owner]
        )

        neighbor_offset = (
            self.neighbor_face_centers
            - self.cell_centers[self.neighbor]
        )

        owner_change = jnp.einsum(
            "fvd,fd->fv",
            gradients[self.owner],
            owner_offset,
        )

        neighbor_change = jnp.einsum(
            "fvd,fd->fv",
            gradients[self.neighbor],
            neighbor_offset,
        )
        owner_allowed = jnp.where(
        owner_change > 0.0,
        W_max[self.owner] - W[self.owner],
        W_min[self.owner] - W[self.owner],
    )

        neighbor_allowed = jnp.where(
            neighbor_change > 0.0,
            W_max[self.neighbor] - W[self.neighbor],
            W_min[self.neighbor] - W[self.neighbor],
        )

        owner_face_limiter = (
            self.venkatakrishnan_function(
                owner_allowed,
                owner_change,
            )
        )

        neighbor_face_limiter = (
            self.venkatakrishnan_function(
                neighbor_allowed,
                neighbor_change,
            )
        )

        limiters = jnp.ones_like(W)

        limiters = limiters.at[self.owner].min(
            owner_face_limiter
        )

        limiters = limiters.at[self.neighbor].min(
            neighbor_face_limiter
        )

        boundary_offset = (
            self.boundary_face_centers
            - self.cell_centers[self.boundary_owner]
        )

        boundary_change = jnp.einsum(
            "fvd,fd->fv",
            gradients[self.boundary_owner],
            boundary_offset,
        )

        boundary_allowed = jnp.where(
            boundary_change > 0.0,
            W_max[self.boundary_owner]
            - W[self.boundary_owner],
            W_min[self.boundary_owner]
            - W[self.boundary_owner],
        )

        boundary_face_limiter = (
            self.venkatakrishnan_function(
                boundary_allowed,
                boundary_change,
            )
        )

        limiters = limiters.at[
            self.boundary_owner
        ].min(boundary_face_limiter)

        return limiters

    def reconstruct_boundary_primitives(self, U):
        W = self.conserved_to_primitive(U)

        gradients = (
            self.compute_all_primitive_gradients(U)
        )

        limiters = self.compute_primitive_limiters(
            U,
            gradients,
        )

        limited_gradients = (
            limiters[:, :, None]
            * gradients
        )

        boundary_offset = (
            self.boundary_face_centers
            - self.cell_centers[self.boundary_owner]
        )

        boundary_change = jnp.einsum(
            "fvd,fd->fv",
            limited_gradients[self.boundary_owner],
            boundary_offset,
        )

        W_inside = (
            W[self.boundary_owner]
            + boundary_change
        )

        return W_inside

    def ghost_from_boundary_primitives(
        self,
        W_inside,
    ):
        density = W_inside[:, 0]
        velocity = W_inside[:, 1:4]
        pressure = W_inside[:, 4]

        normal_velocity = jnp.sum(
            velocity * self.boundary_normals,
            axis=1,
        )

        reflected_velocity = (
            velocity
            - 2.0
            * normal_velocity[:, None]
            * self.boundary_normals
        )

        W_wall = jnp.concatenate(
            [
                density[:, None],
                reflected_velocity,
                pressure[:, None],
            ],
            axis=1,
        )

        W_ghost = W_inside

        wall_mask = (
            (self.boundary_tags == self.walls_tag)
            | (
                self.boundary_tags
                == self.sphere_wall_tag
            )
        )

        W_ghost = jnp.where(
            wall_mask[:, None],
            W_wall,
            W_ghost,
        )

        inlet_mask = (
            self.boundary_tags == self.inlet_tag
        )

        W_inlet = jnp.broadcast_to(
            self.W_farfield,
            W_inside.shape,
        )

        W_ghost = jnp.where(
            inlet_mask[:, None],
            W_inlet,
            W_ghost,
        )

        return self.primitive_to_conserved(
            W_ghost
        )

    def make_boundary_ghost_states(self, U):
        W_inside = self.conserved_to_primitive(
            U[self.boundary_owner]
        )
        return self.ghost_from_boundary_primitives(
            W_inside
        )
    def hllc_wave_speeds(self, U_L, U_R, normals):
        W_L = self.conserved_to_primitive(U_L)
        W_R = self.conserved_to_primitive(U_R)

        rho_L = W_L[..., 0]
        velocity_L = W_L[..., 1:4]
        pressure_L = W_L[..., 4]

        rho_R = W_R[..., 0]
        velocity_R = W_R[..., 1:4]
        pressure_R = W_R[..., 4]

        # Velocity perpendicular to each face
        normal_velocity_L = jnp.sum(
            velocity_L * normals,
            axis=-1,
        )

        normal_velocity_R = jnp.sum(
            velocity_R * normals,
            axis=-1,
        )

        sound_speed_L = jnp.sqrt(
            self.gamma * pressure_L / rho_L
        )

        sound_speed_R = jnp.sqrt(
            self.gamma * pressure_R / rho_R
        )

        # Estimates of the two outer acoustic-wave speeds
        S_L = jnp.minimum(
            normal_velocity_L - sound_speed_L,
            normal_velocity_R - sound_speed_R,
        )

        S_R = jnp.maximum(
            normal_velocity_L + sound_speed_L,
            normal_velocity_R + sound_speed_R,
        )

        numerator = (
            pressure_R
            - pressure_L
            + rho_L
            * normal_velocity_L
            * (S_L - normal_velocity_L)
            - rho_R
            * normal_velocity_R
            * (S_R - normal_velocity_R)
        )

        denominator = (
            rho_L * (S_L - normal_velocity_L)
            - rho_R * (S_R - normal_velocity_R)
        )

        # Protect against an extremely small denominator
        safe_denominator = jnp.where(
            jnp.abs(denominator) > 1e-12,
            denominator,
            jnp.where(
                denominator >= 0.0,
                1e-12,
                -1e-12,
            ),
        )

        S_M = numerator / safe_denominator

        return S_L, S_M, S_R

    def hllc_star_state(
        self,
        U,
        W,
        S,
        S_M,
        normals,
    ):
        density = W[..., 0]
        velocity = W[..., 1:4]
        pressure = W[..., 4]

        energy = U[..., 4]

        normal_velocity = jnp.sum(
            velocity * normals,
            axis=-1,
        )

        denominator = S - S_M

        safe_denominator = jnp.where(
            jnp.abs(denominator) > 1e-12,
            denominator,
            jnp.where(
                denominator >= 0.0,
                1e-12,
                -1e-12,
            ),
        )

        density_star = (
            density
            * (S - normal_velocity)
            / safe_denominator
        )

        # Only the normal part of velocity changes.
        # Tangential velocity is preserved.
        velocity_star = (
            velocity
            + (S_M - normal_velocity)[..., None]
            * normals
        )

        momentum_star = (
            density_star[..., None]
            * velocity_star
        )

        pressure_star = (
            pressure
            + density
            * (S - normal_velocity)
            * (S_M - normal_velocity)
        )

        energy_star = (
            (S - normal_velocity) * energy
            - pressure * normal_velocity
            + pressure_star * S_M
        ) / safe_denominator

        return jnp.concatenate(
            [
                density_star[..., None],
                momentum_star,
                energy_star[..., None],
            ],
            axis=-1,
        )
    
    def hllc_flux(self, U_L, U_R, normals):
        W_L = self.conserved_to_primitive(U_L)
        W_R = self.conserved_to_primitive(U_R)

        F_L = self.normal_physical_flux(
            U_L,
            normals,
        )

        F_R = self.normal_physical_flux(
            U_R,
            normals,
        )

        S_L, S_M, S_R = self.hllc_wave_speeds(
            U_L,
            U_R,
            normals,
        )

        U_star_L = self.hllc_star_state(
            U_L,
            W_L,
            S_L,
            S_M,
            normals,
        )

        U_star_R = self.hllc_star_state(
            U_R,
            W_R,
            S_R,
            S_M,
            normals,
        )

        # Fluxes inside the two star regions
        F_star_L = (
            F_L
            + S_L[..., None]
            * (U_star_L - U_L)
        )

        F_star_R = (
            F_R
            + S_R[..., None]
            * (U_star_R - U_R)
        )

        # Select the state containing speed zero
        flux = jnp.where(
            (S_L >= 0.0)[..., None],
            F_L,
            jnp.where(
                (S_M >= 0.0)[..., None],
                F_star_L,
                jnp.where(
                    (S_R > 0.0)[..., None],
                    F_star_R,
                    F_R,
                ),
            ),
        )

        return flux
    def correct_internal_face_gradients(
        self,
        cell_values,
        cell_gradients,
    ):
        # Vector from owner centre to neighbor centre
        displacement = self.face_displacements

        # Initial face gradient: average both cell gradients
        face_gradients = 0.5 * (
            cell_gradients[self.owner]
            + cell_gradients[self.neighbor]
        )

        # Actual difference measured between the cells
        actual_change = (
            cell_values[self.neighbor]
            - cell_values[self.owner]
        )

        # Change predicted by the averaged gradient
        predicted_change = jnp.einsum(
            "fvd,fd->fv",
            face_gradients,
            displacement,
        )

        normal_distance = jnp.sum(
            displacement * self.normals,
            axis=-1,
        )

        safe_normal_distance = jnp.where(
            jnp.abs(normal_distance) > 1e-12,
            normal_distance,
            jnp.where(
                normal_distance >= 0.0,
                1e-12,
                -1e-12,
            ),
        )

        correction_size = (
            actual_change - predicted_change
        ) / safe_normal_distance[:, None]

        corrected_gradients = (
            face_gradients
            + correction_size[..., None]
            * self.normals[:, None, :]
        )

        return corrected_gradients

    def correct_boundary_face_gradients(
        self,
        cell_values,
        cell_gradients,
        boundary_values,
    ):
        displacement = (
            self.boundary_face_centers
            - self.cell_centers[self.boundary_owner]
        )
    
        # Begin with the owner-cell gradient
        face_gradients = (
            cell_gradients[self.boundary_owner]
        )
    
        # Change imposed by the boundary condition
        actual_change = (
            boundary_values
            - cell_values[self.boundary_owner]
        )
    
        predicted_change = jnp.einsum(
            "fvd,fd->fv",
            face_gradients,
            displacement,
        )
    
        normal_distance = jnp.sum(
            displacement * self.boundary_normals,
            axis=-1,
        )
    
        safe_normal_distance = jnp.where(
            jnp.abs(normal_distance) > 1e-12,
            normal_distance,
            jnp.where(
                normal_distance >= 0.0,
                1e-12,
                -1e-12,
            ),
        )
    
        correction_size = (
            actual_change - predicted_change
        ) / safe_normal_distance[:, None]
    
        corrected_gradients = (
            face_gradients
            + correction_size[..., None]
            * self.boundary_normals[:, None, :]
        )
    
        return corrected_gradients

    def compute_wale_viscosity(self, U, velocity_gradient,):  
        W = self.conserved_to_primitive(U)
        density = W[:, 0]
    
        # g_ij = du_i / dx_j
        G = velocity_gradient
    
        # Resolved strain-rate tensor
        S = 0.5 * (
            G + jnp.swapaxes(G, 1, 2)
        )
    
        # Matrix product G_ik G_kj
        G_squared = jnp.einsum(
            "nik,nkj->nij",
            G,
            G,
        )
    
        # Symmetric part of G squared
        symmetric_G_squared = 0.5 * (
            G_squared
            + jnp.swapaxes(G_squared, 1, 2)
        )
    
        trace_G_squared = jnp.trace(
            G_squared,
            axis1=1,
            axis2=2,
        )
    
        identity = jnp.eye(
            3,
            dtype=U.dtype,
        )
    
        # Traceless symmetric square of the velocity gradient
        S_d = (
            symmetric_G_squared
            - (1.0 / 3.0)
            * trace_G_squared[:, None, None]
            * identity[None, :, :]
        )
    
        S_contraction = jnp.sum(
            S * S,
            axis=(1, 2),
        )
    
        S_d_contraction = jnp.sum(
            S_d * S_d,
            axis=(1, 2),
        )
    
        numerator = (
            S_d_contraction ** 1.5
        )
    
        denominator = (
            S_contraction ** 2.5
            + S_d_contraction ** 1.25
            + 1e-30
        )
    
        # Unstructured 3D filter width
        filter_width = (
            self.cell_volumes ** (1.0 / 3.0)
        )
    
        kinematic_eddy_viscosity = (
            (
                self.wale_constant
                * filter_width
            )**2
            * numerator
            / denominator
        )
    
        # Convert kinematic viscosity to dynamic viscosity
        turbulent_viscosity = (
            density * kinematic_eddy_viscosity
        )
    
        return turbulent_viscosity

    def compute_effective_transport(
        self,
        U,
        velocity_gradient,
    ):
        turbulent_viscosity = (
            self.compute_wale_viscosity(
                U,
                velocity_gradient,
            )
        )

        effective_viscosity = (
            self.dynamic_viscosity
            + turbulent_viscosity
        )

        turbulent_conductivity = (
            turbulent_viscosity
            * self.specific_heat_cp
            / self.turbulent_prandtl
        )

        effective_conductivity = (
            self.thermal_conductivity
            + turbulent_conductivity
        )

        return (
            effective_viscosity,
            effective_conductivity,
            turbulent_viscosity,
        )


def run_sphere_smoke_test():
    solver = Solver("mesh/sphere_tunnel.msh")
    compiled_simulate = jax.jit(
        lambda U: solver.simulate(U, 10)
    )
    final_U, final_time = compiled_simulate(solver.U_initial)
    final_U.block_until_ready()

    final_W = solver.conserved_to_primitive(final_U)
    velocity_gradient, _ = solver.compute_primitive_gradients(final_U)
    mu_t = solver.compute_wale_viscosity(final_U, velocity_gradient)

    print("Final time:", final_time)
    print("All finite:", jnp.all(jnp.isfinite(final_U)))
    print(
        "Density range:",
        jnp.min(final_W[:, 0]),
        jnp.max(final_W[:, 0]),
    )
    print(
        "Pressure range:",
        jnp.min(final_W[:, 4]),
        jnp.max(final_W[:, 4]),
    )
    print("WALE viscosity range:", jnp.min(mu_t), jnp.max(mu_t))
    print(
        "Maximum SGS/molecular ratio:",
        jnp.max(mu_t / solver.dynamic_viscosity),
    )


if __name__ == "__main__":
    run_sphere_smoke_test()
