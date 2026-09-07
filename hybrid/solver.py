"""Shared compressible state and face fluxes for hybrid element blocks."""

from collections.abc import Callable

import jax.numpy as jnp

from .elements import make_reference_element
from .mesh import HybridMesh
from .state import CompressibleStateLayout


class HybridCompressibleSolver:
    """Physics shared by tetrahedron, prism, and hexahedron DG operators.

    Spatial operators remain type-specific, but every block uses the same
    conservative variables and numerical flux. Optional chemistry receives a
    source callback rather than being embedded in the face-flux implementation.
    """

    def __init__(
        self,
        mesh_path,
        species_names=(),
        gamma=1.4,
        gas_constant=1.0,
        polynomial_degree=2,
        inviscid_flux="hllc",
        dynamic_viscosity=0.0,
        prandtl=0.72,
        use_wale=False,
        wale_constant=0.5,
        turbulent_prandtl=0.9,
        chemistry_source: Callable | None = None,
    ):
        self.mesh = HybridMesh(mesh_path)
        self.layout = CompressibleStateLayout(tuple(species_names))
        self.gamma = float(gamma)
        self.gas_constant = float(gas_constant)
        self.polynomial_degree = int(polynomial_degree)
        if inviscid_flux not in {"hllc", "rusanov"}:
            raise ValueError("inviscid_flux must be 'hllc' or 'rusanov'")
        self.inviscid_flux = inviscid_flux
        self.dynamic_viscosity = float(dynamic_viscosity)
        self.prandtl = float(prandtl)
        self.use_wale = bool(use_wale)
        self.wale_constant = float(wale_constant)
        self.turbulent_prandtl = float(turbulent_prandtl)
        self.specific_heat_cp = (
            self.gamma * self.gas_constant / (self.gamma - 1.0)
        )
        self.thermal_conductivity = (
            self.dynamic_viscosity * self.specific_heat_cp / self.prandtl
        )
        self.reference_elements = {
            block.element_type: make_reference_element(
                block.element_type,
                self.polynomial_degree,
            )
            for block in self.mesh.element_blocks
        }
        self.chemistry_source_function = chemistry_source

    def primitive_to_conserved(self, W):
        """Convert [rho,u,v,w,p,Y...] to [rho,rho*u,rho*E,rho*Y...]."""
        density = W[..., 0]
        velocity = W[..., 1:4]
        pressure = W[..., 4]
        momentum = density[..., None] * velocity
        total_energy = (
            pressure / (self.gamma - 1.0)
            + 0.5 * density * jnp.sum(velocity**2, axis=-1)
        )
        pieces = [
            density[..., None],
            momentum,
            total_energy[..., None],
        ]
        if self.layout.species_names:
            mass_fractions = W[..., self.layout.species]
            pieces.append(density[..., None] * mass_fractions)
        return jnp.concatenate(pieces, axis=-1)

    def conserved_to_primitive(self, U):
        density = U[..., 0]
        momentum = U[..., 1:4]
        total_energy = U[..., 4]
        safe_density = jnp.maximum(density, 1.0e-14)
        velocity = momentum / safe_density[..., None]
        kinetic_energy = 0.5 * jnp.sum(momentum**2, axis=-1) / safe_density
        pressure = (self.gamma - 1.0) * (total_energy - kinetic_energy)
        pieces = [
            density[..., None],
            velocity,
            pressure[..., None],
        ]
        if self.layout.species_names:
            pieces.append(U[..., self.layout.species] / safe_density[..., None])
        return jnp.concatenate(pieces, axis=-1)

    def normal_physical_flux(self, U, normal):
        """Euler flux dotted with an arbitrary unit face normal."""
        W = self.conserved_to_primitive(U)
        density = W[..., 0]
        velocity = W[..., 1:4]
        pressure = W[..., 4]
        normal_velocity = jnp.sum(velocity * normal, axis=-1)
        mass_flux = density * normal_velocity
        momentum_flux = (
            density[..., None] * velocity * normal_velocity[..., None]
            + pressure[..., None] * normal
        )
        energy_flux = normal_velocity * (U[..., 4] + pressure)
        pieces = [
            mass_flux[..., None],
            momentum_flux,
            energy_flux[..., None],
        ]
        if self.layout.species_names:
            pieces.append(
                U[..., self.layout.species] * normal_velocity[..., None]
            )
        return jnp.concatenate(pieces, axis=-1)

    def physical_flux_tensor(self, U):
        """Return Cartesian Euler fluxes with shape (..., 3, variables)."""
        axes = jnp.eye(3, dtype=U.dtype)
        return jnp.stack(
            [self.normal_physical_flux(U, axes[axis]) for axis in range(3)],
            axis=-2,
        )

    def rusanov_flux(self, U_left, U_right, normal):
        W_left = self.conserved_to_primitive(U_left)
        W_right = self.conserved_to_primitive(U_right)
        velocity_left = W_left[..., 1:4]
        velocity_right = W_right[..., 1:4]
        normal_velocity_left = jnp.sum(velocity_left * normal, axis=-1)
        normal_velocity_right = jnp.sum(velocity_right * normal, axis=-1)
        sound_left = jnp.sqrt(
            self.gamma * W_left[..., 4] / W_left[..., 0]
        )
        sound_right = jnp.sqrt(
            self.gamma * W_right[..., 4] / W_right[..., 0]
        )
        maximum_speed = jnp.maximum(
            jnp.abs(normal_velocity_left) + sound_left,
            jnp.abs(normal_velocity_right) + sound_right,
        )
        return 0.5 * (
            self.normal_physical_flux(U_left, normal)
            + self.normal_physical_flux(U_right, normal)
        ) - 0.5 * maximum_speed[..., None] * (U_right - U_left)

    @staticmethod
    def _safe_signed(value, tolerance=1.0e-14):
        """Keep a denominator away from zero without changing its sign."""
        return jnp.where(
            jnp.abs(value) > tolerance,
            value,
            jnp.where(value >= 0.0, tolerance, -tolerance),
        )

    def hllc_wave_speeds(self, U_left, U_right, normal):
        """Estimate the left, contact, and right Euler wave speeds."""
        W_left = self.conserved_to_primitive(U_left)
        W_right = self.conserved_to_primitive(U_right)

        rho_left = W_left[..., 0]
        rho_right = W_right[..., 0]
        velocity_left = W_left[..., 1:4]
        velocity_right = W_right[..., 1:4]
        pressure_left = W_left[..., 4]
        pressure_right = W_right[..., 4]
        normal_velocity_left = jnp.sum(velocity_left * normal, axis=-1)
        normal_velocity_right = jnp.sum(velocity_right * normal, axis=-1)
        sound_left = jnp.sqrt(
            jnp.maximum(self.gamma * pressure_left / rho_left, 0.0)
        )
        sound_right = jnp.sqrt(
            jnp.maximum(self.gamma * pressure_right / rho_right, 0.0)
        )

        speed_left = jnp.minimum(
            normal_velocity_left - sound_left,
            normal_velocity_right - sound_right,
        )
        speed_right = jnp.maximum(
            normal_velocity_left + sound_left,
            normal_velocity_right + sound_right,
        )
        denominator = (
            rho_left * (speed_left - normal_velocity_left)
            - rho_right * (speed_right - normal_velocity_right)
        )
        speed_middle = (
            pressure_right
            - pressure_left
            + rho_left
            * normal_velocity_left
            * (speed_left - normal_velocity_left)
            - rho_right
            * normal_velocity_right
            * (speed_right - normal_velocity_right)
        ) / self._safe_signed(denominator)
        return speed_left, speed_middle, speed_right

    def hllc_star_state(self, U, W, speed, speed_middle, normal):
        """Conservative state between an acoustic wave and the contact wave."""
        density = W[..., 0]
        velocity = W[..., 1:4]
        pressure = W[..., 4]
        normal_velocity = jnp.sum(velocity * normal, axis=-1)
        denominator = self._safe_signed(speed - speed_middle)
        compression = (speed - normal_velocity) / denominator
        density_star = density * compression
        velocity_star = (
            velocity
            + (speed_middle - normal_velocity)[..., None] * normal
        )
        momentum_star = density_star[..., None] * velocity_star
        pressure_star = (
            pressure
            + density
            * (speed - normal_velocity)
            * (speed_middle - normal_velocity)
        )
        energy_star = (
            (speed - normal_velocity) * U[..., 4]
            - pressure * normal_velocity
            + pressure_star * speed_middle
        ) / denominator

        pieces = [
            density_star[..., None],
            momentum_star,
            energy_star[..., None],
        ]
        if self.layout.species_names:
            # The contact wave carries the mass fractions of its own side.
            pieces.append(U[..., self.layout.species] * compression[..., None])
        return jnp.concatenate(pieces, axis=-1)

    def hllc_flux(self, U_left, U_right, normal):
        """HLLC flux with a pointwise Rusanov fallback for invalid states."""
        W_left = self.conserved_to_primitive(U_left)
        W_right = self.conserved_to_primitive(U_right)
        flux_left = self.normal_physical_flux(U_left, normal)
        flux_right = self.normal_physical_flux(U_right, normal)
        speed_left, speed_middle, speed_right = self.hllc_wave_speeds(
            U_left, U_right, normal
        )
        star_left = self.hllc_star_state(
            U_left, W_left, speed_left, speed_middle, normal
        )
        star_right = self.hllc_star_state(
            U_right, W_right, speed_right, speed_middle, normal
        )
        flux_star_left = (
            flux_left + speed_left[..., None] * (star_left - U_left)
        )
        flux_star_right = (
            flux_right + speed_right[..., None] * (star_right - U_right)
        )
        hllc = jnp.where(
            (speed_left >= 0.0)[..., None],
            flux_left,
            jnp.where(
                (speed_middle >= 0.0)[..., None],
                flux_star_left,
                jnp.where(
                    (speed_right > 0.0)[..., None],
                    flux_star_right,
                    flux_right,
                ),
            ),
        )

        valid_inputs = (
            (W_left[..., 0] > 0.0)
            & (W_right[..., 0] > 0.0)
            & (W_left[..., 4] > 0.0)
            & (W_right[..., 4] > 0.0)
        )
        valid_hllc = valid_inputs & jnp.all(jnp.isfinite(hllc), axis=-1)
        robust_flux = jnp.where(
            valid_hllc[..., None],
            hllc,
            self.rusanov_flux(U_left, U_right, normal),
        )
        # Besides being mathematically consistent, make the identical-state
        # property bitwise exact. This prevents curved-element inverse mass
        # matrices from amplifying harmless HLLC roundoff into fake waves.
        identical = jnp.all(U_left == U_right, axis=-1)
        return jnp.where(
            identical[..., None],
            self.normal_physical_flux(U_left, normal),
            robust_flux,
        )

    def numerical_flux(self, U_left, U_right, normal):
        if self.inviscid_flux == "hllc":
            return self.hllc_flux(U_left, U_right, normal)
        return self.rusanov_flux(U_left, U_right, normal)

    def viscous_variables(self, U):
        """Return [u,v,w,T], whose gradients define Newton/Fourier fluxes."""
        W = self.conserved_to_primitive(U)
        temperature = W[..., 4] / (
            jnp.maximum(W[..., 0], 1.0e-14) * self.gas_constant
        )
        return jnp.concatenate(
            [W[..., 1:4], temperature[..., None]], axis=-1
        )

    def wale_dynamic_viscosity(self, U, viscous_gradient, filter_width):
        """WALE eddy dynamic viscosity at arbitrary evaluation points."""
        velocity_gradient = viscous_gradient[..., 0:3, :]
        strain = 0.5 * (
            velocity_gradient + jnp.swapaxes(velocity_gradient, -1, -2)
        )
        gradient_squared = jnp.einsum(
            "...ik,...kj->...ij", velocity_gradient, velocity_gradient
        )
        symmetric_squared = 0.5 * (
            gradient_squared + jnp.swapaxes(gradient_squared, -1, -2)
        )
        trace_squared = jnp.trace(gradient_squared, axis1=-2, axis2=-1)
        identity = jnp.eye(3, dtype=U.dtype)
        traceless_squared = (
            symmetric_squared
            - trace_squared[..., None, None] * identity / 3.0
        )
        strain_norm = jnp.sum(strain**2, axis=(-2, -1))
        traceless_norm = jnp.sum(traceless_squared**2, axis=(-2, -1))
        kinematic = (
            (self.wale_constant * filter_width) ** 2
            * traceless_norm**1.5
            / (strain_norm**2.5 + traceless_norm**1.25 + 1.0e-30)
        )
        return jnp.maximum(U[..., 0], 0.0) * kinematic

    def viscous_physical_flux(
        self,
        U,
        gradient,
        filter_width=None,
        turbulent_gradient=None,
    ):
        """Return Cartesian Navier--Stokes viscous flux (...,3,nvars)."""
        W = self.conserved_to_primitive(U)
        velocity = W[..., 1:4]
        viscosity = jnp.full(U.shape[:-1], self.dynamic_viscosity, U.dtype)
        conductivity = jnp.full(
            U.shape[:-1], self.thermal_conductivity, U.dtype
        )
        if self.use_wale and filter_width is not None:
            # BR2 lifting enforces interface/wall consistency, but it is not a
            # resolved turbulent fluctuation.  In particular, an impulsively
            # initialized no-slip wall can make that lifting singularly large.
            # Evaluate WALE from the resolved polynomial gradient instead.
            wale_gradient = (
                gradient if turbulent_gradient is None else turbulent_gradient
            )
            turbulent = self.wale_dynamic_viscosity(
                U, wale_gradient, filter_width
            )
            viscosity = viscosity + turbulent
            conductivity = (
                conductivity
                + turbulent * self.specific_heat_cp / self.turbulent_prandtl
            )

        velocity_gradient = gradient[..., 0:3, :]
        temperature_gradient = gradient[..., 3, :]
        divergence = jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
        identity = jnp.eye(3, dtype=U.dtype)
        stress = viscosity[..., None, None] * (
            velocity_gradient
            + jnp.swapaxes(velocity_gradient, -1, -2)
            - (2.0/3.0) * divergence[..., None, None] * identity
        )
        momentum_flux = jnp.swapaxes(stress, -1, -2)
        viscous_work = jnp.einsum("...i,...id->...d", velocity, stress)
        energy_flux = viscous_work + conductivity[..., None] * temperature_gradient
        flux = jnp.zeros(
            U.shape[:-1] + (3, self.layout.number_of_variables),
            dtype=U.dtype,
        )
        flux = flux.at[..., :, 1:4].set(momentum_flux)
        flux = flux.at[..., :, 4].set(energy_flux)
        return flux

    def chemistry_source(self, U):
        """Return a local conservative source with no mesh coupling."""
        if self.chemistry_source_function is None:
            return jnp.zeros_like(U)
        source = self.chemistry_source_function(U, self)
        if source.shape != U.shape:
            raise ValueError(
                f"Chemistry source shape {source.shape} does not match {U.shape}"
            )
        # Chemistry cannot directly create total mixture mass or momentum.
        source = source.at[..., 0:4].set(0.0)
        return source

    def uniform_state_blocks(
        self,
        density,
        velocity,
        pressure,
        mass_fractions=None,
    ):
        velocity = jnp.asarray(velocity)
        if velocity.shape != (3,):
            raise ValueError("velocity must have shape (3,)")
        if self.layout.species_names:
            if mass_fractions is None:
                raise ValueError("Reacting layouts require mass_fractions")
            mass_fractions = jnp.asarray(mass_fractions)
            if mass_fractions.shape != (len(self.layout.species_names),):
                raise ValueError("mass_fractions has the wrong species count")
            if not jnp.isclose(jnp.sum(mass_fractions), 1.0):
                raise ValueError("mass_fractions must sum to one")

        primitive = [density, *velocity, pressure]
        if self.layout.species_names:
            primitive.extend(mass_fractions)
        primitive = jnp.asarray(primitive)

        blocks = {}
        for block in self.mesh.element_blocks:
            nodes_per_cell = len(
                self.reference_elements[block.element_type].nodes
            )
            W = jnp.broadcast_to(
                primitive,
                (len(block.connectivity), nodes_per_cell, primitive.size),
            )
            blocks[block.element_type] = self.primitive_to_conserved(W)
        return blocks

    def state_blocks_from_primitive_function(self, geometry, function):
        """Evaluate ``function(xyz) -> primitive state`` at solution nodes."""
        blocks = {}
        for block in self.mesh.element_blocks:
            coordinates = geometry.solution_node_coordinates[block.element_type]
            primitive = jnp.asarray(function(coordinates))
            expected = coordinates.shape[:-1] + (self.layout.number_of_variables,)
            if primitive.shape != expected:
                raise ValueError(
                    f"Primitive initializer returned {primitive.shape}; "
                    f"expected {expected} for {block.element_type}"
                )
            blocks[block.element_type] = self.primitive_to_conserved(primitive)
        return blocks
