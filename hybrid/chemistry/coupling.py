"""Vectorized fluid-chemistry coupling for compressible reacting flows.

Connects the 16-variable Navier--Stokes state:
    U = [rho, rho*u, rho*v, rho*w, E, rho_e-, rho_N+, ..., rho_O2]
with the stiff PointImplicitIntegrator thermochemical kinetics.
"""

from typing import Sequence
import jax
import jax.numpy as jnp
import numpy as np

from .mechanism import ReactionMechanism
from .air11_park import Air11ParkMechanism, AIR11_SPECIES
from .relaxation import LandauTellerRelaxation
from .integrator import PointImplicitIntegrator


class ReactingFlowCoupler:
    """Vectorized thermochemistry operator for hybrid DG/FV element blocks.

    Provides operator-split chemical reaction advance, temperature recovery from
    conserved total energy, frozen sound speed, and species mass consistency.
    """

    def __init__(
        self,
        mechanism: ReactionMechanism | None = None,
        relaxation: LandauTellerRelaxation | None = None,
        max_newton_iters: int = 12,
        tolerance: float = 1.0e-7,
        dimensionless: bool = False,
        reference_temperature: float = 300.0,
        reference_pressure: float = 101325.0,
        reference_length: float = 1.0,
        gamma: float = 1.4,
    ):
        if mechanism is None:
            self.mechanism = Air11ParkMechanism()
        else:
            self.mechanism = mechanism

        self.relaxation = relaxation
        self.integrator = PointImplicitIntegrator(
            self.mechanism,
            relaxation=self.relaxation,
            max_newton_iters=max_newton_iters,
            tolerance=tolerance,
        )
        self.num_species = self.mechanism.num_species
        self.species_names = self.mechanism.species_names
        self.dimensionless = bool(dimensionless)
        self.gamma = float(gamma)

        # Physical aerothermodynamic reference scales for non-dimensional flow
        self.T_ref = float(reference_temperature)
        self.p_ref = float(reference_pressure)
        self.L_ref = float(reference_length)
        R_univ = 8.31446261815324
        mw_n2 = 2.0 * 0.0140067
        mw_o2 = 2.0 * 0.015999
        mean_mw = 1.0 / (0.767 / mw_n2 + 0.233 / mw_o2)
        self.R_ref = R_univ / mean_mw
        self.rho_ref = self.p_ref / (self.R_ref * self.T_ref)
        self.u_ref = float(np.sqrt(self.R_ref * self.T_ref))
        self.e_ref = float(self.u_ref**2)
        self.t_ref = float(self.L_ref / self.u_ref)

        # Vectorize single-cell reaction over (elements, nodes)
        self._vmapped_cell_step = jax.vmap(
            jax.vmap(self._react_single_cell, in_axes=(0, None)),
            in_axes=(0, None),
        )

    def compute_temperature(
        self,
        rho_species: jax.Array,
        internal_energy: jax.Array,
        T_init: float | jax.Array = 1000.0,
    ) -> jax.Array:
        """Recover heavy-particle translational temperature T from internal energy.

        Args:
            rho_species: Partial densities (..., num_species) [kg/m^3]
            internal_energy: Specific thermal internal energy (...,) [J/kg]
            T_init: Initial temperature guess [K]
        """
        flat_rho = rho_species.reshape(-1, self.num_species)
        flat_e = internal_energy.reshape(-1)
        flat_t_init = jnp.broadcast_to(T_init, flat_e.shape)

        def eval_single(r, e, t0):
            return self.mechanism.recover_temperature_from_energy(r, e, T_init=t0)

        flat_T = jax.vmap(eval_single)(flat_rho, flat_e, flat_t_init)
        return flat_T.reshape(internal_energy.shape)

    def mixture_properties(
        self,
        rho_species: jax.Array,
        T: jax.Array,
    ) -> dict[str, jax.Array]:
        """Evaluate mixture thermodynamic properties over arbitrary batch shapes."""
        flat_rho = rho_species.reshape(-1, self.num_species)
        flat_T = T.reshape(-1)

        def eval_single(r, t):
            return self.mechanism.mixture_properties(r, t)

        flat_props = jax.vmap(eval_single)(flat_rho, flat_T)
        return {
            k: v.reshape(T.shape) if v.ndim == 1 else v.reshape(T.shape + v.shape[1:])
            for k, v in flat_props.items()
        }

    def _react_single_cell(
        self,
        u_node: jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Apply point-implicit reaction update to a single node state vector U (16,)."""
        rho_total = jnp.maximum(u_node[0], 1.0e-14)
        momentum = u_node[1:4]
        total_energy = u_node[4]
        rho_species = u_node[5:5 + self.num_species]

        # Ensure species sum equals rho_total and species are non-negative
        rho_species = jnp.maximum(rho_species, 0.0)
        sum_species = jnp.maximum(jnp.sum(rho_species), 1.0e-20)
        rho_species = rho_species * (rho_total / sum_species)

        # Kinetic energy and specific thermal internal energy
        e_kin = 0.5 * jnp.sum(momentum**2) / (rho_total**2)

        if self.dimensionless:
            e_thermal_star = jnp.maximum((total_energy / rho_total) - e_kin, 1.0e-4)
            e_thermal = e_thermal_star * self.e_ref
            rho_species_phys = rho_species * self.rho_ref
            T_guess = jnp.maximum((self.gamma - 1.0) * e_thermal_star * self.T_ref, 200.0)
            dt_phys = dt * self.t_ref

            # Recover physical temperature
            T_init = self.mechanism.recover_temperature_from_energy(
                rho_species_phys, e_thermal, T_init=T_guess, num_iters=8
            )

            # Implicit chemical kinetics advance with coupled temperature recovery
            rho_new_phys, _, _ = self.integrator.step(
                rho_species_phys,
                T_init,
                dt_phys,
                constant_temperature=False,
            )
            rho_new = rho_new_phys / self.rho_ref
        else:
            e_thermal = jnp.maximum(
                (total_energy / rho_total) - e_kin,
                1.0e3,
            )
            T_init = self.mechanism.recover_temperature_from_energy(
                rho_species, e_thermal, T_init=1000.0, num_iters=8
            )
            rho_new, _, _ = self.integrator.step(
                rho_species,
                T_init,
                dt,
                constant_temperature=False,
            )

        # Strictly enforce mass conservation and non-negativity
        rho_new = jnp.maximum(rho_new, 0.0)
        sum_new = jnp.maximum(jnp.sum(rho_new), 1.0e-20)
        rho_new = rho_new * (rho_total / sum_new)

        # Assemble updated conservative node vector
        u_out = u_node.at[5:5 + self.num_species].set(rho_new)
        return u_out

    def react_block(
        self,
        U_block: jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Apply point-implicit reaction advance to an element block (E, N, 16)."""
        if U_block.shape[-1] < 5 + self.num_species:
            raise ValueError(
                f"State has {U_block.shape[-1]} variables; expected at least "
                f"{5 + self.num_species} for reacting mixture."
            )
        if isinstance(dt, (jax.Array, np.ndarray)) and dt.ndim > 0:
            flat_dt = jnp.squeeze(dt)
            vmap_elem = jax.vmap(
                jax.vmap(self._react_single_cell, in_axes=(0, None)),
                in_axes=(0, 0),
            )
            return vmap_elem(U_block, flat_dt)
        return self._vmapped_cell_step(U_block, dt)

    def advance(
        self,
        states: dict[str, jax.Array],
        dt: float | jax.Array,
    ) -> dict[str, jax.Array]:
        """Apply point-implicit chemical reaction advance to all mesh blocks."""
        return {
            name: self.react_block(block, dt)
            for name, block in states.items()
        }

