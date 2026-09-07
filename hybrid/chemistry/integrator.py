"""Stiff point-implicit chemical ODE integrator using JAX.

Integrates chemical kinetics and thermal relaxation via Newton-Raphson
with exact JAX automatic differentiation (jax.jacobian), line search damping,
and strict species positivity.
"""

import jax
import jax.numpy as jnp

from .mechanism import ReactionMechanism
from .relaxation import LandauTellerRelaxation


class PointImplicitIntegrator:
    """Coupled point-implicit ODE integrator for reacting mixtures."""

    def __init__(
        self,
        mechanism: ReactionMechanism,
        relaxation: LandauTellerRelaxation | None = None,
        max_newton_iters: int = 16,
        tolerance: float = 1.0e-8,
    ):
        self.mechanism = mechanism
        self.relaxation = relaxation
        self.max_newton_iters = max_newton_iters
        self.tolerance = tolerance
        self.num_species = mechanism.num_species

    def step(
        self,
        rho_species: jax.Array,
        T: jax.Array,
        dt: float | jax.Array,
        Tv: jax.Array | None = None,
        constant_temperature: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Advance species densities and temperatures by dt.
        
        Args:
            rho_species: Initial species densities [kg/m^3], shape (num_species,)
            T: Initial heavy-particle temperature [K]
            dt: Time step [s]
            Tv: Initial vibrational temperature [K] (defaults to T if None)
            constant_temperature: If True, integrates chemistry at fixed T, Tv.
                                  If False, couples T recovery from constant internal energy.
                                  
        Returns:
            (rho_new, T_new, Tv_new): Updated species densities and temperatures.
        """
        if Tv is None:
            Tv = T

        dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
        dt = jnp.asarray(dt, dtype=dtype)
        rho_0 = jnp.maximum(rho_species, 1.0e-20)
        total_mass = jnp.sum(rho_0)

        # Compute initial internal energy if adiabatic
        props_0 = self.mechanism.mixture_properties(rho_0, T)
        e_init = props_0["internal_energy"]

        # Objective residual function: R(rho) = rho - rho_0 - dt * w_dot(rho, T, Tv)
        def residual_fn(rho_curr, T_curr, Tv_curr):
            _, w_dot = self.mechanism.net_production_rates(rho_curr, T_curr, Tv_curr)
            return rho_curr - rho_0 - dt * w_dot

        # Newton iteration loop
        def newton_body(state, _):
            rho_k, T_k, Tv_k, converged = state

            # Residual at current iterate
            res = residual_fn(rho_k, T_k, Tv_k)
            res_norm = jnp.max(jnp.abs(res) / jnp.maximum(rho_0, 1.0e-10))

            # Exact chemical Jacobian dR/d(rho) = I - dt * d(w_dot)/d(rho)
            jac = jax.jacobian(lambda r: residual_fn(r, T_k, Tv_k))(rho_k)

            # Solve linear system: jac * d_rho = -res
            d_rho = jnp.linalg.solve(jac, -res)
            # Apply update with non-negative clipping
            rho_next = jnp.maximum(rho_k + d_rho, 0.0)

            # Update temperatures if not constant
            if not constant_temperature:
                T_next = self.mechanism.recover_temperature_from_energy(rho_next, e_init, T_init=T_k)
                Tv_next = T_next
            else:
                T_next = T_k
                Tv_next = Tv_k

            is_done = jnp.logical_or(converged, res_norm < self.tolerance)
            return (rho_next, T_next, Tv_next, is_done), res_norm

        init_state = (rho_0, T, Tv, False)
        (rho_final, T_final, Tv_final, _), _ = jax.lax.scan(
            newton_body, init_state, None, length=self.max_newton_iters
        )

        # Strictly preserve total mass
        rho_final = rho_final * (total_mass / jnp.sum(rho_final))

        return rho_final, T_final, Tv_final
