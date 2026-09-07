"""Abstract ReactionMechanism framework for compressible reacting flows.

Designed to support arbitrary chemical mechanisms (hypersonic air models,
jet fuel, hydrogen-oxygen, and rocket combustion) with zero solver changes.
All evaluations are vectorized, JIT-compatible, and differentiable in JAX.
"""

from dataclasses import dataclass, field
from typing import Sequence
import jax
import jax.numpy as jnp

from .nasa import (
    NASAPolynomial,
    R_UNIV,
    P_STD,
    cp_over_R,
    h_over_RT,
    s_over_R,
    g_over_RT,
    cv_over_R,
    e_over_RT,
)


@dataclass(frozen=True)
class Species:
    """Description of a chemical species."""
    name: str
    molar_mass: float      # kg / mol
    charge: float = 0.0    # elementary charge units (e.g., -1.0 for e-, +1.0 for N+)
    poly: NASAPolynomial | None = None
    is_diatomic: bool = False
    theta_v: float = 0.0   # characteristic vibrational temperature (K)


@dataclass(frozen=True)
class Reaction:
    """Elementary or third-body reaction definition."""
    # (species_index, stoichiometric_coefficient)
    reactants: tuple[tuple[int, int], ...]
    products: tuple[tuple[int, int], ...]
    # Arrhenius parameters: kf = A * T^b * exp(-theta / T) [SI units: m, mol, s, K]
    A: float
    b: float
    theta: float
    # Third body collision partner
    third_body: bool = False
    # species_index -> enhanced collision efficiency (default 1.0 for third-body)
    efficiencies: tuple[tuple[int, float], ...] = ()
    reversible: bool = True
    # Controlling temperature mode for forward and reverse rates:
    # 'T' (translational), 'Ta' (geometric mean sqrt(T*Tv)), 'Tv' (vibrational/electron)
    fwd_temp: str = "T"
    rev_temp: str = "T"


class ReactionMechanism:
    """Base class and engine for reaction mechanisms in JAX.
    
    Provides:
    - Net molar production rates omega_dot [mol / (m^3 * s)]
    - Net mass production rates w_dot [kg / (m^3 * s)]
    - Strict discrete mass conservation: sum(w_dot) == 0
    - Mixture thermodynamic properties (MW, R_mix, cp, cv, gamma, enthalpy, sound speed)
    - Temperature inversion T(e, Y) via vectorized Newton-Raphson
    """

    def __init__(self, species: Sequence[Species], reactions: Sequence[Reaction]):
        self.species = tuple(species)
        self.reactions = tuple(reactions)
        self.num_species = len(self.species)
        self.num_reactions = len(self.reactions)
        self.species_names = tuple(s.name for s in self.species)

        # Build stoichiometric matrices: nu_prime, nu_double_prime, delta_nu
        nu_prime = jnp.zeros((self.num_reactions, self.num_species), dtype=jnp.float64)
        nu_double_prime = jnp.zeros((self.num_reactions, self.num_species), dtype=jnp.float64)

        for r_idx, rxn in enumerate(self.reactions):
            for s_idx, coeff in rxn.reactants:
                nu_prime = nu_prime.at[r_idx, s_idx].add(float(coeff))
            for s_idx, coeff in rxn.products:
                nu_double_prime = nu_double_prime.at[r_idx, s_idx].add(float(coeff))

        self.nu_prime = nu_prime
        self.nu_double_prime = nu_double_prime
        self.delta_nu = nu_double_prime - nu_prime
        self.delta_n = jnp.sum(self.delta_nu, axis=1)

        # Molar masses and charges
        self.molar_masses = jnp.array([s.molar_mass for s in self.species], dtype=jnp.float64)
        self.charges = jnp.array([s.charge for s in self.species], dtype=jnp.float64)

        # Arrhenius constants
        self.A = jnp.array([r.A for r in self.reactions], dtype=jnp.float64)
        self.b = jnp.array([r.b for r in self.reactions], dtype=jnp.float64)
        self.theta = jnp.array([r.theta for r in self.reactions], dtype=jnp.float64)

        # Third body efficiency matrix: shape (num_reactions, num_species)
        alpha_M = jnp.zeros((self.num_reactions, self.num_species), dtype=jnp.float64)
        for r_idx, rxn in enumerate(self.reactions):
            if rxn.third_body:
                # Default efficiency 1.0 for all species except electrons (index 0 if named 'e-')
                for s_idx, s in enumerate(self.species):
                    if s.name != "e-":
                        alpha_M = alpha_M.at[r_idx, s_idx].set(1.0)
                # Apply custom efficiencies
                for s_idx, eff in rxn.efficiencies:
                    alpha_M = alpha_M.at[r_idx, s_idx].set(eff)
        self.alpha_M = alpha_M

        # Temperature mode flags: 0: 'T', 1: 'Ta', 2: 'Tv'
        mode_map = {"T": 0, "Ta": 1, "Tv": 2}
        self.fwd_temp_modes = jnp.array([mode_map[r.fwd_temp] for r in self.reactions], dtype=jnp.int32)
        self.rev_temp_modes = jnp.array([mode_map[r.rev_temp] for r in self.reactions], dtype=jnp.int32)
        self.is_third_body = jnp.array([r.third_body for r in self.reactions], dtype=bool)

        # NASA polynomial containers
        self.polys = tuple(s.poly for s in self.species)

    def select_temperature(self, mode: int | jax.Array, T: jax.Array, Tv: jax.Array) -> jax.Array:
        """Select controlling temperature based on mode flag."""
        Ta = jnp.sqrt(jnp.maximum(T * Tv, 1.0))
        return jnp.where(mode == 1, Ta, jnp.where(mode == 2, Tv, T))

    def forward_rate_constants(self, T: jax.Array, Tv: jax.Array) -> jax.Array:
        """Compute forward rate constants kf_r for all reactions."""
        T_eff = jax.vmap(lambda m: self.select_temperature(m, T, Tv))(self.fwd_temp_modes)
        T_safe = jnp.maximum(T_eff, 10.0)
        return self.A * (T_safe ** self.b) * jnp.exp(-self.theta / T_safe)

    def species_gibbs_over_RT(self, T: jax.Array) -> jax.Array:
        """Compute dimensionless Gibbs free energy g°/(RT) for all species."""
        g_vals = [g_over_RT(poly, T) for poly in self.polys]
        return jnp.stack(g_vals, axis=0)

    def equilibrium_constants(self, T: jax.Array, Tv: jax.Array) -> jax.Array:
        """Compute equilibrium constants Kc_r via detailed balance in SI units [mol/m^3]^delta_n."""
        T_eff = jax.vmap(lambda m: self.select_temperature(m, T, Tv))(self.rev_temp_modes)
        T_safe = jnp.maximum(T_eff, 10.0)

        # Evaluate g°/(RT) at each reaction's effective reverse temperature
        # Use vmap to evaluate g_over_RT across reaction temperatures
        def eval_rxn_Kc(r_idx):
            tr = T_safe[r_idx]
            g_s = jnp.stack([g_over_RT(p, tr) for p in self.polys], axis=0)
            delta_G_over_RT = jnp.dot(self.delta_nu[r_idx], g_s)
            Kp = jnp.exp(-jnp.clip(delta_G_over_RT, -500.0, 500.0))
            conv = (P_STD / (R_UNIV * tr)) ** self.delta_n[r_idx]
            return Kp * conv

        return jax.vmap(eval_rxn_Kc)(jnp.arange(self.num_reactions))

    def net_production_rates(
        self,
        rho_species: jax.Array,
        T: jax.Array,
        Tv: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute molar rates [mol/(m^3*s)] and mass rates [kg/(m^3*s)].
        
        Args:
            rho_species: Partial densities of species, shape (num_species,) in kg/m^3.
            T: Heavy-particle translational temperature in K.
            Tv: Vibrational/electron temperature in K (defaults to T if None).
            
        Returns:
            (omega_dot, w_dot): Molar and mass production rates of each species.
        """
        if Tv is None:
            Tv = T

        # Safe molar concentrations C_s = rho_s / M_s in mol / m^3
        C = jnp.maximum(rho_species / self.molar_masses, 0.0)

        # Forward and reverse rate constants
        kf = self.forward_rate_constants(T, Tv)
        Kc = self.equilibrium_constants(T, Tv)
        # Reverse rate constants kr = kf / Kc
        kr = kf / jnp.maximum(Kc, 1.0e-300)

        # Forward and backward concentration products:
        # qf_r = kf_r * prod_s (C_s ^ nu'_rs)
        # qr_r = kr_r * prod_s (C_s ^ nu''_rs)
        log_C = jnp.log(jnp.maximum(C, 1.0e-100))
        log_fwd_prod = jnp.dot(self.nu_prime, log_C)
        log_rev_prod = jnp.dot(self.nu_double_prime, log_C)

        fwd_reactants = jnp.exp(jnp.clip(log_fwd_prod, -500.0, 500.0))
        rev_products = jnp.exp(jnp.clip(log_rev_prod, -500.0, 500.0))

        # Third-body concentrations [M]_r
        M_eff = jnp.dot(self.alpha_M, C)
        collider = jnp.where(self.is_third_body, M_eff, 1.0)

        qf = kf * fwd_reactants * collider
        qr = kr * rev_products * collider
        q_net = qf - qr

        # Net molar production rates: omega_dot_s = sum_r delta_nu_rs * q_net_r
        omega_dot = jnp.dot(self.delta_nu.T, q_net)

        # Net mass production rates: w_dot_s = M_s * omega_dot_s
        w_dot = omega_dot * self.molar_masses

        # Enforce exact floating-point mass conservation: sum(w_dot) == 0
        total_density = jnp.maximum(jnp.sum(rho_species), 1.0e-14)
        Y = jnp.maximum(rho_species, 0.0) / total_density
        w_dot = w_dot - Y * jnp.sum(w_dot)

        return omega_dot, w_dot

    def mixture_properties(
        self,
        rho_species: jax.Array,
        T: jax.Array,
    ) -> dict[str, jax.Array]:
        """Compute thermodynamic properties of the mixture at state (rho_i, T)."""
        total_density = jnp.maximum(jnp.sum(rho_species), 1.0e-14)
        Y = jnp.maximum(rho_species, 0.0) / total_density
        # Mean molecular weight [kg/mol]
        mean_mw = 1.0 / jnp.sum(Y / self.molar_masses)
        R_mix = R_UNIV / mean_mw

        # Dimensionless species properties
        cp_s = jnp.stack([cp_over_R(p, T) for p in self.polys]) * (R_UNIV / self.molar_masses)
        h_s = jnp.stack([h_over_RT(p, T) for p in self.polys]) * (R_UNIV * T / self.molar_masses)
        e_s = jnp.stack([e_over_RT(p, T) for p in self.polys]) * (R_UNIV * T / self.molar_masses)
        cv_s = jnp.stack([cv_over_R(p, T) for p in self.polys]) * (R_UNIV / self.molar_masses)

        # Mixture properties
        cp_mix = jnp.sum(Y * cp_s)
        cv_mix = jnp.sum(Y * cv_s)
        h_mix = jnp.sum(Y * h_s)
        e_mix = jnp.sum(Y * e_s)
        gamma_frozen = cp_mix / jnp.maximum(cv_mix, 1.0e-10)
        p = total_density * R_mix * T
        speed_of_sound = jnp.sqrt(gamma_frozen * R_mix * T)

        return {
            "density": total_density,
            "mean_molecular_weight": mean_mw,
            "gas_constant": R_mix,
            "pressure": p,
            "cp": cp_mix,
            "cv": cv_mix,
            "enthalpy": h_mix,
            "internal_energy": e_mix,
            "gamma_frozen": gamma_frozen,
            "speed_of_sound": speed_of_sound,
        }

    def recover_temperature_from_energy(
        self,
        rho_species: jax.Array,
        target_internal_energy: jax.Array,
        T_init: float | jax.Array = 1000.0,
        num_iters: int = 10,
    ) -> jax.Array:
        """Invert mixture internal energy e(T, Y) to recover temperature T."""
        total_density = jnp.maximum(jnp.sum(rho_species), 1.0e-14)
        Y = jnp.maximum(rho_species, 0.0) / total_density

        def step(T_curr, _):
            T_safe = jnp.clip(T_curr, 50.0, 50000.0)
            e_s = jnp.stack([e_over_RT(p, T_safe) for p in self.polys]) * (R_UNIV * T_safe / self.molar_masses)
            cv_s = jnp.stack([cv_over_R(p, T_safe) for p in self.polys]) * (R_UNIV / self.molar_masses)
            e_mix = jnp.sum(Y * e_s)
            cv_mix = jnp.sum(Y * cv_s)
            residual = e_mix - target_internal_energy
            dT = -residual / jnp.maximum(cv_mix, 1.0e-5)
            # Damped update to prevent overshoot
            T_next = jnp.clip(T_curr + jnp.clip(dT, -0.5 * T_curr, 0.5 * T_curr), 100.0, 50000.0)
            return T_next, None

        T_final, _ = jax.lax.scan(step, jnp.asarray(T_init, dtype=jnp.float64), None, length=num_iters)
        return T_final
