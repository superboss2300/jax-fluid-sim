"""Landau-Teller vibrational relaxation with Millikan-White and Park corrections.

Models vibrational-translational energy exchange (Q_T-V) for diatomic molecules
in hypersonic thermal non-equilibrium flows.
"""

from typing import Sequence
import jax
import jax.numpy as jnp

from .nasa import R_UNIV, P_STD
from .mechanism import Species

# Avogadro constant [1/mol]
N_AVO = 6.02214076e23


class LandauTellerRelaxation:
    """Computes Landau-Teller vibrational energy exchange Q_T-V [W / m^3]."""

    def __init__(self, species: Sequence[Species]):
        self.species = tuple(species)
        self.num_species = len(species)

        # Diatomic masks and properties
        diatomic_indices = []
        theta_v_list = []
        mw_list = []

        for idx, s in enumerate(species):
            if s.is_diatomic and s.theta_v > 0.0:
                diatomic_indices.append(idx)
                theta_v_list.append(s.theta_v)
                mw_list.append(s.molar_mass)

        self.diatomic_indices = jnp.array(diatomic_indices, dtype=jnp.int32)
        self.theta_v = jnp.array(theta_v_list, dtype=jnp.float64)
        self.mw_diatomic = jnp.array(mw_list, dtype=jnp.float64)
        self.num_diatomics = len(diatomic_indices)
        self.molar_masses = jnp.array([s.molar_mass for s in species], dtype=jnp.float64)

    def specific_vibrational_energy(self, Tv: jax.Array) -> jax.Array:
        """Specific vibrational energy e_v,m(Tv) [J / kg] for all diatomic species."""
        Tv_safe = jnp.maximum(Tv, 50.0)
        exp_arg = jnp.clip(self.theta_v / Tv_safe, 1.0e-4, 80.0)
        denom = jnp.expm1(exp_arg)
        # e_v = (R / M) * theta_v / (exp(theta_v / Tv) - 1)
        return (R_UNIV / self.mw_diatomic) * (self.theta_v / jnp.maximum(denom, 1.0e-30))

    def mixture_vibrational_energy(self, rho_species: jax.Array, Tv: jax.Array) -> jax.Array:
        """Total mixture vibrational energy density E_v = sum(rho_m * e_v,m) [J / m^3]."""
        e_v = self.specific_vibrational_energy(Tv)
        rho_diatomic = rho_species[self.diatomic_indices]
        return jnp.sum(rho_diatomic * e_v)

    def recover_vibrational_temperature(
        self,
        rho_species: jax.Array,
        target_Ev: jax.Array,
        Tv_init: float | jax.Array = 1000.0,
        num_iters: int = 10,
    ) -> jax.Array:
        """Invert E_v(Tv) to recover vibrational temperature Tv via Newton-Raphson."""
        rho_diatomic = rho_species[self.diatomic_indices]
        total_rho_d = jnp.maximum(jnp.sum(rho_diatomic), 1.0e-14)

        def step(Tv_curr, _):
            Tv_safe = jnp.clip(Tv_curr, 50.0, 50000.0)
            exp_arg = jnp.clip(self.theta_v / Tv_safe, 1.0e-4, 80.0)
            exp_val = jnp.exp(exp_arg)
            expm1_val = jnp.expm1(exp_arg)
            denom = jnp.maximum(expm1_val, 1.0e-30)

            # e_v = (R / M) * theta_v / denom
            e_v = (R_UNIV / self.mw_diatomic) * (self.theta_v / denom)
            # cv_v = de_v / dTv = (R / M) * (theta_v / Tv)^2 * exp / (exp - 1)^2
            cv_v = (R_UNIV / self.mw_diatomic) * ((self.theta_v / Tv_safe)**2) * (exp_val / (denom**2))

            E_v = jnp.sum(rho_diatomic * e_v)
            dE_v_dTv = jnp.maximum(jnp.sum(rho_diatomic * cv_v), 1.0e-10)

            residual = E_v - target_Ev
            dTv = -residual / dE_v_dTv
            Tv_next = jnp.clip(Tv_curr + jnp.clip(dTv, -0.5 * Tv_curr, 0.5 * Tv_curr), 50.0, 50000.0)
            return Tv_next, None

        Tv_final, _ = jax.lax.scan(step, jnp.asarray(Tv_init, dtype=jnp.float64), None, length=num_iters)
        return Tv_final

    def relaxation_times(self, rho_species: jax.Array, T: jax.Array, p: jax.Array) -> jax.Array:
        """Compute Landau-Teller relaxation times tau_m [s] for all diatomic species."""
        T_safe = jnp.maximum(T, 100.0)
        p_safe = jnp.maximum(p, 1.0)
        p_atm = p_safe / P_STD

        # Safe molar fractions X_s
        C = jnp.maximum(rho_species / self.molar_masses, 0.0)
        total_C = jnp.maximum(jnp.sum(C), 1.0e-14)
        X = C / total_C

        # Number density n = total_C * N_AVO [1/m^3]
        n_density = total_C * N_AVO

        def compute_tau_m(m_idx):
            # Diatomic species index in full species array
            s_m = self.diatomic_indices[m_idx]
            mw_m_g = self.mw_diatomic[m_idx] * 1000.0  # g / mol
            theta_m = self.theta_v[m_idx]

            # Pairwise Millikan-White times tau_ms
            def pair_tau(s_idx):
                mw_s_g = self.molar_masses[s_idx] * 1000.0  # g / mol
                mu_ms = (mw_m_g * mw_s_g) / (mw_m_g + mw_s_g)  # reduced mass in g/mol
                A_ms = 1.16e-3 * jnp.sqrt(mu_ms) * (theta_m ** (4.0 / 3.0))
                # ln(p * tau_ms [atm * s]) = A_ms * (T^(-1/3) - 0.015 * mu_ms^(1/4)) - 18.42
                ln_p_tau = A_ms * (T_safe ** (-1.0 / 3.0) - 0.015 * (mu_ms ** 0.25)) - 18.42
                ln_p_tau_clamped = jnp.clip(ln_p_tau, -30.0, 30.0)
                p_tau_atm = jnp.exp(ln_p_tau_clamped)
                tau_ms = p_tau_atm / p_atm
                return jnp.maximum(tau_ms, 1.0e-12)

            tau_pairs = jax.vmap(pair_tau)(jnp.arange(self.num_species))
            # Harmonic average weighted by mole fraction: 1 / tau_MW = sum(X_s / tau_ms)
            tau_MW = 1.0 / jnp.maximum(jnp.sum(X / tau_pairs), 1.0e-12)

            # Park high-temperature limiting collision time: tau_P = 1 / (sigma_v * c_bar * n)
            # Limiting cross section sigma_v [m^2]
            sigma_v = 1.0e-21 * ((50000.0 / T_safe) ** 2)
            # Thermal average speed c_bar = sqrt(8 * R * T / (pi * M))
            c_bar = jnp.sqrt(8.0 * R_UNIV * T_safe / (jnp.pi * self.mw_diatomic[m_idx]))
            tau_Park = 1.0 / jnp.maximum(sigma_v * c_bar * n_density, 1.0e-12)

            # Effective relaxation time
            return tau_MW + tau_Park

        return jax.vmap(compute_tau_m)(jnp.arange(self.num_diatomics))

    def energy_exchange_rate(
        self,
        rho_species: jax.Array,
        T: jax.Array,
        Tv: jax.Array,
        p: jax.Array,
    ) -> jax.Array:
        """Compute net Landau-Teller energy exchange Q_T-V [W / m^3]."""
        e_v_star = self.specific_vibrational_energy(T)
        e_v = self.specific_vibrational_energy(Tv)
        tau = self.relaxation_times(rho_species, T, p)

        rho_diatomic = rho_species[self.diatomic_indices]
        q_m = rho_diatomic * (e_v_star - e_v) / jnp.maximum(tau, 1.0e-12)
        return jnp.sum(q_m)
