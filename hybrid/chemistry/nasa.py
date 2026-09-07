"""NASA-7 thermodynamic property evaluation using JAX.

Evaluates dimensionless heat capacity (cp/R), enthalpy (h/RT), entropy (s/R),
Gibbs free energy (g/RT), internal energy (e/RT), and isochoric heat capacity (cv/R)
for standard NASA-7 polynomial fits across low and high temperature ranges.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp

# Universal gas constant in SI units: J / (mol * K)
R_UNIV = 8.31446261815324
# Standard atmospheric pressure in Pa
P_STD = 101325.0


class NASAPolynomial(NamedTuple):
    """NASA-7 polynomial parameterization for a single chemical species."""
    t_low: float
    t_mid: float
    t_high: float
    # low-temperature coefficients [a0, a1, a2, a3, a4, a5, a6]
    coeffs_low: jax.Array
    # high-temperature coefficients [a0, a1, a2, a3, a4, a5, a6]
    coeffs_high: jax.Array


def make_nasa7(
    t_low: float,
    t_mid: float,
    t_high: float,
    low: list[float] | tuple[float, ...],
    high: list[float] | tuple[float, ...],
) -> NASAPolynomial:
    """Construct a NASAPolynomial with jnp array coefficients."""
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    return NASAPolynomial(
        t_low=float(t_low),
        t_mid=float(t_mid),
        t_high=float(t_high),
        coeffs_low=jnp.asarray(low, dtype=dtype),
        coeffs_high=jnp.asarray(high, dtype=dtype),
    )


@jax.jit
def select_coeffs(poly: NASAPolynomial, T: float | jax.Array) -> jax.Array:
    """Select between low and high temperature coefficient vectors."""
    return jnp.where(T < poly.t_mid, poly.coeffs_low, poly.coeffs_high)


@jax.jit
def cp_over_R(poly: NASAPolynomial, T: float | jax.Array) -> jax.Array:
    """Dimensionless molar heat capacity at constant pressure cp° / R."""
    T = jnp.asarray(T, dtype=jnp.float64)
    T_clamped = jnp.clip(T, poly.t_low, poly.t_high)
    a = select_coeffs(poly, T_clamped)
    # a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4
    return (
        a[0]
        + a[1] * T_clamped
        + a[2] * (T_clamped**2)
        + a[3] * (T_clamped**3)
        + a[4] * (T_clamped**4)
    )


@jax.jit
def h_over_RT(poly: NASAPolynomial, T: float | jax.Array) -> jax.Array:
    """Dimensionless molar enthalpy h° / (R * T)."""
    T = jnp.asarray(T, dtype=jnp.float64)
    T_clamped = jnp.clip(T, poly.t_low, poly.t_high)
    a = select_coeffs(poly, T_clamped)
    # a0 + a1*T/2 + a2*T^2/3 + a3*T^3/4 + a4*T^4/5 + a5/T
    return (
        a[0]
        + a[1] * T_clamped / 2.0
        + a[2] * (T_clamped**2) / 3.0
        + a[3] * (T_clamped**3) / 4.0
        + a[4] * (T_clamped**4) / 5.0
        + a[5] / T_clamped
    )


@jax.jit
def s_over_R(poly: NASAPolynomial, T: float | jax.Array) -> jax.Array:
    """Dimensionless molar entropy s° / R."""
    T = jnp.asarray(T, dtype=jnp.float64)
    T_clamped = jnp.clip(T, poly.t_low, poly.t_high)
    a = select_coeffs(poly, T_clamped)
    # a0*ln(T) + a1*T + a2*T^2/2 + a3*T^3/3 + a4*T^4/4 + a6
    return (
        a[0] * jnp.log(T_clamped)
        + a[1] * T_clamped
        + a[2] * (T_clamped**2) / 2.0
        + a[3] * (T_clamped**3) / 3.0
        + a[4] * (T_clamped**4) / 4.0
        + a[6]
    )


@jax.jit
def g_over_RT(poly: NASAPolynomial, T: float | jax.Array) -> jax.Array:
    """Dimensionless molar Gibbs free energy g° / (R * T) = h/(RT) - s/R."""
    return h_over_RT(poly, T) - s_over_R(poly, T)


@jax.jit
def e_over_RT(poly: NASAPolynomial, T: float | jax.Array) -> jax.Array:
    """Dimensionless molar internal energy e° / (R * T) = h/(RT) - 1 for ideal gas."""
    return h_over_RT(poly, T) - 1.0


@jax.jit
def cv_over_R(poly: NASAPolynomial, T: float | jax.Array) -> jax.Array:
    """Dimensionless molar heat capacity at constant volume cv° / R = cp/R - 1."""
    return cp_over_R(poly, T) - 1.0

