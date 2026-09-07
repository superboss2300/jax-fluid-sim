"""Unit tests for modular thermochemistry framework and Air-11 Park model."""

import numpy as np
import jax.numpy as jnp

from hybrid.chemistry import (
    Air11ParkMechanism,
    LandauTellerRelaxation,
    PointImplicitIntegrator,
    Species,
    Reaction,
    ReactionMechanism,
    make_nasa7,
)


def test_air11_thermodynamics():
    """Verify species thermodynamic polynomial evaluations are finite and physical."""
    mech = Air11ParkMechanism()
    assert mech.num_species == 11
    assert mech.num_reactions == 22

    # Check properties across low and high temperatures
    for T in [300.0, 1000.0, 4000.0, 8000.0]:
        g = mech.species_gibbs_over_RT(jnp.asarray(T, dtype=jnp.float64))
        assert g.shape == (11,)
        assert jnp.all(jnp.isfinite(g))

        # Check ideal gas mixture properties
        rho_species = jnp.zeros(11, dtype=jnp.float64).at[9].set(0.8).at[10].set(0.2)
        props = mech.mixture_properties(rho_species, jnp.asarray(T, dtype=jnp.float64))
        assert abs(float(props["density"]) - 1.0) < 1e-6
        assert float(props["cp"]) > float(props["cv"]) > 0.0
        assert 1.0 < float(props["gamma_frozen"]) < 1.7
        assert float(props["speed_of_sound"]) > 200.0


def test_air11_park_rate_benchmark():
    """Verify forward rates match the C++ validation benchmark at T = 8000 K."""
    mech = Air11ParkMechanism()
    T = 8000.0
    rho_N2 = 0.8
    rho_O2 = 0.4
    mw_N2 = 2.0 * 0.0140067
    mw_O2 = 2.0 * 0.015999
    c_N2 = rho_N2 / mw_N2
    c_O2 = rho_O2 / mw_O2
    collider = c_N2 + c_O2
    kf = 7.0e15 * (T ** -1.6) * np.exp(-113200.0 / T)
    expected_n_rate = 2.0 * kf * c_N2 * collider  # mol / (m^3 * s)

    # Initial state: only N2 and O2 present
    rho = jnp.zeros(11, dtype=jnp.float64).at[9].set(rho_N2).at[10].set(rho_O2)
    omega_dot, w_dot = mech.net_production_rates(rho, jnp.asarray(T, dtype=jnp.float64))

    # Species 6 is atomic nitrogen 'N'
    observed_n_rate = float(omega_dot[6])
    rel_error = abs(observed_n_rate - expected_n_rate) / expected_n_rate
    print(f"Air-11 benchmark: expected={expected_n_rate:.6e}, observed={observed_n_rate:.6e}, error={rel_error:.4e}")
    assert rel_error < 1.0e-3


def test_mass_conservation():
    """Verify sum of mass production rates is zero down to machine precision."""
    mech = Air11ParkMechanism()

    test_states = [
        # Standard cold air
        jnp.zeros(11, dtype=jnp.float64).at[9].set(0.79).at[10].set(0.21),
        # Partially dissociated air
        jnp.array([1e-6, 1e-5, 1e-5, 1e-4, 1e-5, 1e-5, 0.2, 0.1, 0.05, 0.4, 0.1], dtype=jnp.float64),
        # Ionized air
        jnp.array([1e-4, 0.05, 0.05, 0.05, 0.02, 0.02, 0.3, 0.2, 0.01, 0.1, 0.05], dtype=jnp.float64),
    ]

    for state in test_states:
        for T in [500.0, 3000.0, 8000.0, 15000.0]:
            _, w_dot = mech.net_production_rates(state, jnp.asarray(T, dtype=jnp.float64))
            max_rate = float(jnp.max(jnp.abs(w_dot)))
            if max_rate > 1.0e-10:
                rel_mass_sum = float(abs(jnp.sum(w_dot))) / max_rate
                assert rel_mass_sum < 1.0e-14
            else:
                assert abs(float(jnp.sum(w_dot))) < 1.0e-12


def test_vibrational_relaxation():
    """Verify Landau-Teller relaxation drives Tv toward T."""
    mech = Air11ParkMechanism()
    relax = LandauTellerRelaxation(mech.species)

    rho = jnp.zeros(11, dtype=jnp.float64).at[9].set(0.8).at[10].set(0.2)
    T = jnp.asarray(5000.0, dtype=jnp.float64)
    Tv = jnp.asarray(1000.0, dtype=jnp.float64)
    p = jnp.asarray(101325.0, dtype=jnp.float64)

    # When T > Tv, vibrational energy should increase (Q_T-V > 0)
    Q = float(relax.energy_exchange_rate(rho, T, Tv, p))
    assert Q > 0.0

    # Temperature recovery roundtrip
    Ev = relax.mixture_vibrational_energy(rho, Tv)
    Tv_recovered = float(relax.recover_vibrational_temperature(rho, Ev, Tv_init=800.0))
    assert abs(Tv_recovered - 1000.0) < 1.0e-2


def test_point_implicit_integrator():
    """Verify point-implicit integration is stable, positive, and conserves mass."""
    mech = Air11ParkMechanism()
    integrator = PointImplicitIntegrator(mech, max_newton_iters=16, tolerance=1.0e-7)

    rho_0 = jnp.zeros(11, dtype=jnp.float64).at[9].set(0.8).at[10].set(0.4)
    total_mass_0 = float(jnp.sum(rho_0))
    T = jnp.asarray(8000.0, dtype=jnp.float64)

    # Step forward by 1e-6 s (large compared to chemical stiffness)
    rho_new, T_new, _ = integrator.step(rho_0, T, dt=1.0e-6, constant_temperature=True)

    # Positivity check
    assert jnp.all(rho_new >= 0.0)
    # Mass conservation check
    assert abs(float(jnp.sum(rho_new)) - total_mass_0) < 1e-12
    # Dissociation produces N and O atoms
    assert float(rho_new[6]) > 0.0  # N produced
    assert float(rho_new[7]) > 0.0  # O produced
    assert float(rho_new[9]) < float(rho_0[9])  # N2 consumed
    assert float(rho_new[10]) < float(rho_0[10])  # O2 consumed


def test_rocket_combustion_extensibility():
    """Verify framework extensibility: easily define a rocket/jet combustion mechanism."""
    # Toy 3-species hydrogen-oxygen mechanism: 2 H2 + O2 <=> 2 H2O
    h2_poly = make_nasa7(200.0, 1000.0, 6000.0,
                         [3.298, 8.249e-4, -8.143e-7, -9.475e-11, 4.134e-13, -1012.5, -3.294],
                         [2.932, 8.266e-4, -1.464e-7, 1.541e-11, -6.888e-16, -813.0, -1.024])
    o2_poly = make_nasa7(200.0, 1000.0, 6000.0,
                         [3.782, -2.996e-3, 9.847e-6, -9.681e-9, 3.243e-12, -1063.9, 3.657],
                         [3.660, 6.563e-4, -1.411e-7, 2.057e-11, -1.299e-15, -1215.9, 3.415])
    h2o_poly = make_nasa7(200.0, 1000.0, 6000.0,
                          [4.198, -2.036e-3, 6.520e-6, -5.487e-9, 1.771e-12, -30293.7, -0.849],
                          [2.677, 2.973e-3, -7.737e-7, 9.443e-11, -4.269e-15, -29885.8, 6.882])

    species = [
        Species("H2", molar_mass=0.002016, poly=h2_poly),
        Species("O2", molar_mass=0.031998, poly=o2_poly),
        Species("H2O", molar_mass=0.018015, poly=h2o_poly),
    ]

    # Global combustion reaction: 2 H2 + O2 -> 2 H2O
    reactions = [
        Reaction(
            reactants=((0, 2), (1, 1)),
            products=((2, 2),),
            A=1.0e11, b=0.0, theta=15000.0,
            reversible=True,
        )
    ]

    rocket_mech = ReactionMechanism(species, reactions)
    assert rocket_mech.num_species == 3
    assert rocket_mech.num_reactions == 1

    # Evaluate at combustion state: rich H2/O2 at 2000 K
    rho = jnp.array([0.2, 0.8, 0.01], dtype=jnp.float64)
    omega_dot, w_dot = rocket_mech.net_production_rates(rho, jnp.asarray(2000.0, dtype=jnp.float64))

    assert float(w_dot[2]) > 0.0  # H2O is being produced
    assert float(w_dot[0]) < 0.0  # H2 is being consumed
    assert float(w_dot[1]) < 0.0  # O2 is being consumed
    assert abs(float(jnp.sum(w_dot))) < 1.0e-12  # Exact mass conservation


if __name__ == "__main__":
    print("Running test_air11_thermodynamics...")
    test_air11_thermodynamics()
    print("Running test_air11_park_rate_benchmark...")
    test_air11_park_rate_benchmark()
    print("Running test_mass_conservation...")
    test_mass_conservation()
    print("Running test_vibrational_relaxation...")
    test_vibrational_relaxation()
    print("Running test_point_implicit_integrator...")
    test_point_implicit_integrator()
    print("Running test_rocket_combustion_extensibility...")
    test_rocket_combustion_extensibility()
    print("ALL AIR-11 & THERMOCHEMISTRY TESTS PASSED SUCCESSFULLY!")

