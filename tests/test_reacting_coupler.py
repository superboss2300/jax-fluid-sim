"""Unit test for ReactingFlowCoupler with Air-11 thermochemistry."""

import unittest
import jax
import jax.numpy as jnp
import numpy as np

from hybrid.chemistry import (
    Air11ParkMechanism,
    AIR11_SPECIES,
    ReactingFlowCoupler,
)


class TestReactingFlowCoupler(unittest.TestCase):
    def setUp(self):
        self.mechanism = Air11ParkMechanism()
        self.coupler = ReactingFlowCoupler(self.mechanism, max_newton_iters=16)

    def test_temperature_inversion_roundtrip(self):
        """Verify temperature recovery from internal energy across temperatures."""
        # 76.7% N2, 23.3% O2
        rho_species = jnp.zeros(11)
        rho_species = rho_species.at[9].set(0.767)  # N2
        rho_species = rho_species.at[10].set(0.233)  # O2

        for T_target in [500.0, 1500.0, 4000.0, 8000.0]:
            props = self.mechanism.mixture_properties(rho_species, jnp.asarray(T_target))
            e_th = props["internal_energy"]
            T_recovered = self.mechanism.recover_temperature_from_energy(
                rho_species, e_th, T_init=1000.0
            )
            self.assertAlmostEqual(float(T_recovered), float(T_target), delta=1.0)

    def test_coupler_block_conservation_and_dissociation(self):
        """Verify batch element update preserves mass, momentum, total energy and dissociates O2."""
        num_elements = 4
        num_nodes = 10
        total_vars = 16

        # Initialize mock element block (E, N, 16)
        rho = 1.2
        u = 2000.0  # Hypersonic velocity ~ Mach 6
        v = 0.0
        w = 0.0
        T = 6000.0  # Stagnation shock-layer temperature

        # Species: N2 and O2
        Y = np.zeros(11)
        Y[9] = 0.767  # N2
        Y[10] = 0.233  # O2
        rho_species = rho * Y

        props = self.mechanism.mixture_properties(jnp.asarray(rho_species), jnp.asarray(T))
        e_th = float(props["internal_energy"])
        e_kin = 0.5 * (u**2 + v**2 + w**2)
        total_energy = rho * (e_th + e_kin)

        u_node = np.zeros(total_vars)
        u_node[0] = rho
        u_node[1] = rho * u
        u_node[2] = rho * v
        u_node[3] = rho * w
        u_node[4] = total_energy
        u_node[5:16] = rho_species

        U_block = jnp.broadcast_to(
            jnp.asarray(u_node), (num_elements, num_nodes, total_vars)
        )

        dt = 1.0e-7  # Hypersonic chemical time step
        U_updated = self.coupler.react_block(U_block, dt)

        # Verify block shapes
        self.assertEqual(U_updated.shape, (num_elements, num_nodes, total_vars))

        # Check mass conservation: sum(rho_i) == rho
        rho_sum = jnp.sum(U_updated[..., 5:16], axis=-1)
        np.testing.assert_allclose(rho_sum, rho, rtol=1.0e-7)

        # Check total density, momentum, energy are unchanged
        np.testing.assert_allclose(U_updated[..., 0], U_block[..., 0], rtol=1.0e-7)
        np.testing.assert_allclose(U_updated[..., 1:4], U_block[..., 1:4], rtol=1.0e-7)
        np.testing.assert_allclose(U_updated[..., 4], U_block[..., 4], rtol=1.0e-7)

        # Check species positivity
        self.assertTrue(bool(jnp.all(U_updated[..., 5:16] >= 0.0)))

        # Check that O2 dissociated into atomic O (species 7 is O, species 10 is O2)
        rho_O = float(U_updated[0, 0, 5 + 7])
        self.assertGreater(rho_O, 1.0e-10, "Oxygen should dissociate at 6000 K")


if __name__ == "__main__":
    unittest.main()

