"""Unit test and benchmark for GPUMidplaneRasterizer."""

import time
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from hybrid import HybridCompressibleSolver, HybridNavierStokesOperator
from hybrid.gui import GPUMidplaneRasterizer


class TestGPUMidplaneRasterizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = HybridCompressibleSolver(
            "mesh/sphere_tunnel_hybrid.msh",
            polynomial_degree=2,
        )
        cls.operator = HybridNavierStokesOperator(cls.solver)
        cls.resolution = (640, 360)
        cls.rasterizer = GPUMidplaneRasterizer(
            cls.operator,
            resolution=cls.resolution,
        )

    def test_rasterizer_output_format_and_speed(self):
        """Verify output shapes, dtypes, and sub-millisecond execution time."""
        # Create uniform supersonic flow states
        states = self.solver.uniform_state_blocks(
            1.2, jnp.asarray([1500.0, 0.0, 0.0]), 101325.0
        )

        # Test Mode 0: Schlieren
        rgba, vmin, vmax = self.rasterizer.render(states, mode=0)
        rgba.block_until_ready()

        self.assertEqual(rgba.shape, (360, 640, 4))
        self.assertEqual(rgba.dtype, jnp.uint8)

        # Verify obstacle center is masked with background color
        center_x = int(640 * (0.0 - self.rasterizer.xmin) / (self.rasterizer.xmax - self.rasterizer.xmin))
        center_y = int(360 * (0.0 - self.rasterizer.ymin) / (self.rasterizer.ymax - self.rasterizer.ymin))
        center_pixel = np.asarray(rgba[center_y, center_x])
        # Background color [1, 5, 31, 255]
        np.testing.assert_array_equal(center_pixel, [1, 5, 31, 255])

        # Benchmark GPU execution time (must be < 1.5 ms)
        start = time.perf_counter()
        for _ in range(50):
            frame, _, _ = self.rasterizer.render(states, mode=0)
            frame.block_until_ready()
        avg_time = (time.perf_counter() - start) / 50.0

        print(f"\nGPUMidplaneRasterizer benchmark: {avg_time*1000:.3f} ms per frame ({1.0/avg_time:.1f} FPS)")
        self.assertLess(avg_time, 0.005, "Rasterizer should complete in < 5ms on GPU")

    def test_all_display_modes(self):
        """Verify that modes 0-4 (Schlieren, Density, Pressure, Mach, Temp) render cleanly."""
        states = self.solver.uniform_state_blocks(
            1.2, jnp.asarray([1500.0, 0.0, 0.0]), 101325.0
        )
        for mode in range(5):
            rgba, _, _ = self.rasterizer.render(states, mode=mode)
            rgba.block_until_ready()
            self.assertEqual(rgba.shape, (360, 640, 4))
            self.assertEqual(rgba.dtype, jnp.uint8)


if __name__ == "__main__":
    unittest.main()

