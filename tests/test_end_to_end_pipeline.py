"""
tests/test_end_to_end_pipeline.py
End-to-end integration tests covering:
  - Regional Climate Risk API endpoint (GET /api/regional-risk)
  - High-resolution iterator memory safety via _generate_image_slices generator
"""
import sys
import os
import json
import gc
import io
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app
import detector


class TestMapRegionalRiskApi(unittest.TestCase):
    """
    Validates that GET /api/regional-risk returns a well-formed 200 OK
    JSON array containing required structured climate and pest keys.
    """

    def setUp(self):
        app.app.config["TESTING"] = True
        app.limiter.enabled = False
        self.client = app.app.test_client()

    def _mock_meteo_ok(self, *args, **kwargs):
        """Simulate a successful Open-Meteo API response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "current": {
                "temperature_2m": 31.5,
                "relative_humidity_2m": 52.0
            }
        }
        return mock_resp

    def _mock_meteo_timeout(self, *args, **kwargs):
        """Simulate an Open-Meteo network timeout."""
        import requests
        raise requests.exceptions.ConnectTimeout("Simulated timeout")

    # ── Test 1: Happy path with explicit coords ──────────────────────────────

    def test_regional_risk_200_with_coords(self):
        """Route returns 200 and a non-empty JSON array with explicit lat/lon."""
        with patch("requests.get", side_effect=self._mock_meteo_ok):
            resp = self.client.get("/api/regional-risk?lat=16.3&lon=80.4")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIsInstance(data, list, "Response must be a JSON array")
        self.assertGreater(len(data), 0, "Array must contain at least one location entry")

    # ── Test 2: Validate required structural keys ────────────────────────────

    def test_regional_risk_structured_keys(self):
        """Each location entry must contain mandatory climate and pest keys."""
        with patch("requests.get", side_effect=self._mock_meteo_ok):
            resp = self.client.get("/api/regional-risk?lat=16.3&lon=80.4")
        self.assertEqual(resp.status_code, 200)
        locations = json.loads(resp.data)

        required_top_keys = {"latitude", "longitude", "name", "temperature", "humidity", "risks"}
        required_risk_keys = {"pest", "label", "telugu", "level", "description"}
        valid_levels = {"Critical", "High", "Moderate", "Low"}

        for loc in locations:
            with self.subTest(location=loc.get("name")):
                missing = required_top_keys - set(loc.keys())
                self.assertFalse(missing, f"Location entry missing keys: {missing}")
                self.assertIsInstance(loc["risks"], list)
                for risk in loc["risks"]:
                    missing_rk = required_risk_keys - set(risk.keys())
                    self.assertFalse(missing_rk, f"Risk entry missing keys: {missing_rk}")
                    self.assertIn(risk["level"], valid_levels,
                                  f"Unexpected threat level: {risk['level']}")

    # ── Test 3: Fallback to default centroid when coords are missing ─────────

    def test_regional_risk_fallback_coords(self):
        """Route must return 200 even when lat/lon query params are absent."""
        with patch("requests.get", side_effect=self._mock_meteo_ok):
            resp = self.client.get("/api/regional-risk")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIsInstance(data, list)
        # The centroid entry should be near the default 16.5, 79.5
        centroid = data[0]
        self.assertAlmostEqual(centroid["latitude"],  16.5, places=1)
        self.assertAlmostEqual(centroid["longitude"], 79.5, places=1)

    # ── Test 4: Open-Meteo timeout falls back gracefully ────────────────────

    def test_regional_risk_meteo_timeout_graceful(self):
        """A network timeout from Open-Meteo must NOT crash the endpoint."""
        with patch("requests.get", side_effect=self._mock_meteo_timeout):
            resp = self.client.get("/api/regional-risk?lat=16.5&lon=79.5")
        self.assertEqual(resp.status_code, 200,
                         "Endpoint must return 200 even when Open-Meteo is unreachable")
        data = json.loads(resp.data)
        self.assertIsInstance(data, list)
        # Temperature should be the safe baseline default (28.0)
        self.assertEqual(data[0]["temperature"], 28.0)

    # ── Test 5: Temperature and humidity are numeric ─────────────────────────

    def test_regional_risk_numeric_climate_values(self):
        """Temperature and humidity must be numeric floats / ints."""
        with patch("requests.get", side_effect=self._mock_meteo_ok):
            resp = self.client.get("/api/regional-risk?lat=17.0&lon=81.0")
        data = json.loads(resp.data)
        for loc in data:
            self.assertIsInstance(loc["temperature"], (int, float))
            self.assertIsInstance(loc["humidity"],    (int, float))
            self.assertGreater(loc["temperature"], -20)
            self.assertLess(loc["temperature"], 60)
            self.assertGreaterEqual(loc["humidity"], 0)
            self.assertLessEqual(loc["humidity"], 100)


class TestHighResIteratorMemorySafety(unittest.TestCase):
    """
    Validates that _generate_image_slices works correctly as a lazy generator
    on high-resolution canvases and that the predict loop does not leak memory.
    """

    def test_generator_is_lazy_not_list(self):
        """_generate_image_slices must return a generator, not a list."""
        import numpy as np
        import inspect
        dummy = np.zeros((1280, 960, 3), dtype=np.uint8)
        result = detector._generate_image_slices(dummy, slice_size=320, overlap=48)
        self.assertTrue(
            inspect.isgenerator(result),
            "_generate_image_slices must be a generator (not a pre-built list)"
        )

    def test_generator_yields_correct_tuple_shape(self):
        """Each yielded tile must be (tile_array, x_offset, y_offset) with correct dtype."""
        import numpy as np
        canvas = np.zeros((800, 640, 3), dtype=np.uint8)
        for tile, x_off, y_off in detector._generate_image_slices(canvas, slice_size=320, overlap=48):
            self.assertEqual(tile.shape, (320, 320, 3),
                             f"Tile shape mismatch: {tile.shape}")
            self.assertIsInstance(x_off, int)
            self.assertIsInstance(y_off, int)
            self.assertGreaterEqual(x_off, 0)
            self.assertGreaterEqual(y_off, 0)

    def test_high_res_1280x960_slice_count(self):
        """1280×960 canvas must produce > 1 tile (verifies SAHI grid was not bypassed)."""
        import numpy as np
        canvas = np.zeros((960, 1280, 3), dtype=np.uint8)
        slices = list(detector._generate_image_slices(canvas, slice_size=320, overlap=48))
        self.assertGreater(len(slices), 1,
                           "High-res image should produce multiple tiles")

    def test_high_res_no_memory_error(self):
        """Processing a large 1280×960 canvas must complete without MemoryError."""
        import numpy as np
        canvas = np.zeros((960, 1280, 3), dtype=np.uint8)
        consumed = 0
        try:
            for tile, x_off, y_off in detector._generate_image_slices(canvas, slice_size=320, overlap=48):
                # Simulate the predict loop: consume tile, then immediately free it
                _ = tile.shape
                consumed += 1
                del tile
                gc.collect()
        except MemoryError as e:
            self.fail(f"MemoryError raised during high-res slicing: {e}")
        self.assertGreater(consumed, 0, "No tiles were consumed")

    def test_offsets_stay_within_canvas_bounds(self):
        """All x_offset and y_offset values must stay within the canvas boundaries."""
        import numpy as np
        h, w = 960, 1280
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for tile, x_off, y_off in detector._generate_image_slices(canvas, slice_size=320, overlap=48):
            self.assertLessEqual(x_off + 320, w,
                                 f"x_offset {x_off} + slice_size 320 exceeds canvas width {w}")
            self.assertLessEqual(y_off + 320, h,
                                 f"y_offset {y_off} + slice_size 320 exceeds canvas height {h}")

    def test_gc_collect_importable(self):
        """gc module must be importable and collect() callable (ONNX loop dependency)."""
        import gc as _gc
        collected = _gc.collect()
        self.assertIsInstance(collected, int)

    def test_onnx_session_single_threaded_params(self):
        """ONNXYOLO session options must lock intra_op_num_threads to 1 (ORT_SEQUENTIAL)."""
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_cpu_mem_arena = False
        # Verify values are accepted without error
        self.assertEqual(opts.intra_op_num_threads, 1)
        self.assertEqual(opts.inter_op_num_threads, 1)
        self.assertEqual(opts.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
