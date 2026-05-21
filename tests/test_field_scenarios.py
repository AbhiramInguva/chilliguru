import sys
import os
import unittest
import json
import io
from unittest.mock import MagicMock, patch

# Ensure the workspace directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app

class TestFieldScenarios(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.app.config["TESTING"] = True
        app.limiter.enabled = False
        self.client = app.app.test_client()
        # Mock synthetic checker so it doesn't block standard requests
        self.synthetic_patcher = patch("app.detector.check_synthetic", return_value=0.0)
        self.mock_check_synthetic = self.synthetic_patcher.start()

    def tearDown(self):
        self.synthetic_patcher.stop()

    def test_payload_exceeding_size_cap_blocks(self):
        """
        Giant files should trigger a 413 payload error.
        """
        # app.MAX_IMAGE_BYTES is 4 * 1024 * 1024 (4 MB)
        oversized_data = io.BytesIO(b"\xff\xd8\xff" + b"\x00" * (app.MAX_IMAGE_BYTES + 100))
        data = {
            "image": (oversized_data, "large.jpg"),
            "message": "test large image size check"
        }
        response = self.client.post("/detect", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 413)
        res_json = json.loads(response.data)
        self.assertEqual(res_json["error"], "Payload too large")
        self.assertIn("request_id", res_json)

    @patch("app.get_hf_client")
    @patch("app._cb_is_open")
    @patch("app.detector.detect_from_memory")
    def test_out_of_domain_image_rejection(self, mock_detect_local, mock_cb, mock_get_client):
        """
        Animals / non-chilli images should trigger a 422 structural validation block.
        """
        mock_cb.return_value = False
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Simulating HF Space returning "non_chilli" out-of-domain response
        mock_client.predict.return_value = {
            "success": True,
            "low_confidence": True,
            "is_low_confidence": True,
            "top_detection": {
                "label": "non_chilli",
                "telugu": "",
                "confidence": 95.0,
                "type": "ood",
                "raw_label": "non_chilli"
            },
            "all_detections": [
                {
                    "label": "non_chilli",
                    "telugu": "",
                    "confidence": 95.0,
                    "type": "ood",
                    "raw_label": "non_chilli"
                }
            ]
        }
        
        data = {
            "image": (io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"), "ood_animal.jpg"),
            "message": "test out of domain animal"
        }
        
        response = self.client.post("/detect", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 422)
        res_json = json.loads(response.data)
        self.assertEqual(res_json["error"], "Cannot identify crop. Please upload a clear photo of a chilli plant.")
        self.assertIn("request_id", res_json)
        
        # Verify local fallback detector was NOT called
        mock_detect_local.assert_not_called()

    @patch("app.get_hf_client")
    @patch("app._cb_is_open")
    @patch("app.detector.detect_from_memory")
    def test_macro_background_guardrail_suppression(self, mock_detect_local, mock_cb, mock_get_client):
        """
        Macro background overrides (crop_anomaly) should return a clean 200 response.
        """
        mock_cb.return_value = False
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock HF space to return None/fail to force local fallback
        mock_client.predict.side_effect = Exception("HF offline")
        
        # Mock local fallback to return the overridden Crop Anomaly response with confidence >= 80% (e.g. 85.0%)
        mock_detect_local.return_value = {
            "success": True,
            "low_confidence": True,
            "top_detection": {
                "label": "Crop Anomaly [పంట అసాధారణత]",
                "raw_label": "crop_anomaly",
                "type": "pest",
                "confidence": 85.0,
                "telugu": "పంట అసాధారణత"
            },
            "all_detections": [
                {
                    "label": "Crop Anomaly [పంట అసాధారణత]",
                    "raw_label": "crop_anomaly",
                    "type": "pest",
                    "confidence": 85.0,
                    "telugu": "పంట అసాధారణత"
                }
            ],
            "phase": 1
        }
        
        # Mock Groq client to avoid network calls during SSE stream generation
        with patch("app.get_client") as mock_groq:
            mock_comp = MagicMock()
            mock_groq.return_value.chat.completions.create.return_value = [mock_comp]
            mock_comp.choices = [MagicMock()]
            mock_comp.choices[0].delta.content = "Advisory for crop anomaly"
            
            data = {
                "image": (io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"), "macro.jpg"),
                "message": "macro image test"
            }
            
            response = self.client.post("/detect", data=data, content_type="multipart/form-data")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "text/event-stream")
            _ = list(response.response)
            
            # Verify local detector was called
            mock_detect_local.assert_called_once()

    @patch("app.get_hf_client")
    @patch("app._cb_is_open")
    @patch("app.detector.detect_from_memory")
    def test_mealybug_class_4_mapping(self, mock_detect_local, mock_cb, mock_get_client):
        """
        Class 4 / Mealybug predictions should be correctly mapped to "Mealybugs [పిండి పురుగు]".
        """
        mock_cb.return_value = False
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Simulating HF Space returning Class 4 / "Mealybugs" or "Pest-Phenacoccus solenopsis (Mealybug)" response
        mock_client.predict.return_value = {
            "success": True,
            "low_confidence": False,
            "is_low_confidence": False,
            "top_detection": {
                "label": "Pest-Phenacoccus solenopsis (Mealybug)",
                "telugu": "తెల్ల దూది పురుగు",
                "confidence": 92.0,
                "type": "pest",
                "class_id": 4,
                "raw_label": "Pest-Phenacoccus solenopsis (Mealybug)"
            },
            "all_detections": [
                {
                    "label": "Pest-Phenacoccus solenopsis (Mealybug)",
                    "telugu": "తెల్ల దూది పురుగు",
                    "confidence": 92.0,
                    "type": "pest",
                    "class_id": 4,
                    "raw_label": "Pest-Phenacoccus solenopsis (Mealybug)"
                }
            ]
        }
        
        # Mock Groq client to avoid network calls during SSE stream generation
        with patch("app.get_client") as mock_groq:
            mock_comp = MagicMock()
            mock_groq.return_value.chat.completions.create.return_value = [mock_comp]
            mock_comp.choices = [MagicMock()]
            mock_comp.choices[0].delta.content = "Advisory for mealybugs"
            
            data = {
                "image": (io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"), "mealybug.jpg"),
                "message": "test mealybug translation"
            }
            
            response = self.client.post("/detect", data=data, content_type="multipart/form-data")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "text/event-stream")
            
            # Read the SSE response stream to make sure we parse the JSON event with "top_detection"
            lines = list(response.response)
            # Find the line starting with "data: " that contains the JSON payload
            detection_data = None
            for line in lines:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: "):
                    payload = decoded[6:]
                    try:
                        parsed = json.loads(payload)
                        if parsed.get("type") == "meta" and "detection" in parsed:
                            detection_data = parsed.get("detection")
                            break
                    except json.JSONDecodeError:
                        continue
            
            self.assertIsNotNone(detection_data)
            self.assertEqual(detection_data["label"], "Mealybugs [పిండి పురుగు]")
            self.assertEqual(detection_data["telugu"], "పిండి పురుగు")
            self.assertEqual(detection_data["raw_label"], "Mealybugs")

if __name__ == "__main__":
    unittest.main()

