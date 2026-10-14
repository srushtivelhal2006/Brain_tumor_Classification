"""
Integration tests for Flask Web App and REST API endpoints.
"""

import unittest
import io
from PIL import Image
from web_app import app


class TestWebAppAPI(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        """Tests GET /api/health endpoint returns 200 OK and valid metadata."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("classes", data)
        self.assertEqual(len(data["classes"]), 4)

    def test_predict_endpoint_with_sample(self):
        """Tests POST /api/predict using built-in sample identifier."""
        response = self.client.post('/api/predict', data={
            'sample': 'normal_sample.png',
            'gradcam': 'true',
            'enhance': 'false'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predicted_class", data)
        self.assertIn("probabilities", data)
        self.assertIn("gradcam_base64", data)
        self.assertIsNotNone(data["gradcam_base64"])

    def test_predict_endpoint_with_uploaded_file(self):
        """Tests POST /api/predict with multipart image file upload."""
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        response = self.client.post(
            '/api/predict',
            content_type='multipart/form-data',
            data={'image': (img_bytes, 'test_scan.png')}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predicted_class", data)
        self.assertIn("confidence_percent", data)


if __name__ == "__main__":
    unittest.main()
