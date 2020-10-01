# tests/test_endpoints.py

from ..app import app
from flask import url_for
import unittest


class FlaskTodosTest(unittest.TestCase):

    def setUp(self):
        """Set up test application client"""
        self.app = app.test_client() 
        self.app.testing = True

    def tearDown(self):
        """Clear DB after running tests"""

    def test_home_status_code(self):
        """Assert that user successfully lands on homepage"""
        result = self.app.get('/')
        self.assertEqual(result.status_code, 200)
