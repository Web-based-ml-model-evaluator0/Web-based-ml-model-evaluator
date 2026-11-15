
import unittest

from src.services.model_service import ModelService


class TestModelAvailability(unittest.TestCase):

    def setUp(self):
        # Reset all models to enabled before each test.
        for name in ModelService.get_model_availability():
            ModelService.set_model_availability(name, True)

    def test_models_enabled_by_default(self):
        availability = ModelService.get_model_availability()
        self.assertTrue(all(availability.values()))

    def test_disable_model(self):
        ModelService.set_model_availability(
            "Logistic Regression", False
        )

        self.assertFalse(
            ModelService.is_model_available("Logistic Regression")
        )
        self.assertNotIn(
            "Logistic Regression",
            ModelService.get_supported_models("classification"),
        )

    def test_disabled_model_cannot_be_created(self):
        ModelService.set_model_availability(
            "Logistic Regression", False
        )

        with self.assertRaisesRegex(ValueError, "disabled"):
            ModelService.create_model_instance("Logistic Regression")

    def test_enable_model_again(self):
        ModelService.set_model_availability(
            "Logistic Regression", False
        )
        ModelService.set_model_availability(
            "Logistic Regression", True
        )

        self.assertIn(
            "Logistic Regression",
            ModelService.get_supported_models("classification"),
        )

    def test_unknown_model_cannot_be_managed(self):
        with self.assertRaises(ValueError):
            ModelService.set_model_availability(
                "Unknown Model", False
            )


if __name__ == "__main__":
    unittest.main()
