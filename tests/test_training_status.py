"""
Unit test for Issue #11 (ML-F-011 Training Progress and Status, TC-TRAIN-02).
Tests training progress tracking, state transitions, callbacks,
and failure status recording during long-running operations.
"""

import unittest
import numpy as np

from src.services.model_service import (
    ModelService,
    ModelTrainingError,
    TrainingStatus,
    TrainingStatusTracker,
)


class TestTrainingStatus(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(40, 3)
        self.y = np.array([0] * 20 + [1] * 20)

    def test_tracker_initial_state(self):
        """TC-TRAIN-02 Step 1: Tracker initializes in PENDING state with 0% progress."""
        tracker = TrainingStatusTracker()
        self.assertEqual(tracker.state, TrainingStatus.PENDING)
        self.assertEqual(tracker.progress_pct, 0.0)
        self.assertFalse(tracker.is_completed())
        self.assertFalse(tracker.is_failed())
        self.assertEqual(len(tracker.history), 0)

    def test_tracker_state_transitions_and_history(self):
        """TC-TRAIN-02 Step 2: Tracker records transitions and maintains event history."""
        tracker = TrainingStatusTracker()
        tracker.update(TrainingStatus.STARTED, 0.2, "Initializing...")
        tracker.update(TrainingStatus.TRAINING, 0.6, "Fitting estimator...")
        tracker.update(TrainingStatus.COMPLETED, 1.0, "Completed!")

        self.assertEqual(tracker.state, TrainingStatus.COMPLETED)
        self.assertTrue(tracker.is_completed())
        self.assertFalse(tracker.is_failed())
        self.assertEqual(len(tracker.history), 3)
        self.assertEqual(tracker.history[0]["state"], TrainingStatus.STARTED)
        self.assertEqual(tracker.history[1]["state"], TrainingStatus.TRAINING)
        self.assertEqual(tracker.history[2]["state"], TrainingStatus.COMPLETED)

    def test_tracker_callback_execution(self):
        """TC-TRAIN-02 Step 3: Callback receives state, progress percentage, and payload."""
        events_received = []

        def callback(state, progress, event):
            events_received.append((state, progress, event["message"]))

        tracker = TrainingStatusTracker(callback=callback)
        tracker.update(TrainingStatus.STARTED, 0.15, "Step 1")
        tracker.update(TrainingStatus.COMPLETED, 1.0, "Step 2")

        self.assertEqual(len(events_received), 2)
        self.assertEqual(events_received[0], (TrainingStatus.STARTED, 0.15, "Step 1"))
        self.assertEqual(events_received[1], (TrainingStatus.COMPLETED, 1.0, "Step 2"))

    def test_model_training_updates_tracker_on_success(self):
        """TC-TRAIN-02 Step 4: ModelService.train_model updates tracker through lifecycle to COMPLETED."""
        tracker = TrainingStatusTracker()
        result = ModelService.train_model(
            model_name="Decision Tree Classifier",
            X_train=self.X,
            y_train=self.y,
            status_tracker=tracker,
        )
        self.assertTrue(tracker.is_completed())
        self.assertEqual(tracker.progress_pct, 1.0)
        states = [h["state"] for h in tracker.history]
        self.assertIn(TrainingStatus.STARTED, states)
        self.assertIn(TrainingStatus.TRAINING, states)
        self.assertIn(TrainingStatus.COMPLETED, states)
        self.assertIn("training_time", tracker.history[-1]["metadata"])

    def test_model_training_updates_tracker_on_failure(self):
        """TC-TRAIN-02 Step 5: Failed training updates tracker to FAILED state and stores message."""
        tracker = TrainingStatusTracker()
        with self.assertRaises(ModelTrainingError):
            ModelService.train_model(
                model_name="Decision Tree Classifier",
                X_train=[],
                y_train=self.y,
                status_tracker=tracker,
            )
        self.assertTrue(tracker.is_failed())
        self.assertEqual(tracker.state, TrainingStatus.FAILED)
        self.assertIsNotNone(tracker.error_message)
        self.assertIn("empty", tracker.error_message.lower())


if __name__ == "__main__":
    unittest.main()
