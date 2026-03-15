"""Tests for ProgressiveDetector Bayesian update logic."""

import pytest
from super_brain.progressive import bayesian_update, ProgressiveDetector


class TestBayesianUpdate:
    """Test the core Bayesian update formula."""

    def test_first_observation_adopts_value(self):
        """With no prior (conf=0), fully adopts observation."""
        val, conf = bayesian_update(0.5, 0.0, 0.80, 0.60)
        assert val == pytest.approx(0.80, abs=0.01)
        assert conf > 0.0

    def test_high_conf_prior_resists_change(self):
        """Strong prior barely moves with weak observation."""
        val, conf = bayesian_update(0.30, 0.80, 0.70, 0.20)
        assert val < 0.45  # still closer to prior (0.30)

    def test_equal_confidence_averages(self):
        """Equal confidence → midpoint."""
        val, conf = bayesian_update(0.20, 0.50, 0.80, 0.50)
        assert val == pytest.approx(0.50, abs=0.01)

    def test_confidence_grows_asymptotically(self):
        """Confidence increases but never exceeds 0.95."""
        conf = 0.0
        for _ in range(50):
            _, conf = bayesian_update(0.5, conf, 0.5, 0.5)
        assert conf <= 0.95
        assert conf > 0.90

    def test_zero_observation_confidence_no_change(self):
        """Zero-confidence observation doesn't move prior."""
        val, conf = bayesian_update(0.30, 0.50, 0.90, 0.0)
        assert val == pytest.approx(0.30, abs=0.01)
        assert conf == pytest.approx(0.50, abs=0.01)


class TestProgressiveDetectorInit:
    """Test ProgressiveDetector initialization and state."""

    def test_init_empty_priors(self):
        """Starts with no priors."""
        pd = ProgressiveDetector.__new__(ProgressiveDetector)
        pd._priors = {}
        pd._history = []
        assert pd.get_profile_dict() == {}
        assert pd.get_history() == []

    def test_set_and_get_prior(self):
        """Can manually set priors for testing."""
        pd = ProgressiveDetector.__new__(ProgressiveDetector)
        pd._priors = {"narcissism": (0.65, 0.40)}
        pd._history = []
        profile = pd.get_profile_dict()
        assert profile["narcissism"]["value"] == pytest.approx(0.65)
        assert profile["narcissism"]["confidence"] == pytest.approx(0.40)
