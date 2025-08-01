import math

def test_brier_score():
    """Test Brier score calculation."""
    # Example probabilities and outcomes
    probabilities = [0.7, 0.75, 0.77, 0.78, 0.65]
    outcomes = [1, 1, 1, 1, 0]

    # Calculate Brier score
    brier_score = sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / len(probabilities)

    # Assert the score is within an expected range
    assert math.isclose(brier_score, 0.11, rel_tol=1e-2), f"Unexpected Brier score: {brier_score}"
