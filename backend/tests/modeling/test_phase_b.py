import numpy as np
import pytest

from abc_credit.modeling.phase_b import choose_threshold, evaluate, predict_applications


def test_declined_is_positive_and_confusion_matrix_orientation():
    metrics = evaluate([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.2], 0.5)
    assert metrics['confusion_matrix_approved_declined'] == [[1, 1], [1, 1]]
    assert metrics['declined_recall'] == 0.5


def test_threshold_changes_with_business_costs():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.1, 0.2, 0.4, 0.3])
    strict = choose_threshold(y, p, 10, 1)
    loose = choose_threshold(y, p, 1, 10)
    assert strict < loose
    assert np.sum((y == 1) & (p < strict)) == 0


def test_inference_rejects_missing_fields():
    import pandas as pd
    with pytest.raises(ValueError, match='Age'):
        predict_applications({'feature_columns': ['Age']}, pd.DataFrame({'income': [1]}))
