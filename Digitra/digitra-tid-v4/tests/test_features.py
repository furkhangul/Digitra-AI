import numpy as np

from src.features import N_FEATURES, build_two_hand_features


def hand(offset: float = 0.0) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.linspace(offset, offset + 0.2, 21)
    points[:, 1] = np.linspace(0.1, 0.4, 21)
    return points


def test_feature_shape_one_and_two_hands():
    assert build_two_hand_features([hand()]).shape == (N_FEATURES,)
    assert build_two_hand_features([hand(0.4), hand(0.0)]).shape == (N_FEATURES,)


def test_hand_order_does_not_change_features():
    a, b = hand(0.0), hand(0.4)
    np.testing.assert_allclose(
        build_two_hand_features([a, b]),
        build_two_hand_features([b, a]),
    )
