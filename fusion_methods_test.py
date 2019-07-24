import numpy as np
import pytest
from ensamble.fusion_methods import *


def consistent_decisions():
    return [np.array([[0, 1, 0], [1, 0, 0]]),
            np.array([[0, 1, 0], [1, 0, 0]])]


def test_avrg_fusion_returns_array_with_valid_shape():
    decisions = consistent_decisions()
    avrg = avrg_fusion(decisions)
    assert np.isclose(avrg[0], [0, 1, 0]).all()
    assert np.isclose(avrg[1], [1, 0, 0]).all()
