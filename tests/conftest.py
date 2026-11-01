"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tiny_cones():
    """Drei Mini-Kegelprogramme mit bekanntem Optimum, je ein Kegel:
    Orthant: min x mit x >= 1 (s = x - 1 >= 0)  -> 1;  Lorentz: min t mit (t; 3, 4) im Kegel -> 5;  PSD: min x mit [[x, 1],[1, x]] >= 0 -> 1."""
    import kegel_kern as K

    sq2 = K.SQ2
    lp = K.ConeProblem(np.array([1.0]), np.array([[-1.0]]), np.array([-1.0]), (("l", 1),), np.array([2.0]))
    soc = K.ConeProblem(np.array([1.0]), np.array([[-1.0], [0.0], [0.0]]), np.array([0.0, 3.0, 4.0]), (("q", 3),), np.array([6.0]))
    # svec([[x, 1],[1, x]]) = (x, sqrt2 * 1, x); s = b - A x mit A = (-1, 0, -1), b = (0, sqrt2, 0)
    sdp = K.ConeProblem(np.array([1.0]), np.array([[-1.0], [0.0], [-1.0]]), np.array([0.0, sq2, 0.0]), (("s", 2),), np.array([2.0]))
    return {"lp": (lp, 1.0), "soc": (soc, 5.0), "sdp": (sdp, 1.0)}
