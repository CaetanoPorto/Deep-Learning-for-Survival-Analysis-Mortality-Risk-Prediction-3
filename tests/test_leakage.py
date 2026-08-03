"""Testes das checagens anti-vazamento (Etapa 5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.checks.leakage import assert_no_target_leakage, era_predictability, find_target_leakage


def test_find_target_leakage():
    assert find_target_leakage(["num__age_mid", "desc__race_White"]) == []
    assert find_target_leakage(["num__duration"]) == ["num__duration"]
    assert find_target_leakage(["oh__event_1", "x__vital_status"]) == ["oh__event_1", "x__vital_status"]


def test_assert_no_target_leakage():
    assert_no_target_leakage(["num__age_mid", "her2__her2_status_Positive"])  # ok, não levanta
    with pytest.raises(AssertionError):
        assert_no_target_leakage(["num__duration"])


def test_probe_detecta_era_quando_ela_vaza():
    # Se a própria era estiver entre as features, o probe recupera ~perfeito.
    rng = np.random.default_rng(0)
    n = 3000
    era = rng.integers(0, 3, n)
    X = np.column_stack([era.astype(float), rng.normal(size=n)])
    r = era_predictability(X[:2000], era[:2000], X[2000:], era[2000:])
    assert r.accuracy > 0.95
    assert r.macro_auc > 0.95


def test_probe_nao_ve_era_no_ruido():
    # Features independentes da era -> probe fica no baseline, AUC ~0,5.
    rng = np.random.default_rng(1)
    n = 3000
    era = rng.integers(0, 3, n)
    X = rng.normal(size=(n, 5))
    r = era_predictability(X[:2000], era[:2000], X[2000:], era[2000:])
    assert r.lift < 0.05
    assert r.macro_auc < 0.60
