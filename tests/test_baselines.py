"""Smoke tests dos baselines: cada modelo aprende um sinal sintético (C-index > 0,6) e
os splines expandem as contínuas."""

from __future__ import annotations

import numpy as np

from src.config import GBSConfig, RSFConfig
from src.evaluate.metrics import harrell_cindex, to_structured
from src.models.boosting import fit_gbs, gbs_risk
from src.models.cox import build_cox_frame, build_spline_features, cox_risk, fit_cox
from src.models.forest import fit_rsf, rsf_risk


def _synth(n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    # tempo ~ exponencial com hazard exp(0,9 * f0): f0 alto -> morre antes
    dur = np.clip(-np.log(rng.uniform(size=n)) / np.exp(0.9 * X[:, 0]), 0.05, 50).astype(float)
    ev = np.ones(n, dtype=int)
    return X, dur, ev


def test_cox_aprende_sinal():
    X, dur, ev = _synth()
    frame = build_cox_frame(X, [f"num__f{i}" for i in range(6)], dur, ev)
    model = fit_cox(frame)
    assert harrell_cindex(dur, ev, cox_risk(model, frame)) > 0.6


def test_rsf_aprende_sinal():
    X, dur, ev = _synth()
    model = fit_rsf(X, to_structured(dur, ev), RSFConfig(n_estimators=40, min_samples_leaf=10))
    assert harrell_cindex(dur, ev, rsf_risk(model, X)) > 0.6


def test_gbs_aprende_sinal():
    X, dur, ev = _synth()
    model = fit_gbs(X, to_structured(dur, ev), GBSConfig(n_estimators=60))
    assert harrell_cindex(dur, ev, gbs_risk(model, X)) > 0.6


def test_splines_expandem_continuas():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 10)).astype(np.float32)
    names = ["num__a", "num__b"] + [f"nom__c{i}" for i in range(8)]
    Xtr, Xva, Xte, nm = build_spline_features(X[:40], X[40:50], X[50:], names)
    assert Xtr.shape[1] == len(nm)
    assert Xtr.shape[1] > 10  # 2 contínuas viraram várias colunas de spline
    assert Xva.shape[1] == Xtr.shape[1] and Xte.shape[1] == Xtr.shape[1]
    assert nm[-8:] == [f"nom__c{i}" for i in range(8)]  # as não-contínuas mantidas ao final
