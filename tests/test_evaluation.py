"""Testes das funções de avaliação (Etapa 8) em dados sintéticos — não tocam nenhum dado
real (muito menos o conjunto de teste)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ERA_LABELS
from src.evaluate.bootstrap import bootstrap_ci
from src.evaluate.calibration import calibration_by_decile, survival_at
from src.evaluate.comparison import ModelPrediction, model_metrics
from src.evaluate.curves import brier_at_times
from src.evaluate.metrics import to_structured
from src.evaluate.sensitivity import unknown_time_sensitivity
from src.evaluate.stratified import metrics_by_era


def _synth(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    times = np.arange(0, 120, 5).astype(float)
    risk = rng.normal(size=n)
    rate = np.exp(0.3 * risk) * 0.01
    surv = pd.DataFrame(np.exp(-np.outer(times, rate)), index=times)  # decrescente por indivíduo
    dur = rng.integers(0, 115, n).astype(float)
    ev = rng.integers(0, 2, n)
    return risk, surv, dur, ev


def test_bootstrap_ci_ordena():
    lo, hi = bootstrap_ci(lambda idx: float(idx.mean()), 100, n_boot=50)
    assert lo <= hi


def test_brier_omite_horizonte_alem_do_maximo():
    _, surv, dur, ev = _synth()
    b = brier_at_times(surv, dur, ev, [12, 1000])
    assert 12 in b and 1000 not in b  # 1000 > max(dur) -> omitido


def test_model_metrics_chaves_e_faixa():
    risk, surv, dur, ev = _synth()
    row = model_metrics(ModelPrediction("m", risk, surv), dur, ev, to_structured(dur, ev), tau=90.0, n_boot=30)
    for k in ["harrell", "uno", "antolini", "ibs", "harrell_lo", "harrell_hi", "brier_60"]:
        assert k in row
    assert 0.0 <= row["harrell"] <= 1.0
    assert row["harrell_lo"] <= row["harrell_hi"]


def test_calibration_retorna_decis_em_faixa():
    _, surv, dur, ev = _synth(n=600)
    cal = calibration_by_decile(surv, dur, ev, horizon=60, n_bins=5)
    assert len(cal) <= 5
    assert cal["previsto"].between(0, 1).all()
    assert cal["observado"].between(0, 1).all()


def test_metrics_by_era():
    risk, surv, dur, ev = _synth(n=800)
    era = np.random.default_rng(1).choice(ERA_LABELS, size=800)
    m = metrics_by_era(era, dur, ev, risk, surv, to_structured(dur, ev), tau=90.0, min_n=10)
    assert {"era", "n", "harrell", "uno", "antolini"} <= set(m.columns)
    assert len(m) >= 1


def test_sensitivity_delta_correto():
    def frac_eventos(d, e, r):
        return float(np.mean(e))
    out = unknown_time_sensitivity([1, 2, 3], [0, 1, 0], [0.1, 0.2, 0.3], [0.5, 0.6], metric_fn=frac_eventos)
    assert abs(out["base"] - 1 / 3) < 1e-9          # 1 evento em 3
    assert abs(out["with_unknown"] - 3 / 5) < 1e-9  # + 2 óbitos (t=0) -> 3 em 5
    assert out["n_unknown"] == 2


def test_survival_at_funcao_escada():
    times = np.array([0, 10, 20, 30], dtype=float)
    surv = pd.DataFrame({0: [1.0, 0.8, 0.6, 0.4]}, index=times)
    assert survival_at(surv, 25)[0] == 0.6  # maior tempo <= 25 é 20
