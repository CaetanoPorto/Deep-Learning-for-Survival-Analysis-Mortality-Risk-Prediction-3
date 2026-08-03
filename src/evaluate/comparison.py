"""Tabela comparativa entre modelos (Protocolo de Validação): C-Harrell, C-Uno,
C-Antolini, IBS e Brier@{12,36,60,120}, com IC bootstrap nos C-index.

Todos os modelos são avaliados no MESMO conjunto (mesma amostra fixa — ADR-010). Os
C-index (comparação principal) levam IC por bootstrap; IBS/Brier entram como estimativa
pontual (bootstrap da curva inteira 200× é caro; a calibração cobre o lado da calibração).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import EVAL_HORIZONS, N_BOOTSTRAP
from src.evaluate.bootstrap import bootstrap_ci
from src.evaluate.curves import antolini_cindex, brier_at_times, integrated_brier_score
from src.evaluate.metrics import harrell_cindex, to_structured, uno_cindex


@dataclass
class ModelPrediction:
    name: str
    risk: np.ndarray  # escore de risco no conjunto de avaliação (maior = pior)
    surv: pd.DataFrame  # curva S(t) no conjunto de avaliação (índice = tempos)


def model_metrics(
    pred: ModelPrediction, duration, event, y_train, tau: float,
    n_boot: int = N_BOOTSTRAP, seed: int = 42,
) -> dict:
    dur = np.asarray(duration)
    ev = np.asarray(event)
    risk = np.asarray(pred.risk)
    y_eval = to_structured(dur, ev)
    n = len(dur)

    row = {
        "model": pred.name,
        "harrell": harrell_cindex(dur, ev, risk),
        "uno": uno_cindex(y_train, y_eval, risk, tau),
        "antolini": antolini_cindex(pred.surv, dur, ev),
        "ibs": integrated_brier_score(pred.surv, dur, ev),
    }
    h_lo, h_hi = bootstrap_ci(lambda idx: harrell_cindex(dur[idx], ev[idx], risk[idx]), n, n_boot, seed)
    u_lo, u_hi = bootstrap_ci(
        lambda idx: uno_cindex(y_train, to_structured(dur[idx], ev[idx]), risk[idx], tau), n, n_boot, seed
    )
    row.update(harrell_lo=h_lo, harrell_hi=h_hi, uno_lo=u_lo, uno_hi=u_hi)

    briers = brier_at_times(pred.surv, dur, ev, EVAL_HORIZONS)
    for t in EVAL_HORIZONS:
        row[f"brier_{t}"] = briers.get(t, float("nan"))
    return row


def comparison_table(
    preds: list[ModelPrediction], duration, event, y_train, tau: float,
    n_boot: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Uma linha por modelo. Ordenada por C-Uno decrescente (métrica principal)."""
    rows = [model_metrics(p, duration, event, y_train, tau, n_boot) for p in preds]
    return pd.DataFrame(rows).sort_values("uno", ascending=False).reset_index(drop=True)
