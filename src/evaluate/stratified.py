"""Métricas estratificadas por era (Protocolo de Validação: reportar SEMPRE por era).

Mostra que o desempenho não vem do tempo de calendário: se o C-index se mantém dentro de
cada era, o modelo discrimina por biologia, não pela era. Também é onde se vê que o
horizonte de 10 anos não é avaliável na era 2018-2022.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ERA_LABELS
from src.evaluate.curves import antolini_cindex
from src.evaluate.metrics import harrell_cindex, to_structured, uno_cindex


def metrics_by_era(era, duration, event, risk, surv: pd.DataFrame, y_train, tau: float, min_n: int = 100) -> pd.DataFrame:
    era = np.asarray(era).astype(str)
    dur = np.asarray(duration, dtype=float)
    ev = np.asarray(event)
    risk = np.asarray(risk)

    rows = []
    for e in ERA_LABELS:
        mask = era == e
        if mask.sum() < min_n:
            continue
        cols = np.where(mask)[0]
        try:
            uno = uno_cindex(y_train, to_structured(dur[mask], ev[mask]), risk[mask], tau)
        except Exception:
            uno = float("nan")
        rows.append({
            "era": e,
            "n": int(mask.sum()),
            "harrell": harrell_cindex(dur[mask], ev[mask], risk[mask]),
            "uno": uno,
            "antolini": antolini_cindex(surv.iloc[:, cols], dur[mask], ev[mask]),
        })
    return pd.DataFrame(rows)
