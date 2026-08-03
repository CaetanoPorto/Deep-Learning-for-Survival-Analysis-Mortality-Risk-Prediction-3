"""Métricas baseadas na curva de sobrevivência inteira S(t): C-index de Antolini
(tempo-dependente, correto para o DeepHit, cujo ranking muda no tempo) e Brier/IBS.

Isolado de `metrics.py` porque depende do pycox (só necessário nos modelos profundos e
na avaliação final), e porque o pycox 0.2.3 precisa de dois shims de compatibilidade com
pandas 2 / scipy — sem eles qualquer chamada a `EvalSurv` quebra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.integrate

# pycox 0.2.3 usa duas APIs que pandas/scipy já removeram:
# 1) `pd.Series.is_monotonic` virou `is_monotonic_increasing` (pandas 2.0).
if not hasattr(pd.Series, "is_monotonic"):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)
# 2) `scipy.integrate.simps` virou `simpson` (scipy 1.14). Aqui (1.13.1) ainda existe,
#    mas o shim protege contra upgrade.
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

from pycox.evaluation import EvalSurv  # noqa: E402 (depois dos shims)


def antolini_cindex(surv_df: pd.DataFrame, duration, event) -> float:
    """C-index de Antolini (tempo-dependente). `censor_surv=None`: o método
    `adj_antolini` já lida com censura na definição dos pares comparáveis.
    """
    ev = EvalSurv(surv_df, np.asarray(duration), np.asarray(event), censor_surv=None)
    return float(ev.concordance_td("adj_antolini"))


def integrated_brier_score(surv_df: pd.DataFrame, duration, event, n_points: int = 100) -> float:
    """IBS (Brier integrado), ponderado por IPCW (censura estimada por KM no fold). A
    grade vai de 0 ao maior tempo observado no fold — extrapolar tornaria o Brier não
    confiável.
    """
    duration = np.asarray(duration)
    ev = EvalSurv(surv_df, duration, np.asarray(event), censor_surv="km")
    time_grid = np.linspace(0, duration.max(), n_points)
    return float(ev.integrated_brier_score(time_grid))


def brier_at_times(surv_df: pd.DataFrame, duration, event, times) -> dict[int, float]:
    """Brier score em horizontes específicos (ex.: 12/36/60/120 meses). Horizontes além
    do maior tempo observado no fold são omitidos (não são avaliáveis — Protocolo de
    Validação: o horizonte de 10 anos não existe na era 2018-2022).
    """
    duration = np.asarray(duration)
    ev = EvalSurv(surv_df, duration, np.asarray(event), censor_surv="km")
    tmax = duration.max()
    valid = np.array([t for t in times if t <= tmax], dtype=float)
    out: dict[int, float] = {}
    if len(valid):
        scores = ev.brier_score(valid)
        for t in valid:
            out[int(t)] = float(scores.loc[t])
    return out
