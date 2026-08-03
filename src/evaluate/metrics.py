"""Métricas de sobrevivência — discriminação.

C-index de Harrell (lifelines) para comparabilidade com a literatura, e C-index de Uno
(scikit-survival, IPCW) como métrica principal de discriminação: com 67% de censura, o
Harrell é enviesado e o Uno corrige por ponderação da censura (Protocolo de Validação).

Convenção de `risk`: MAIOR valor = MAIOR risco (evento mais cedo) — é o que
`predict_partial_hazard` do Cox e o `.predict` de RSF/GBS devolvem.
"""

from __future__ import annotations

import numpy as np
from lifelines.utils import concordance_index as _lifelines_ci
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv


def to_structured(duration, event) -> np.ndarray:
    """Array estruturado do scikit-survival: (event: bool, time: float)."""
    return Surv.from_arrays(
        event=np.asarray(event).astype(bool),
        time=np.asarray(duration).astype(float),
    )


def harrell_cindex(duration, event, risk) -> float:
    """C-index de Harrell. `lifelines` espera que MAIOR score = sobrevida MAIOR, então
    invertemos o sinal do risco aqui dentro (quem chama passa risco: maior = pior).
    """
    return float(_lifelines_ci(np.asarray(duration), -np.asarray(risk), np.asarray(event)))


def uno_cindex(y_train: np.ndarray, y_eval: np.ndarray, risk_eval, tau: float | None = None) -> float:
    """C-index de Uno (IPCW). `y_train` estima a distribuição de censura; `tau` trunca a
    cauda para estabilidade. Devolve só o índice (o scikit-survival retorna uma tupla).
    """
    result = concordance_index_ipcw(y_train, y_eval, np.asarray(risk_eval), tau=tau)
    return float(result[0])
