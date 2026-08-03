"""Análise de sensibilidade do tempo desconhecido (Armadilha - Deleção informativa / ADR-004).

Os 5.988 registros de `Survival months = Unknown` (100% óbitos) foram removidos da coorte;
descartá-los enviesa a sobrevivência para cima. Aqui os reincluímos como óbito em t=0
(limite pessimista) e medimos o delta nas métricas. Delta desprezível -> exclusão
justificada empiricamente; delta grande -> vira limitação de peso.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.evaluate.metrics import harrell_cindex


def unknown_time_sensitivity(
    duration, event, risk, unknown_risk, metric_fn: Callable = harrell_cindex,
) -> dict:
    """Compara a métrica sem e com os tempos desconhecidos reincluídos (t=0, evento=1).

    `unknown_risk` = escore de risco do MESMO modelo aplicado à coorte de tempo
    desconhecido (features pré-processadas com o transformer já ajustado no treino).
    """
    dur = np.asarray(duration, dtype=float)
    ev = np.asarray(event)
    risk = np.asarray(risk)
    ur = np.asarray(unknown_risk)

    base = metric_fn(dur, ev, risk)
    dur2 = np.concatenate([dur, np.zeros(len(ur))])
    ev2 = np.concatenate([ev, np.ones(len(ur), dtype=int)])
    risk2 = np.concatenate([risk, ur])
    with_unknown = metric_fn(dur2, ev2, risk2)

    return {
        "base": float(base),
        "with_unknown": float(with_unknown),
        "delta": float(with_unknown - base),
        "n_unknown": int(len(ur)),
    }
