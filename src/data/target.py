"""Construção do alvo de sobrevivência (duration, event).

Endpoint = sobrevivência global (all-cause): `event=1` significa "morreu por qualquer
causa", não "morreu de câncer de mama" (ADR-001). `event` é `int` (nunca `bool`) para
a arquitetura ficar agnóstica ao número de causas — a migração a riscos competitivos
troca {0,1} por {0,1,2} sem reescrever nada (ADR-002).

Diferença vs. código antigo: NÃO há `dropna` aqui. As linhas sem tempo já foram
separadas em `cohort.apply_cohort` (deleção informativa tratada explicitamente). Este
módulo recebe uma coorte de tempo conhecido e só deriva os rótulos.
"""

from __future__ import annotations

import pandas as pd

from src.config import COL_SURVIVAL_MONTHS, COL_VITAL_STATUS, EVENT_MAP

DURATION_COL = "duration"
EVENT_COL = "event"


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva `duration` (meses, float) e `event` (0/1, int) e remove as colunas brutas
    de desfecho. Assume tempo conhecido em todas as linhas (coorte já filtrada).
    """
    df = df.copy()

    duration = pd.to_numeric(df[COL_SURVIVAL_MONTHS], errors="coerce")
    if duration.isna().any():
        n = int(duration.isna().sum())
        raise ValueError(
            f"{n} linhas com `{COL_SURVIVAL_MONTHS}` não numérico chegaram a build_target. "
            "O tempo desconhecido deveria ter sido separado em apply_cohort."
        )

    event = df[COL_VITAL_STATUS].map(EVENT_MAP)
    if event.isna().any():
        bad = df.loc[event.isna(), COL_VITAL_STATUS].unique().tolist()
        raise ValueError(
            f"`{COL_VITAL_STATUS}` com valores fora de {EVENT_MAP}: {bad}"
        )

    df[DURATION_COL] = duration.astype(float)
    df[EVENT_COL] = event.astype(int)
    return df.drop(columns=[COL_SURVIVAL_MONTHS, COL_VITAL_STATUS])
