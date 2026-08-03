"""Etapa 1 do RUNBOOK — inclusão/exclusão de coorte.

Aplica, na ordem dos Critérios de Inclusão e Exclusão:
  1. restrição a `Sex == Female` (decisão de escopo, ADR-004);
  2. separação de `Survival months == Unknown` (sem tempo não há alvo).

Diferença central vs. código antigo: o tempo desconhecido **não** é descartado por um
`dropna` silencioso. Ele é 100% óbito (deleção informativa — enviesa a sobrevivência
para cima), então é retirado da coorte principal MAS devolvido à parte, para a análise
de sensibilidade obrigatória (reincluir como duration=0, event=1). Ver
"Armadilha - Deleção informativa de tempo desconhecido" e ADR-004.

Requer que `era_diagnostico` já tenha sido reconstruída (a era vem ANTES de tudo).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import (
    COL_SEX,
    COL_SURVIVAL_MONTHS,
    COL_VITAL_STATUS,
    SEX_KEEP_VALUE,
    SURVIVAL_UNKNOWN_TOKEN,
)
from src.data.era import ERA_COLUMN


@dataclass
class CohortFlow:
    """Contagens do fluxo de coorte — vira a tabela colada em Critérios de Inclusão."""

    n_raw: int
    n_male_removed: int
    n_female: int
    n_unknown_time_total: int  # em toda a base (para conferir os 64 homens)
    n_unknown_time_male: int
    n_unknown_time_female_removed: int
    n_analytic: int
    n_unknown_time_female_dead: int  # invariante: == n_unknown_time_female_removed
    n_duration_zero_analytic: int  # mantidos (ADR-004), só reportados
    era_counts_analytic: dict[str, int] = field(default_factory=dict)
    era_max_followup_analytic: dict[str, int] = field(default_factory=dict)


@dataclass
class CohortResult:
    analytic: pd.DataFrame  # coorte feminina com tempo conhecido (+ era_diagnostico)
    unknown_time: pd.DataFrame  # feminino com tempo desconhecido — só p/ sensibilidade
    flow: CohortFlow


def _era_max_followup(df: pd.DataFrame) -> dict[str, int]:
    months = pd.to_numeric(df[COL_SURVIVAL_MONTHS], errors="coerce")
    tmp = pd.DataFrame({"era": df[ERA_COLUMN].to_numpy(), "months": months.to_numpy()})
    grouped = tmp.groupby("era", observed=True)["months"].max()
    return {era: int(v) for era, v in grouped.items() if pd.notna(v)}


def apply_cohort(df: pd.DataFrame) -> CohortResult:
    """Aplica os filtros de coorte e devolve (analítica, tempo-desconhecido, fluxo)."""
    if ERA_COLUMN not in df.columns:
        raise KeyError(
            f"apply_cohort exige a coluna {ERA_COLUMN!r}. Reconstrua a era ANTES do filtro."
        )

    n_raw = len(df)
    is_male = df[COL_SEX] != SEX_KEEP_VALUE
    is_unknown_time = df[COL_SURVIVAL_MONTHS] == SURVIVAL_UNKNOWN_TOKEN

    n_unknown_total = int(is_unknown_time.sum())
    n_unknown_male = int((is_unknown_time & is_male).sum())

    female = df[~is_male]
    n_female = len(female)

    female_unknown = female[COL_SURVIVAL_MONTHS] == SURVIVAL_UNKNOWN_TOKEN
    analytic = female[~female_unknown].reset_index(drop=True)
    unknown_time = female[female_unknown].reset_index(drop=True)

    n_unknown_female_dead = int((unknown_time[COL_VITAL_STATUS] == "Dead").sum())
    n_duration_zero = int((analytic[COL_SURVIVAL_MONTHS] == "0000").sum())

    flow = CohortFlow(
        n_raw=n_raw,
        n_male_removed=int(is_male.sum()),
        n_female=n_female,
        n_unknown_time_total=n_unknown_total,
        n_unknown_time_male=n_unknown_male,
        n_unknown_time_female_removed=int(female_unknown.sum()),
        n_analytic=len(analytic),
        n_unknown_time_female_dead=n_unknown_female_dead,
        n_duration_zero_analytic=n_duration_zero,
        era_counts_analytic={
            era: int(n) for era, n in analytic[ERA_COLUMN].value_counts().items()
        },
        era_max_followup_analytic=_era_max_followup(analytic),
    )
    return CohortResult(analytic=analytic, unknown_time=unknown_time, flow=flow)


def restrict_to_eras(df: pd.DataFrame, eras: list[str] | None) -> pd.DataFrame:
    """Etapa 3: restringe a coorte às eras selecionadas (ADR-003). `eras=None` devolve a
    base completa (exploração). A era continua sendo estrato/restrição, nunca feature.
    """
    if eras is None:
        return df
    if ERA_COLUMN not in df.columns:
        raise KeyError(f"restrict_to_eras exige a coluna {ERA_COLUMN!r}.")
    return df[df[ERA_COLUMN].isin(eras)].reset_index(drop=True)
