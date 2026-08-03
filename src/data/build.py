"""Orquestrador do pipeline de dados — a ordem anti-vazamento num só lugar.

Centralizar aqui evita o erro do código antigo, onde o encadeamento estava espalhado
pelos scripts e a era era destruída dentro de `clean()`. Este módulo é a única porta de
entrada para montar a coorte, e cresce etapa a etapa (RUNBOOK), sempre respeitando:

    load(str) -> reconstruir era -> inclusão/exclusão -> [Etapa 2: limpeza SEER]
    -> alvo -> (SPLIT, fora daqui) -> (fit só no treino, fora daqui)

Nenhum `fit` e nenhum split acontecem neste módulo — isso é responsabilidade de
`preprocessing/`, depois do split.

Etapa 1 implementada: coorte + alvo (features ainda BRUTAS; a limpeza é a Etapa 2).
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data.cleaning import clean_features
from src.data.cohort import CohortResult, apply_cohort, restrict_to_eras
from src.data.era import ERA_COLUMN, reconstruct_era
from src.data.loading import load_raw_full
from src.data.target import build_target

_UNSET = object()  # sentinela: distingue "não passado" de "None" (= base completa)


def add_era(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta `era_diagnostico` reconstruída a partir das colunas ainda brutas.
    Passo obrigatório ANTES de qualquer limpeza de `Blank(s)`.
    """
    df = df.copy()
    df[ERA_COLUMN] = reconstruct_era(df)
    return df


def build_cohort(raw: pd.DataFrame | None = None) -> CohortResult:
    """Etapa 1 completa: (carrega o CSV se necessário ->) reconstrói era -> aplica
    coorte -> deriva duration/event na coorte analítica.

    Devolve `CohortResult` com:
      - `analytic`: coorte feminina de tempo conhecido, com `era_diagnostico`,
        `duration`, `event` e as features ainda BRUTAS (limpeza é Etapa 2);
      - `unknown_time`: coorte feminina de tempo desconhecido (bruta), reservada só
        para a análise de sensibilidade (não recebe alvo aqui);
      - `flow`: as contagens do fluxo de coorte.
    """
    if raw is None:
        raw = load_raw_full()

    result = apply_cohort(add_era(raw))
    result.analytic = build_target(result.analytic)
    return result


def build_analytic_frame(raw: pd.DataFrame | None = None, era_restriction=_UNSET) -> CohortResult:
    """Etapas 1+2+3: coorte + limpeza SEER + restrição de era.

    Devolve `CohortResult` com `analytic` já limpo (features derivadas conforme Contrato
    de Dados) e, se `era_restriction` for uma lista de eras, filtrado a elas (Etapa 3,
    ADR-003). `era_restriction=None` -> base completa; omitido -> usa `config.ERA_RESTRICTION`.
    `unknown_time` fica bruto (sensibilidade). Continua SEM split e SEM `fit`.
    """
    if era_restriction is _UNSET:
        era_restriction = config.ERA_RESTRICTION

    result = build_cohort(raw)
    result.analytic = clean_features(result.analytic)
    result.analytic = restrict_to_eras(result.analytic, era_restriction)
    return result
