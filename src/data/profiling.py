"""Etapa 0 do RUNBOOK — perfil empírico da base.

Varre o CSV inteiro em streaming e conta tudo o que o "Perfil Empírico da Base" e as
invariantes do "Contrato de Dados" exigem. Nenhum número aqui é estimado: são contagens
exatas. O resultado é comparado com os valores medidos gravados no config; divergência
significa que o CSV mudou -> a Etapa 0 falha e o pipeline PARA (RUNBOOK Etapa 0).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    COL_GRADE_2018,
    COL_GRADE_THRU_2017,
    COL_NODES_EXAMINED,
    COL_NODES_POSITIVE,
    COL_SEX,
    COL_SURVIVAL_MONTHS,
    COL_VITAL_STATUS,
    BLANK_TOKEN,
    RAW_CSV_PATH,
    SURVIVAL_UNKNOWN_TOKEN,
)
from src.data.era import reconstruct_era
from src.data.loading import iter_raw_chunks

SURVIVAL_ZERO_TOKEN = "0000"  # `Survival months` é zero-padded de 4 dígitos


@dataclass
class ProfileReport:
    """Todos os números medidos numa passada pelo CSV."""

    n_rows: int = 0
    vital_status_counts: Counter = field(default_factory=Counter)
    sex_counts: Counter = field(default_factory=Counter)
    era_counts: Counter = field(default_factory=Counter)
    era_max_followup: dict[str, int] = field(default_factory=dict)
    survival_unknown: int = 0
    survival_unknown_and_dead: int = 0  # invariante: deve igualar survival_unknown
    survival_zero: int = 0
    nodes_positive_98: int = 0
    nodes_examined_00: int = 0
    nodes_98_and_00: int = 0  # invariante: deve igualar os dois acima
    grade_blank_counts: Counter = field(default_factory=Counter)


def _accumulate(report: ProfileReport, chunk: pd.DataFrame) -> None:
    report.n_rows += len(chunk)

    report.vital_status_counts.update(chunk[COL_VITAL_STATUS].value_counts().to_dict())
    report.sex_counts.update(chunk[COL_SEX].value_counts().to_dict())

    sm = chunk[COL_SURVIVAL_MONTHS]
    vital = chunk[COL_VITAL_STATUS]
    is_unknown = sm == SURVIVAL_UNKNOWN_TOKEN
    report.survival_unknown += int(is_unknown.sum())
    report.survival_unknown_and_dead += int((is_unknown & (vital == "Dead")).sum())
    report.survival_zero += int((sm == SURVIVAL_ZERO_TOKEN).sum())

    # Invariante dos linfonodos: nodes_positive == "98" <=> nodes_examined == "00".
    pos98 = chunk[COL_NODES_POSITIVE] == "98"
    exam00 = chunk[COL_NODES_EXAMINED] == "00"
    report.nodes_positive_98 += int(pos98.sum())
    report.nodes_examined_00 += int(exam00.sum())
    report.nodes_98_and_00 += int((pos98 & exam00).sum())

    # Complementaridade do grau: soma dos Blank(s) das duas colunas deve dar N.
    for col in (COL_GRADE_THRU_2017, COL_GRADE_2018):
        report.grade_blank_counts[col] += int((chunk[col] == BLANK_TOKEN).sum())

    # Era reconstruída (a partir das colunas ainda brutas) + follow-up máximo por era.
    era = reconstruct_era(chunk)
    report.era_counts.update(era.value_counts().to_dict())

    months = pd.to_numeric(sm.where(~is_unknown), errors="coerce")
    tmp = pd.DataFrame({"era": era.to_numpy(), "months": months.to_numpy()})
    chunk_max = tmp.groupby("era", observed=True)["months"].max()
    for era_label, value in chunk_max.items():
        if pd.notna(value):
            prev = report.era_max_followup.get(era_label, -1)
            report.era_max_followup[era_label] = max(prev, int(value))


def profile_csv(path: Path = RAW_CSV_PATH, chunksize: int = 200_000) -> ProfileReport:
    """Perfila o CSV inteiro em blocos, agregando as contagens de cada bloco."""
    report = ProfileReport()
    for chunk in iter_raw_chunks(chunksize=chunksize, path=path):
        _accumulate(report, chunk)
    return report
