"""Testes da unificação de pares antigo/novo (harmonização + combine_first)."""

from __future__ import annotations

import pandas as pd

from src.config import (
    COL_ER_NEW, COL_ER_OLD, COL_GRADE_2018, COL_GRADE_THRU_2017,
    COL_HER2_NEW, COL_HER2_OLD, COL_PR_NEW, COL_PR_OLD,
    COL_TUMOR_SIZE_CS, COL_TUMOR_SIZE_SUMMARY,
)
from src.data.cleaning import harmonize_and_combine_pairs


def _pairs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            # linha 0 = pré-2018 (usa a coluna antiga); linha 1 = 2018+ (usa a nova)
            COL_ER_OLD: ["Positive", "Positive"],
            COL_ER_NEW: ["Blank(s)", "ER negative"],
            COL_PR_OLD: ["Negative", "Positive"],
            COL_PR_NEW: ["Blank(s)", "PR positive"],
            COL_HER2_OLD: ["Positive", "Negative"],
            COL_HER2_NEW: ["Blank(s)", "HER2 positive"],
            COL_GRADE_THRU_2017: ["Moderately differentiated; Grade II", "Blank(s)"],
            COL_GRADE_2018: ["Blank(s)", "3"],
            COL_TUMOR_SIZE_CS: ["0015", "Blank(s)"],
            COL_TUMOR_SIZE_SUMMARY: ["Blank(s)", "0020"],
        }
    )


def test_combine_resolve_blank_e_prioriza_novo():
    out = harmonize_and_combine_pairs(_pairs_df())
    # Blank(s) estrutural resolvido pelo outro lado; vocabulário harmonizado; novo vence.
    assert list(out["er_status"]) == ["Positive", "Negative"]   # "ER negative" -> "Negative"
    assert list(out["pr_status"]) == ["Negative", "Positive"]
    assert list(out["her2_status"]) == ["Positive", "Positive"]  # "HER2 positive" -> "Positive"
    assert list(out["tumor_grade"]) == ["2", "3"]                # "Grade II" -> "2"
    assert list(out["tumor_size_mm"]) == ["0015", "0020"]        # ainda string; decodificado depois


def test_colunas_brutas_dos_pares_sao_removidas():
    out = harmonize_and_combine_pairs(_pairs_df())
    for col in (COL_ER_OLD, COL_ER_NEW, COL_GRADE_2018, COL_TUMOR_SIZE_CS):
        assert col not in out.columns
    for col in ("er_status", "pr_status", "her2_status", "tumor_grade", "tumor_size_mm"):
        assert col in out.columns
