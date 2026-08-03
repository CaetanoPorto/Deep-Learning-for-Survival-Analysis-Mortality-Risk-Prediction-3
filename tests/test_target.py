"""Testes da construção do alvo (duration, event)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import COL_SURVIVAL_MONTHS, COL_VITAL_STATUS
from src.data.target import build_target


def _df(survival, vital) -> pd.DataFrame:
    return pd.DataFrame({COL_SURVIVAL_MONTHS: survival, COL_VITAL_STATUS: vital})


def test_duration_e_event_derivados():
    out = build_target(_df(["0012", "0000", "0275"], ["Dead", "Alive", "Dead"]))
    assert list(out["duration"]) == [12.0, 0.0, 275.0]
    assert list(out["event"]) == [1, 0, 1]
    assert out["duration"].dtype == float
    assert out["event"].dtype == np.dtype("int64") or np.issubdtype(out["event"].dtype, np.integer)


def test_rotulos_brutos_sao_removidos():
    out = build_target(_df(["0010"], ["Dead"]))
    assert COL_SURVIVAL_MONTHS not in out.columns
    assert COL_VITAL_STATUS not in out.columns


def test_erro_se_tempo_desconhecido_vazar():
    # "Unknown" nunca deveria chegar aqui (separado na coorte); se chegar, é bug -> erro.
    with pytest.raises(ValueError, match="não numérico"):
        build_target(_df(["0010", "Unknown"], ["Dead", "Dead"]))


def test_erro_se_vital_status_inesperado():
    with pytest.raises(ValueError, match="fora de"):
        build_target(_df(["0010"], ["Zumbi"]))
