"""Testes leves da Etapa 7 (sem treino): discretização do tempo com bin para t=0 e o
C-index de Antolini rodando com os shims de compatibilidade do pycox."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate.curves import antolini_cindex
from src.models import deephit


def test_deephit_discretiza_com_bin_para_t0():
    dur = np.array([0, 0, 5, 10, 20, 50, 100], dtype=float)
    ev = np.array([1, 0, 1, 1, 0, 1, 0])
    lt = deephit.fit_label_transform(deephit.build_label_transform(), dur, ev)
    idx, e = deephit.make_target(lt, dur, ev)
    assert len(idx) == len(dur)
    assert idx.min() >= 0  # t=0 cai num bin válido, sem erro
    assert lt.cuts[0] <= 0.0 + 1e-9  # o primeiro corte cobre t=0


def test_antolini_roda_com_shim_pycox():
    times = np.array([0, 1, 2, 3])
    base = np.linspace(1.0, 0.1, 4)
    # curvas de sobrevivência distintas por indivíduo (decrescentes)
    surv = pd.DataFrame({i: base ** (i + 1) for i in range(5)}, index=times)
    c = antolini_cindex(surv, duration=[1, 2, 3, 1, 2], event=[1, 1, 0, 1, 1])
    assert 0.0 <= c <= 1.0
