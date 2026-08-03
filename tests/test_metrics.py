"""Testes das métricas de discriminação (Harrell, Uno, array estruturado)."""

from __future__ import annotations

from src.evaluate.metrics import harrell_cindex, to_structured, uno_cindex


def test_harrell_perfeito_e_invertido():
    dur = [1, 2, 3, 4, 5]
    ev = [1, 1, 1, 1, 1]
    # risco DECRESCENTE com a duração (maior risco = morre antes) -> concordância perfeita
    assert harrell_cindex(dur, ev, [5, 4, 3, 2, 1]) == 1.0
    # risco crescente com a duração -> totalmente discordante
    assert harrell_cindex(dur, ev, [1, 2, 3, 4, 5]) == 0.0


def test_to_structured():
    y = to_structured([1.0, 2.0, 3.0], [1, 0, 1])
    assert y.dtype.names == ("event", "time")
    assert y["event"].tolist() == [True, False, True]
    assert y["time"].tolist() == [1.0, 2.0, 3.0]


def test_uno_recompensa_bom_ranking():
    dur = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ev = [1, 1, 1, 0, 1, 0]
    y = to_structured(dur, ev)
    bom = [6, 5, 4, 3, 2, 1]  # maior risco para quem morre antes
    ruim = [1, 2, 3, 4, 5, 6]
    assert uno_cindex(y, y, bom, tau=5.0) > uno_cindex(y, y, ruim, tau=5.0)
