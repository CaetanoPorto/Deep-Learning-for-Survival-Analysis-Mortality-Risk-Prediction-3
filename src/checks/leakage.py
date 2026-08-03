"""Checagens anti-vazamento da Etapa 5.

Duas coisas distintas:
  1. `assert_no_target_leakage` — garante que nenhuma coluna de feature deriva do alvo
     (duration/event/survival/vital) nem é a própria era. Vazamento óbvio.
  2. `era_predictability` — treina um classificador FORTE para tentar recuperar
     `era_diagnostico` a partir das features pré-processadas. Se ele conseguir muito acima
     do baseline, a era (confundidor estrutural) está codificada nas features pela porta
     lateral do missing. Ver Armadilha - Era de diagnóstico vaza pelo padrão de Blanks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Substrings proibidas em nomes de coluna de feature.
FORBIDDEN_SUBSTRINGS = ["survival", "vital status", "vital_status", "duration", "event", "era_diagnostico"]


def find_target_leakage(feature_names: list[str]) -> list[str]:
    """Devolve os nomes de feature que casam com alguma substring proibida (vazio = ok)."""
    bad = []
    for col in feature_names:
        low = col.lower()
        if any(sub in low for sub in FORBIDDEN_SUBSTRINGS):
            bad.append(col)
    return bad


def assert_no_target_leakage(feature_names: list[str]) -> None:
    bad = find_target_leakage(feature_names)
    if bad:
        raise AssertionError(
            f"Colunas suspeitas de vazamento do alvo/era em X: {bad}. "
            "duration/event/era_diagnostico não podem virar feature."
        )


@dataclass
class EraProbeResult:
    accuracy: float
    baseline_accuracy: float  # acurácia de chutar sempre a era majoritária
    macro_auc: float  # AUC one-vs-rest, média macro
    n_classes: int
    lift: float  # accuracy - baseline_accuracy


def era_predictability(
    X_train: np.ndarray,
    era_train: np.ndarray,
    X_eval: np.ndarray,
    era_eval: np.ndarray,
    seed: int = 42,
) -> EraProbeResult:
    """Treina um HistGradientBoosting (probe forte e rápido) para prever a era a partir
    das features e avalia num conjunto separado. Quanto mais perto do baseline, menos a
    era está codificada nas features.
    """
    era_train = np.asarray(era_train).astype(str)
    era_eval = np.asarray(era_eval).astype(str)

    clf = HistGradientBoostingClassifier(random_state=seed, max_depth=6, max_iter=200)
    clf.fit(X_train, era_train)

    pred = clf.predict(X_eval)
    proba = clf.predict_proba(X_eval)

    accuracy = float(accuracy_score(era_eval, pred))
    # baseline = frequência da classe majoritária no conjunto de avaliação
    _, counts = np.unique(era_eval, return_counts=True)
    baseline = float(counts.max() / counts.sum())
    try:
        macro_auc = float(roc_auc_score(era_eval, proba, multi_class="ovr",
                                        average="macro", labels=clf.classes_))
    except ValueError:
        macro_auc = float("nan")

    return EraProbeResult(
        accuracy=accuracy,
        baseline_accuracy=baseline,
        macro_auc=macro_auc,
        n_classes=int(len(clf.classes_)),
        lift=accuracy - baseline,
    )
