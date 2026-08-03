"""Etapa 5 do RUNBOOK — pré-processamento (fit SÓ no treino) + checagens anti-vazamento.

Carrega a base, monta a coorte, faz o split, ajusta o ColumnTransformer EXCLUSIVAMENTE
no treino e roda:
  - assert_no_target_leakage  (duration/event/era não aparecem em nenhuma coluna);
  - probe de era: um classificador forte tentando prever era_diagnostico a partir das
    features pré-processadas, na coorte COMPLETA e na coorte principal 2010-2022.

Critério de aceite (ADR-009):
  - assert_no_target_leakage passa -> gate duro obrigatório.
  - O probe de era é DIAGNÓSTICO qualitativo (reporta AUC/lift na coorte completa e na
    principal). NÃO há limiar numérico de AUC para aprovação em nenhuma etapa; o gate
    duro do vazamento de era é a VALIDAÇÃO TEMPORAL da Etapa 8 (treinar 2010-2015,
    testar 2016-2017 — Protocolo de Validação).

Uso:
    python scripts/run_preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from src.checks.leakage import assert_no_target_leakage, era_predictability, find_target_leakage
from src.config import ERA_MAIN_COHORT_RECOMMENDED, RANDOM_SEED, set_global_seed
from src.data.build import build_analytic_frame
from src.data.era import ERA_COLUMN
from src.preprocessing.split import train_val_test_split
from src.preprocessing.transform import INPUT_COLUMNS, build_preprocessor, fit_transform_splits

PROBE_TRAIN_N = 150_000
PROBE_EVAL_N = 50_000


def _subsample(n: int, k: int, rng) -> np.ndarray:
    return rng.choice(n, size=min(k, n), replace=False)


def _probe(X_train, era_train, X_eval, era_eval, rng):
    itr = _subsample(len(X_train), PROBE_TRAIN_N, rng)
    iev = _subsample(len(X_eval), PROBE_EVAL_N, rng)
    return era_predictability(X_train[itr], np.asarray(era_train)[itr],
                              X_eval[iev], np.asarray(era_eval)[iev], seed=RANDOM_SEED)


def _print_probe(titulo, r):
    print(f"  {titulo} ({r.n_classes} eras):")
    print(f"    baseline (era majoritária) : {r.baseline_accuracy:.4f}")
    print(f"    acurácia do probe          : {r.accuracy:.4f}")
    print(f"    lift (acc - baseline)      : {r.lift:+.4f}  ({r.lift * 100:+.1f} pp)")
    print(f"    AUC macro (one-vs-rest)    : {r.macro_auc:.4f}")


def main() -> int:
    set_global_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    print("Carregando a base e montando a coorte analítica (~1-2 min)...")
    df = build_analytic_frame(era_restriction=None).analytic
    split = train_val_test_split(df)

    print("Ajustando o ColumnTransformer SÓ no treino...")
    pre = fit_transform_splits(split.train, split.val, split.test)

    print("\n=== assert_no_target_leakage ===")
    print(f"colunas que o transformer recebeu (features): {INPUT_COLUMNS}")
    print(f"'duration', 'event', 'era_diagnostico' entre elas? "
          f"{any(c in INPUT_COLUMNS for c in ['duration', 'event', ERA_COLUMN])}")
    print(f"\nmatriz X: train={pre.X_train.shape}  val={pre.X_val.shape}  test={pre.X_test.shape}")
    print(f"{len(pre.feature_names)} colunas de feature geradas:")
    print("  " + ", ".join(pre.feature_names))
    bad = find_target_leakage(pre.feature_names)
    print(f"\ncolunas suspeitas (survival/vital/duration/event/era): {bad if bad else 'NENHUMA'}")
    assert_no_target_leakage(pre.feature_names)  # levanta se houver
    print("assert_no_target_leakage: PASSOU [OK]")

    # --- Probe de era: coorte completa ---
    print("\n=== Probe de era: as features preveem era_diagnostico? ===")
    full = _probe(pre.X_train, split.train[ERA_COLUMN], pre.X_val, split.val[ERA_COLUMN], rng)
    _print_probe("COORTE COMPLETA", full)

    # --- Probe de era: coorte principal 2010-2022 (refit do transformer nela) ---
    tr_m = split.train[split.train[ERA_COLUMN].isin(ERA_MAIN_COHORT_RECOMMENDED)]
    va_m = split.val[split.val[ERA_COLUMN].isin(ERA_MAIN_COHORT_RECOMMENDED)]
    ct_m = build_preprocessor()
    Xtr_m = ct_m.fit_transform(tr_m[INPUT_COLUMNS]).astype(np.float32)
    Xva_m = ct_m.transform(va_m[INPUT_COLUMNS]).astype(np.float32)
    main = _probe(Xtr_m, tr_m[ERA_COLUMN], Xva_m, va_m[ERA_COLUMN], rng)
    _print_probe("COORTE PRINCIPAL 2010-2022", main)

    # --- Veredito ---
    print("\n=== Resultado da Etapa 5 ===")
    print("gate duro: assert_no_target_leakage -> PASSOU")
    print(f"diagnóstico (SEM gate numérico, ADR-009): probe de era AUC macro = "
          f"completa {full.macro_auc:.4f} | principal {main.macro_auc:.4f}")
    print("O gate do vazamento de era é a validação temporal (Etapa 8), não o probe.")
    print("\nCRITÉRIO DE ACEITE DA ETAPA 5: PASSOU [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
