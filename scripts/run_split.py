"""Etapa 4 do RUNBOOK — split treino/validação/teste.

Carrega a base inteira, monta a coorte analítica (Etapas 1-3), REVALIDA a limpeza da
Etapa 2 na base completa (checagem full-data que ficou pendente) e faz o split 70/15/15
estratificado por event × era.

Critério de aceite:
    censura entre folds < 3%
    distribuição de era equivalente nos 3 folds
    nenhum fit ocorreu até aqui (o split só particiona linhas)

Uso:
    python scripts/run_split.py [--era-restriction main]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.checks.invariants import all_passed, check_cleaned_frame, check_split, format_checks
from src.config import ERA_LABELS, ERA_MAIN_COHORT_RECOMMENDED, RANDOM_SEED, set_global_seed
from src.data.build import build_analytic_frame
from src.data.era import ERA_COLUMN
from src.preprocessing.split import train_val_test_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era-restriction", choices=["none", "main"], default="none",
                        help="'none' = base completa (exploração); 'main' = 2010-2022")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    restriction = ERA_MAIN_COHORT_RECOMMENDED if args.era_restriction == "main" else None

    print("Carregando a base inteira e montando a coorte analítica (~1-2 min)...")
    result = build_analytic_frame(era_restriction=restriction)
    df = result.analytic
    print(f"coorte analítica: {len(df):,} linhas ({args.era_restriction})")

    # --- Revalidação full-data da Etapa 2 (checagem adiada) ---
    clean_checks = check_cleaned_frame(df)
    n_clean_fail = sum(1 for c in clean_checks if not c.ok)
    print(f"\n[Etapa 2 full-data] invariantes de limpeza: {len(clean_checks) - n_clean_fail}/{len(clean_checks)} OK")
    if n_clean_fail:
        print(format_checks(clean_checks))
        print("\n[X] A limpeza falhou na base completa — PARE.")
        return 1

    # --- Split ---
    split = train_val_test_split(df)
    print(f"\nsplit: train={len(split.train):,}  val={len(split.val):,}  test={len(split.test):,}")

    print("\n=== Taxa de censura por fold ===")
    for name, rate in split.censoring_rates.items():
        print(f"  {name:5}: censura={rate:.4f}  evento={1 - rate:.4f}")
    print(f"  gap máx entre folds: {split.max_censoring_gap:.4f}")

    print("\n=== Proporção de era por fold (equivalência) ===")
    header = "  era        " + "".join(f"{n:>10}" for n in ("train", "val", "test"))
    print(header)
    for era in ERA_LABELS:
        row = "".join(f"{split.era_proportions[n].get(era, 0):>10.4f}" for n in ("train", "val", "test"))
        print(f"  {era:9}{row}")
    print(f"  gap máx de proporção de era entre folds: {split.max_era_gap:.4f}")

    checks = check_split(split, n_total=len(df))
    print("\n=== Etapa 4 — critério de aceite ===")
    print(format_checks(checks))

    n_fail = sum(1 for c in checks if not c.ok)
    print("\n=== Resultado ===")
    print(f"invariantes do split: {len(checks) - n_fail}/{len(checks)} OK")
    if all_passed(checks):
        print("\nCRITÉRIO DE ACEITE DA ETAPA 4: PASSOU [OK]")
        print("Nenhum fit ocorreu (o split só particiona linhas). Pronto para a Etapa 5.")
        return 0
    print("\nCRITÉRIO DE ACEITE DA ETAPA 4: FALHOU [X] — PARE e reporte.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
