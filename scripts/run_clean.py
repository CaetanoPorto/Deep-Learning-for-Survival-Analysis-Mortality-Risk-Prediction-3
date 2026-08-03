"""Etapa 2 do RUNBOOK — limpeza SEER.

Monta a coorte, aplica a limpeza (pares, nulos reais, sentinelas, agrupamentos) e valida
o critério de aceite: nenhum Blank(s) sobrevive, node_count<=90, tumor_size_mm<=200,
categóricas nos níveis certos.

Uso:
    python scripts/run_clean.py --sample-n 50000   # rápido, para iterar
    python scripts/run_clean.py --full             # dataset inteiro (validação real)
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

from src.checks.invariants import all_passed, check_cleaned_frame, format_checks
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_SEED, set_global_seed
from src.data.build import build_analytic_frame
from src.data.features import unmapped_age
from src.data.loading import load_raw_full, load_raw_sample
from src.config import COL_AGE, COL_RADIATION, RADIATION_GROUP_MAP


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    set_global_seed(RANDOM_SEED)

    if args.full or args.sample_n is None:
        print("Carregando dataset completo (~1,36M linhas)...")
        raw = load_raw_full()
    else:
        print(f"Carregando amostra de {args.sample_n} linhas (seed={RANDOM_SEED})...")
        raw = load_raw_sample(sample_n=args.sample_n, seed=RANDOM_SEED)

    # cobertura dos mapas (antes de limpar) — surpresa de dados faz parar.
    unmapped_ages = unmapped_age(raw[COL_AGE])
    unmapped_rad = sorted(set(raw[COL_RADIATION].unique()) - set(RADIATION_GROUP_MAP))
    if unmapped_ages:
        print(f"[aviso] faixas etárias não mapeadas: {unmapped_ages}")
    if unmapped_rad:
        print(f"[aviso] valores de radiação não mapeados: {unmapped_rad}")

    result = build_analytic_frame(raw)
    df = result.analytic

    print(f"\ncoorte analítica: {len(df):,} linhas × {df.shape[1]} colunas")
    print(f"features numéricas: {NUMERIC_FEATURES}")
    print(f"features categóricas: {CATEGORICAL_FEATURES}")
    print("guardado (não modelado): breast_subtype, tumor_size_diffuse")

    print("\n=== node_status (recupera nodo-positivos escondidos) ===")
    print(df["node_status"].value_counts(dropna=False).to_string())
    print("\n=== numéricas (describe) ===")
    print(df[NUMERIC_FEATURES].describe().loc[["min", "50%", "max"]].to_string())
    print(f"\ntumor_size_diffuse (código 998): {int(df['tumor_size_diffuse'].sum()):,}")

    print("\n=== taxa de NaN por feature categórica ===")
    for col in CATEGORICAL_FEATURES:
        na = df[col].isna().mean()
        print(f"  {col:16} NaN={na:6.2%}")

    checks = check_cleaned_frame(df)
    print("\n=== Etapa 2 — critério de aceite ===")
    print(format_checks(checks))

    n_fail = sum(1 for c in checks if not c.ok)
    print("\n=== Resultado ===")
    print(f"invariantes de limpeza: {len(checks) - n_fail}/{len(checks)} OK" + (f" — {n_fail} FALHA(S)" if n_fail else ""))
    if all_passed(checks) and not unmapped_ages and not unmapped_rad:
        print("\nCRITÉRIO DE ACEITE DA ETAPA 2: PASSOU [OK]")
        return 0
    print("\nCRITÉRIO DE ACEITE DA ETAPA 2: FALHOU [X] — PARE e reporte.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
