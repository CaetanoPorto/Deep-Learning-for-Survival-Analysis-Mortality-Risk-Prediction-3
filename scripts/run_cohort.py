"""Etapa 1 do RUNBOOK — carregamento e coorte.

Carrega o CSV, reconstrói a era, aplica os filtros de coorte e deriva duration/event.
Imprime a tabela de fluxo de coorte (pronta para colar em Critérios de Inclusão e
Exclusão) e valida o critério de aceite contra as contagens medidas.

Critério de aceite:
    N inicial          == 1.365.329
    após filtro sexo   == 1.355.045
    coorte analítica   == 1.349.057
    100% dos removidos por tempo desconhecido são Dead
    max follow-up por era (analítica) == 275 / 227 / 155 / 83 / 59

Uso:
    python scripts/run_cohort.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.checks.invariants import all_passed, check_cohort_flow, format_checks
from src.config import RANDOM_SEED, RAW_CSV_PATH, set_global_seed
from src.data.build import build_cohort
from src.data.era import ERA_COLUMN


def format_flow(flow) -> str:
    return "\n".join(
        [
            f"Base bruta ......................... {flow.n_raw:>10,}",
            f"  - Sexo masculino ................. {-flow.n_male_removed:>10,}",
            f"  = Coorte feminina ................ {flow.n_female:>10,}",
            f"  - Survival months desconhecido ... {-flow.n_unknown_time_female_removed:>10,}"
            f"   (dos {flow.n_unknown_time_total:,} totais, {flow.n_unknown_time_male} são homens)",
            f"  = Coorte analítica ............... {flow.n_analytic:>10,}",
        ]
    )


def main() -> int:
    print(f"CSV: {RAW_CSV_PATH}")
    if not RAW_CSV_PATH.exists():
        print(f"[erro] CSV não encontrado em {RAW_CSV_PATH}.")
        return 2

    set_global_seed(RANDOM_SEED)

    print("\nCarregando o CSV completo e montando a coorte (pode levar ~1 min)...")
    result = build_cohort()
    flow, analytic = result.flow, result.analytic

    print("\n=== Fluxo de coorte (colar em Critérios de Inclusão e Exclusão) ===")
    print(format_flow(flow))

    print("\n=== Distribuição da coorte analítica por era ===")
    for era in analytic[ERA_COLUMN].cat.categories:
        n = flow.era_counts_analytic.get(era, 0)
        fmax = flow.era_max_followup_analytic.get(era, -1)
        print(f"  {era}: n={n:>9,}  follow-up máx={fmax:>4} meses")

    n_event = int(analytic["event"].sum())
    print("\n=== Alvo (coorte analítica) ===")
    print(f"  duration: min={analytic['duration'].min():.0f} max={analytic['duration'].max():.0f} meses")
    print(f"  event=1 (óbitos): {n_event:,} ({analytic['event'].mean():.4f})")
    print(f"  duration==0 mantidos (ADR-004): {flow.n_duration_zero_analytic:,}")
    print(f"  coorte de tempo desconhecido reservada p/ sensibilidade: {len(result.unknown_time):,} linhas")

    checks = check_cohort_flow(flow)
    print("\n=== Etapa 1 — critério de aceite ===")
    print(format_checks(checks))

    n_fail = sum(1 for c in checks if not c.ok)
    print("\n=== Resultado ===")
    print(f"invariantes de coorte: {len(checks) - n_fail}/{len(checks)} OK" + (f" — {n_fail} FALHA(S)" if n_fail else ""))

    if all_passed(checks):
        print("\nCRITÉRIO DE ACEITE DA ETAPA 1: PASSOU [OK]")
        return 0
    print("\nCRITÉRIO DE ACEITE DA ETAPA 1: FALHOU [X] — PARE e reporte.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
