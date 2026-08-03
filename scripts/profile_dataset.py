"""Etapa 0 do RUNBOOK — regenerar/validar o perfil empírico da base.

Varre o CSV real, confere o cabeçalho (22 nomes literais) e compara todas as contagens
com o Perfil Empírico da Base gravado no config.

Critério de aceite: N e as contagens batem com [[Perfil Empírico da Base]]. Se NÃO
baterem, o CSV mudou -> o script sai com código != 0 e você deve atualizar a nota antes
de continuar (não prosseguir com números divergentes).

Uso:
    python scripts/profile_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# O console do Windows (cp1252) não codifica acentos/símbolos; força UTF-8 para os
# prints não quebrarem (e nunca crashar num glifo isolado).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.config import RAW_COLUMNS, RAW_CSV_PATH
from src.checks.invariants import all_passed, check_profile, format_checks
from src.data.loading import read_header
from src.data.profiling import profile_csv


def check_header() -> bool:
    """Confere que o cabeçalho do CSV é exatamente as 22 colunas esperadas, na ordem."""
    measured = read_header()
    expected = RAW_COLUMNS
    if measured == expected:
        print(f"[header] OK — {len(measured)} colunas, nomes e ordem conferem.")
        return True

    print("[header] FALHA — cabeçalho diverge do schema esperado (Dicionário de Variáveis):")
    print(f"         esperado: {len(expected)} colunas | medido: {len(measured)} colunas")
    for i, exp in enumerate(expected):
        got = measured[i] if i < len(measured) else "<ausente>"
        if got != exp:
            print(f"         pos {i}: esperado {exp!r} | medido {got!r}")
    for extra in measured[len(expected):]:
        print(f"         coluna extra no CSV: {extra!r}")
    return False


def main() -> int:
    print(f"CSV: {RAW_CSV_PATH}")
    if not RAW_CSV_PATH.exists():
        print(f"[erro] CSV não encontrado em {RAW_CSV_PATH}. Ajuste RAW_CSV_PATH no config.")
        return 2

    header_ok = check_header()

    print("\nPerfilando o CSV inteiro em streaming (pode levar ~1 min)...")
    report = profile_csv()
    checks = check_profile(report)

    print("\n=== Etapa 0 — Perfil Empírico vs. medido ===")
    print(format_checks(checks))

    profile_ok = all_passed(checks)
    n_fail = sum(1 for c in checks if not c.ok)

    print("\n=== Resultado ===")
    print(f"cabeçalho: {'OK' if header_ok else 'FALHA'}")
    print(f"invariantes: {len(checks) - n_fail}/{len(checks)} OK" + (f" — {n_fail} FALHA(S)" if n_fail else ""))

    if header_ok and profile_ok:
        print("\nCRITÉRIO DE ACEITE DA ETAPA 0: PASSOU [OK]")
        print("O CSV corresponde ao Perfil Empírico da Base. Pode avançar para a Etapa 1.")
        return 0

    print("\nCRITÉRIO DE ACEITE DA ETAPA 0: FALHOU [X]")
    print("O CSV diverge do perfil registrado. NÃO prossiga: atualize [[Perfil Empírico")
    print("da Base]] e avise o Guilherme (RUNBOOK Etapa 0).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
