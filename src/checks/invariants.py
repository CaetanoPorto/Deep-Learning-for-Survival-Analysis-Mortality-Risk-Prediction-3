"""Comparação do perfil medido contra os números esperados (Perfil Empírico da Base /
Contrato de Dados). É o critério de aceite da Etapa 0 e a base dos testes de invariância.
"""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.data.cohort import CohortFlow
from src.data.profiling import ProfileReport


@dataclass
class Check:
    name: str
    expected: object
    measured: object

    @property
    def ok(self) -> bool:
        return self.expected == self.measured


def check_profile(report: ProfileReport) -> list[Check]:
    """Produz a lista de checagens do perfil empírico, na ordem em que aparecem no
    Perfil Empírico da Base. Cada `Check` compara um número medido com o esperado.
    """
    checks: list[Check] = [
        Check("N de linhas", config.EXPECTED_N_ROWS, report.n_rows),
        Check(
            "Vital status: Alive",
            config.EXPECTED_VITAL_STATUS_COUNTS["Alive"],
            report.vital_status_counts.get("Alive", 0),
        ),
        Check(
            "Vital status: Dead",
            config.EXPECTED_VITAL_STATUS_COUNTS["Dead"],
            report.vital_status_counts.get("Dead", 0),
        ),
        Check(
            "Vital status: só {Alive, Dead}",
            {"Alive", "Dead"},
            set(report.vital_status_counts),
        ),
        Check("Survival months == Unknown", config.EXPECTED_SURVIVAL_UNKNOWN, report.survival_unknown),
        Check(
            "Tempo desconhecido é 100% óbito",
            report.survival_unknown,
            report.survival_unknown_and_dead,
        ),
        Check("Survival months == 0000", config.EXPECTED_SURVIVAL_ZERO, report.survival_zero),
        Check(
            "Sexo: Female",
            config.EXPECTED_SEX_COUNTS["Female"],
            report.sex_counts.get("Female", 0),
        ),
        Check(
            "Sexo: Male",
            config.EXPECTED_SEX_COUNTS["Male"],
            report.sex_counts.get("Male", 0),
        ),
    ]

    # Contagem por era + soma == N.
    for era in config.ERA_LABELS:
        checks.append(
            Check(f"Era {era}: n", config.EXPECTED_ERA_COUNTS[era], report.era_counts.get(era, 0))
        )
    checks.append(
        Check("Eras somam N", config.EXPECTED_N_ROWS, sum(report.era_counts.values()))
    )

    # Follow-up máximo por era (a validação decisiva da reconstrução).
    for era in config.ERA_LABELS:
        checks.append(
            Check(
                f"Era {era}: follow-up máx.",
                config.ERA_MAX_FOLLOWUP[era],
                report.era_max_followup.get(era, -1),
            )
        )

    # Invariante dos linfonodos: nodes_positive==98 <=> nodes_examined==00, exatamente
    # 198.635 registros dos dois lados e da interseção.
    checks.append(
        Check("Linfonodos: nodes_positive==98", config.EXPECTED_NODE_INVARIANT_COUNT, report.nodes_positive_98)
    )
    checks.append(
        Check("Linfonodos: nodes_examined==00", config.EXPECTED_NODE_INVARIANT_COUNT, report.nodes_examined_00)
    )
    checks.append(
        Check("Linfonodos: 98 <=> 00 (intersecao)", config.EXPECTED_NODE_INVARIANT_COUNT, report.nodes_98_and_00)
    )

    # Complementaridade do grau: Blank(s) de cada coluna + soma == N.
    for col, expected_blank in config.EXPECTED_GRADE_BLANK_COUNTS.items():
        checks.append(
            Check(f"Grau Blank(s): {col}", expected_blank, report.grade_blank_counts.get(col, 0))
        )
    checks.append(
        Check(
            "Grau: Blank(s) das duas colunas somam N",
            config.EXPECTED_N_ROWS,
            sum(report.grade_blank_counts.values()),
        )
    )

    return checks


def check_cohort_flow(flow: CohortFlow) -> list[Check]:
    """Critério de aceite da Etapa 1: o fluxo de coorte reproduz exatamente as contagens
    medidas nos Critérios de Inclusão e Exclusão.
    """
    checks: list[Check] = [
        Check("N inicial", config.EXPECTED_N_ROWS, flow.n_raw),
        Check("Homens removidos", config.EXPECTED_SEX_COUNTS["Male"], flow.n_male_removed),
        Check("Coorte feminina", config.EXPECTED_SEX_COUNTS["Female"], flow.n_female),
        Check(
            "Tempo desconhecido (total)",
            config.EXPECTED_UNKNOWN_TIME_TOTAL,
            flow.n_unknown_time_total,
        ),
        Check(
            "Tempo desconhecido (homens)",
            config.EXPECTED_UNKNOWN_TIME_MALE,
            flow.n_unknown_time_male,
        ),
        Check(
            "Tempo desconhecido removido (feminino)",
            config.EXPECTED_UNKNOWN_TIME_FEMALE,
            flow.n_unknown_time_female_removed,
        ),
        Check(
            "Removidos por tempo são 100% óbito",
            flow.n_unknown_time_female_removed,
            flow.n_unknown_time_female_dead,
        ),
        Check("Coorte analítica", config.EXPECTED_ANALYTIC_COHORT, flow.n_analytic),
        Check(
            "Feminina − tempo desc. == analítica",
            flow.n_female - flow.n_unknown_time_female_removed,
            flow.n_analytic,
        ),
        Check(
            "Eras (analítica) somam N analítico",
            flow.n_analytic,
            sum(flow.era_counts_analytic.values()),
        ),
    ]
    for era in config.ERA_LABELS:
        checks.append(
            Check(
                f"Era {era}: follow-up máx. (analítica)",
                config.ERA_MAX_FOLLOWUP[era],
                flow.era_max_followup_analytic.get(era, -1),
            )
        )
    return checks


def _domain_ok(series, lo: float, hi: float) -> bool:
    s = series.dropna()
    return bool(s.between(lo, hi).all())


def _levels_ok(series, allowed: set) -> bool:
    return set(series.dropna().unique()).issubset(allowed)


def check_cleaned_frame(df) -> list[Check]:
    """Critério de aceite da Etapa 2 (Contrato de Dados / ADR-005): nenhum Blank(s)
    sobrevive, sentinelas decodificados dentro do domínio, categóricas nos níveis certos.
    """
    # Nenhum Blank(s) sobrevivente em nenhuma coluna.
    blank_cols = [c for c in df.columns if bool((df[c] == config.BLANK_TOKEN).any())]

    node_count, nodes_exam = df["node_count"], df["nodes_examined_n"]
    size = df["tumor_size_mm"]
    checks: list[Check] = [
        Check("Nenhum Blank(s) sobrevive", [], blank_cols),
        Check(f"node_count.max()={node_count.max():.0f} <= 90", True, node_count.max() <= 90),
        Check("node_count em [0,90]∪NaN", True, _domain_ok(node_count, 0, 90)),
        Check(f"nodes_examined_n.max()={nodes_exam.max():.0f} <= 90", True, nodes_exam.max() <= 90),
        Check("nodes_examined_n em [0,90]∪NaN", True, _domain_ok(nodes_exam, 0, 90)),
        Check(f"tumor_size_mm.max()={size.max():.0f} <= 200", True, size.max() <= 200),
        Check("tumor_size_mm em [0,200]∪NaN", True, _domain_ok(size, 0, 200)),
        Check("node_status nos 4 níveis", True, _levels_ok(df["node_status"], set(config.NODE_STATUS_LEVELS))),
        # consistência interna dos linfonodos (sem precisar do bruto)
        Check(
            "negativo ⟺ node_count==0",
            True,
            bool(((df["node_status"] == config.NODE_STATUS_NEGATIVE) == (node_count == 0)).all()),
        ),
        Check(
            "node_count não-nulo ⟹ negativo/positivo",
            True,
            _levels_ok(
                df.loc[node_count.notna(), "node_status"],
                {config.NODE_STATUS_NEGATIVE, config.NODE_STATUS_POSITIVE},
            ),
        ),
        Check("surgery_group nos 4 níveis", True, _levels_ok(df["surgery_group"], set(config.SURGERY_GROUP_LEVELS))),
        Check("surgery_group sem NaN", True, bool(df["surgery_group"].notna().all())),
        Check("radiation_group nos 4 níveis (cobertura total)", True,
              _levels_ok(df["radiation_group"], set(config.RADIATION_GROUP_LEVELS)) and bool(df["radiation_group"].notna().all())),
        Check("chemotherapy em {Yes, No/Unknown}", True, _levels_ok(df["chemotherapy"], {"Yes", "No/Unknown"})),
        Check("age_mid sem NaN (todas as faixas mapeadas)", True, bool(df["age_mid"].notna().all())),
        Check("age_mid em [0,92]", True, _domain_ok(df["age_mid"], 0, 92)),
        Check("stage ordinal ∪ NaN", True, _levels_ok(df["stage"], set(config.STAGE_ORDER))),
        Check("tumor_grade em {1,2,3,4}∪NaN", True, _levels_ok(df["tumor_grade"], set(config.GRADE_ORDER))),
        Check("er/pr/her2 em {Positive,Negative}∪NaN", True, all(
            _levels_ok(df[c], {"Positive", "Negative"}) for c in ("er_status", "pr_status", "her2_status")
        )),
        # sanidade distribucional (Armadilha - Códigos sentinela)
        Check(f"node_count mediana={node_count.median():.0f} <= 2", True, node_count.median() <= 2),
        Check(f"tumor_size_mm p99={size.quantile(0.99):.0f} < 150", True, size.quantile(0.99) < 150),
    ]
    return checks


def check_split(split, n_total: int) -> list[Check]:
    """Critério de aceite da Etapa 4: censura equilibrada entre folds, era equivalente
    nos três folds, e nenhuma linha perdida/duplicada no split.
    """
    n_folds = len(split.train) + len(split.val) + len(split.test)
    frac_train = len(split.train) / n_total
    return [
        Check(
            f"censura entre folds={split.max_censoring_gap:.2%} < {config.CENSORING_RATE_TOLERANCE:.0%}",
            True,
            split.max_censoring_gap < config.CENSORING_RATE_TOLERANCE,
        ),
        Check(
            f"era equivalente: gap máx={split.max_era_gap:.2%} < {config.ERA_BALANCE_TOLERANCE:.0%}",
            True,
            split.max_era_gap < config.ERA_BALANCE_TOLERANCE,
        ),
        Check("nenhuma linha perdida no split", n_total, n_folds),
        Check(f"fração de treino={frac_train:.3f} ≈ 0,70", True, abs(frac_train - config.TRAIN_FRAC) < 0.005),
    ]


def check_cindex(model_name: str, train_c: float, test_c: float) -> list[Check]:
    """Critério de aceite por modelo (RUNBOOK Etapa 6): 0,55 < C-index de teste < 0,95 e
    |gap treino − teste| < 0,05. Fora disso: PARAR (vazamento ou modelo que não aprendeu).
    """
    gap = train_c - test_c
    return [
        Check(
            f"{model_name}: 0,55 < C_teste < 0,95 (={test_c:.4f})",
            True,
            config.SANITY_CINDEX_MIN < test_c < config.SANITY_CINDEX_MAX,
        ),
        Check(
            f"{model_name}: |gap treino-teste| < 0,05 (gap={gap:+.4f})",
            True,
            abs(gap) < config.SANITY_TRAIN_TEST_GAP_MAX,
        ),
    ]


def all_passed(checks: list[Check]) -> bool:
    return all(c.ok for c in checks)


def format_checks(checks: list[Check]) -> str:
    """Tabela texto (status | esperado | medido | checagem) para o log do script."""
    lines = [f"{'status':6} | {'esperado':>14} | {'medido':>14} | checagem", "-" * 78]
    for c in checks:
        mark = "  OK  " if c.ok else " FALHA"
        exp, meas = str(c.expected), str(c.measured)
        # dicts/sets longos: encurta para caber na coluna.
        exp = exp if len(exp) <= 14 else exp[:11] + "..."
        meas = meas if len(meas) <= 14 else meas[:11] + "..."
        lines.append(f"{mark} | {exp:>14} | {meas:>14} | {c.name}")
    return "\n".join(lines)
