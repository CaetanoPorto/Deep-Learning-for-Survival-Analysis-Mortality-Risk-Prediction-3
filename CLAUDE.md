# CLAUDE.md

Guia do Claude Code para este repositório.

---

# ⛔ LEIA ANTES DE ESCREVER QUALQUER LINHA DE CÓDIGO

Este repositório é o **código**. A fonte da verdade metodológica é o **vault Obsidian**:

```
D:\User\Documentos\Memoria Tcc Vault\TCC-Vault
```

A sessão deve ser iniciada com `--add-dir` apontando para lá (ver `INICIAR.md`).
Se o vault não estiver acessível, **pare e avise** — não escreva código adivinhando as
regras do SEER.

**Ordem de leitura obrigatória no início de qualquer tarefa de código:**

1. `09-Execucao\RUNBOOK - Pipeline Completo.md` — o plano, com critério de aceite por etapa
2. `09-Execucao\Contrato de Dados.md` — a especificação que este `src/` tem que cumprir
3. `02-Dados\Regras de Codificação SEER.md` — o que cada código do SEER significa
4. `02-Dados\Blanks e Eras de Diagnóstico.md` — o confundidor estrutural
5. As 4 notas de `06-Armadilhas\` — os erros já cometidos

Números medidos ficam em `02-Dados\Perfil Empírico da Base.md`. **Nunca cite número de
memória; consulte de lá.**

## Guardrails invioláveis

Estas seis regras valem mesmo que você não tenha aberto o vault ainda:

1. **Endpoint = sobrevivência global (all-cause).** `event = 1` se
   `Vital status recode == "Dead"`. O export **não tem causa do óbito** — não escreva
   nada que assuma riscos competitivos. (`05-Decisoes\ADR-001`)
2. **`fit` só depois do split.** Qualquer scaler/imputer/encoder é ajustado
   exclusivamente no treino. Foi assim que o C-index 1.0 apareceu antes.
3. **Reconstrua `era_diagnostico` ANTES de limpar `Blank(s)`.** O padrão de vazios
   codifica o ano do diagnóstico; limpar primeiro destrói a informação. A era é
   estrato do split, nunca feature preditiva solta. (`05-Decisoes\ADR-003`)
4. **Números altos do SEER são códigos, não quantidades.** `98` linfonodos positivos =
   "nenhum examinado"; `95`/`97` = nodo-positivo confirmado; `999` mm = desconhecido.
   Decodifique antes de escalar. (`05-Decisoes\ADR-005`)
5. **Nunca invente dado, resultado ou citação.** `> [!todo]` no vault fica `> [!todo]`
   até o Guilherme responder. Só cite artigos existentes em `07-Literatura\`.
6. **Critério de aceite falhou → pare e reporte.** Não contorne, não relaxe o limiar,
   não comente o assert.

## Invariantes verificadas na base real (usar como teste)

```python
N_TOTAL            = 1_365_329
N_FEMININO         = 1_355_045
N_COORTE_ANALITICA = 1_349_057   # após remover homens e tempo desconhecido
ERAS = {"2000-2003":205_049, "2004-2009":322_229, "2010-2015":361_127,
        "2016-2017":130_721, "2018-2022":346_203}
MAX_FOLLOWUP = {"2000-2003":275, "2004-2009":227, "2010-2015":155,
                "2016-2017":83,  "2018-2022":59}
# nodes_positive == "98"  ⟺  nodes_examined == "00"   (198.635 casos, zero exceções)
# Survival months == "Unknown"  ⟹  sempre "Dead"      (6.052 casos)
```

Se o código produzir números diferentes destes, **há bug no código** (ou o CSV mudou —
nesse caso avise antes de prosseguir).

## Divergências conhecidas entre `src/` e o vault

O `src/` atual foi escrito antes da auditoria de dados e **está desatualizado** em
pontos concretos. Ver a tabela "O que o `src/config.py` precisa mudar" em
`09-Execucao\Contrato de Dados.md`. Resumo:

| Arquivo | Problema |
|---|---|
| `config.py` | endpoint comentado como cause-specific; faixas de sentinela conservadoras demais; `Blank(s)` em `FAKE_NULL_TOKENS` global apaga a era |
| `cleaning.py` | limpa `Blank(s)` antes de reconstruir a era |
| `split.py` | estratifica só por `event`, falta `era_diagnostico` |
| — | não existe `era_diagnostico` nem `node_status` |

---

## Comandos

> Caminhos relativos a **`Claude_Code_TCC/`**, a raiz real do projeto (contém `.venv`,
> `src/`, `scripts/`, `tests/`).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/ -v                        # testes
.\.venv\Scripts\python.exe scripts/sanity_check_data.py               # só dados/pré-proc
.\.venv\Scripts\python.exe scripts/run_evaluation.py --sample-n 50000 # comparação completa
.\.venv\Scripts\python.exe scripts/run_evaluation.py --full           # base inteira
```

`run_baseline.py` / `run_deepsurv.py` / `run_deephit.py` rodam um modelo isolado.

## Arquitetura

- `src/config.py` — fonte única de seeds, caminhos, esquema das colunas SEER, frações de
  split e dataclasses de hiperparâmetros (`CoxConfig`, `MLPConfig`, `TrainConfig`,
  `DeepHitConfig`).
- `src/data/` — `loading.py` (leitura amostrada ou completa, tudo como string),
  `cleaning.py` (máscara de nulos/sentinelas, unificação de pares antigo/novo, filtro de
  sexo), `target.py` (monta `duration`/`event`).
- `src/preprocessing/` — `split.py` (treino/val/teste com checagem de taxa de censura) e
  `pipeline.py` (`ColumnTransformer`, `fit` só no treino — **é a fronteira
  anti-vazamento**; nunca vê `duration`/`event`).
- `src/models/` — um módulo por modelo, com superfície consistente
  `build_*`/`predict_risk`/`predict_survival_function`.
- `src/train/` — `lr_finder.py` e `trainer.py`, compartilhados por DeepSurv/DeepHit.
- `src/evaluate/` — `metrics.py` (C-index de Harrell, Antolini, Brier/IBS via
  `pycox.evaluation.EvalSurv`; **contém dois monkeypatches de compatibilidade — leia os
  comentários antes de "consertar" chamadas do pycox**), `sanity_checks.py`,
  `report.py`.

## Dados

`TCC_ML_DL-20260707T020648Z-3-001/TCC_ML_DL/breast_cancer.csv` — ~300 MB, 1.365.329
linhas, 22 colunas. **Nunca carregue inteiro em contexto**; use `load_raw_sample(n)`.
Dicionário completo das 22 colunas: `02-Dados\Dicionário de Variáveis.md` no vault.

`Artigos Referencias - Guilherme-.../` — PDFs de referência por tema.

## Regras do repositório

- O CSV **nunca** entra no Git (já houve incidente de commit de 285 MB).
- Todo `scripts/*.py` roda por padrão em amostra de 50k; base completa só com `--full`.
- O pipeline assume `Sex == "Female"` já filtrado e `duration`/`event` fora da matriz de
  features por construção — `assert_no_target_leakage` verifica isso após o
  pré-processamento em todos os scripts.
