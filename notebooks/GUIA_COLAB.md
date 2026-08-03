# Guia do Colab — treino e avaliação (passo a passo)

Companheiro do `colab_deep.ipynb`. Fluxo: **setup → validar CSV → robustez em validação → avaliação final (abre o teste 1×)**.

## Antes do Colab (local, uma vez)

1. **Push do repositório** para `CaetanoPorto/Deep-Learning-for-Survival-Analysis-Mortality-Risk-Prediction_2`.
   - O CSV **não** vai junto — está no `.gitignore` (`*.csv`). Confirme com `git status` que nenhum `.csv` está sendo commitado.
2. **Upload do CSV para o Google Drive.** Coloque `breast_cancer.csv` em, por exemplo, `MyDrive/TCC/breast_cancer.csv`. Anote o caminho — você vai colá-lo na célula 6.
3. No Colab, abra `notebooks/colab_deep.ipynb` (ou suba-o) e ative a GPU: **Runtime → Change runtime type → T4 GPU**.

## Célula por célula

| Célula | O que faz | O que conferir |
|---|---|---|
| **1 (md)** | Título e pré-requisitos | — |
| **2 — GPU** | `torch.cuda.is_available()` | Tem que imprimir **`True`** + `Tesla T4`. Se `False`, troque o runtime para GPU e rode de novo. |
| **3 — Drive** | `drive.mount('/content/drive')` | Autorize a conta Google; espere `Mounted at /content/drive`. |
| **4 — clone** | `git clone ...` + `cd repo` | Se o repo for **privado**, troque a URL por `https://<SEU_TOKEN>@github.com/CaetanoPorto/....git` (token = GitHub → Settings → Developer settings → Personal access tokens, escopo `repo`). |
| **5 — install** | `pip install scikit-survival lifelines pycox torchtuples` | ~1–2 min. Ignore avisos de dependência; usa o torch-GPU do Colab. |
| **6 — CSV path** | Define `SEER_CSV_PATH` e faz `assert os.path.exists(...)` | **Edite o caminho** para o seu CSV no Drive. Se o `assert` falhar, o caminho está errado. |
| **8 — Passo 1: profile** | `python scripts/profile_dataset.py` (~1 min) | Tem que dar **`26/26 invariantes OK`** e `CRITÉRIO DE ACEITE DA ETAPA 0: PASSOU`. Se falhar, o CSV difere do Perfil Empírico — **pare** e avise. |
| **10 — Passo 2: run_deep (val)** | `python scripts/run_deep.py --full --seeds 5` | Treina DeepSurv/DeepHit na base inteira, 5 sementes, e reporta **em validação** (o teste NÃO é aberto aqui). Olhe a linha `=== resumo em val ===`: C-index com desvio pequeno entre sementes = estável. GPU torna isso rápido (minutos). |
| **12 — Passo 3a: avaliação FINAL na amostra fixa** | `python scripts/run_evaluation.py --sample-n 100000 --eval-set test --n-boot 200` | **Abre o teste da amostra fixa 1×.** Treina os **6 modelos** (incl. GBS) e imprime tabela comparativa + calibração + era + validação temporal + sensibilidade. ⚠️ **O GBS aqui leva ~1–1,5 h** (O(n²), ADR-010) — mantenha a aba ativa. |
| **13 — Passo 3b: sensibilidade na base inteira** | `python scripts/run_evaluation.py --full --eval-set test --n-boot 200` | **Abre o teste da base inteira 1×.** Roda os modelos que escalam (Cox, Cox+splines, RSF, DeepSurv, DeepHit); **o GBS é pulado automaticamente** (mensagem `[ADR-010]`). Compare o ranking com o da amostra fixa — só reporte se mudar muito. |

## Interpretando a saída da avaliação final (células 12 e 13)

- **Tabela comparativa** — ordenada por C-Uno. Diferença de C-index só conta como melhora se os **IC bootstrap não se sobrepõem** e a diferença é > 0,01 (Protocolo de Validação).
- **Calibração por decil** — `previsto` ≈ `observado` em cada decil = bem calibrado.
- **Métricas por era** — o C-index deve se manter dentro de cada era (não vir só do tempo de calendário).
- **Validação temporal** (o gate de era, ADR-009) — o `Harrell temporal` (treino 2010-15 → teste 2016-17) **não pode desabar** frente ao Harrell do split aleatório. Se desabar, a era estava carregando o resultado.
- **Sensibilidade** — `delta` pequeno ao reincluir o tempo desconhecido (óbito em t=0) = exclusão justificada; `delta` grande = limitação de peso na Discussão.

## Salvar os resultados

A saída é texto no notebook. Para guardar, copie as tabelas para uma nota em `04-Experimentos/` do vault (template de experimento), com a semente e a data — "rodada não registrada não aconteceu" (RUNBOOK Etapa 9).

## Regra de ouro

O **teste é aberto uma única vez** por conjunto (amostra fixa na 3a, base inteira na 3b). Não re-rode as células 12/13 mudando hiperparâmetro para "melhorar" o número — isso transforma o teste em validação. Ajustes saem do Passo 2 (validação).
