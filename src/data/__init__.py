"""Camada de dados: leitura, reconstrução de era, coorte, limpeza SEER, alvo.

A ordem de execução é fixa e obrigatória (Pipeline de Pré-processamento): era ANTES de
limpar `Blank(s)`, unificação de pares ANTES de mascarar sentinelas, e nenhum `fit`
antes do split. O orquestrador que garante essa ordem é `src/data/build.py`.
"""
