"""Portões de verificação do pipeline: invariantes do Contrato de Dados e checagens
anti-vazamento. Cada critério de aceite do RUNBOOK é um conjunto de checagens aqui;
se um portão falha, o pipeline PARA e reporta — não contorna, não relaxa o limiar.
"""
