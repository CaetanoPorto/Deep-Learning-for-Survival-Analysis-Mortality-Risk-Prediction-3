"""Pré-processamento: split e transformação (fit SÓ no treino).

A fronteira anti-vazamento vive aqui. O split (Etapa 4) parte a coorte ANTES de qualquer
`fit`; a transformação (Etapa 5) é ajustada exclusivamente no treino. Ver
[[Vazamento de Dados - C-Index 1.0]].
"""
