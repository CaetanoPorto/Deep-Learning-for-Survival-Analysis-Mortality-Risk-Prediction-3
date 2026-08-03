"""Modelos de sobrevivência. Todos partem da MESMA matriz de features pré-processada e
do MESMO split — é a única forma de comparar C-index entre eles de forma justa.

Baselines (Etapa 6): Cox, Cox+splines (lifelines), RSF, Gradient Boosting (scikit-survival).
Profundos (Etapa 7): DeepSurv, DeepHit (pycox).
"""
