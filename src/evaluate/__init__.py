"""Avaliação: métricas de discriminação/calibração compartilhadas por todos os modelos.

Todos os modelos (Cox, splines, RSF, GBS, DeepSurv, DeepHit) usam exatamente as mesmas
funções daqui, para que a comparação final seja justa — a mesma conta de C-index para
todos, em vez de reimplementar (e arriscar reimplementar errado) por modelo.
"""
