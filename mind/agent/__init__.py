"""MIND — Muñdji Intelligent Neural Developer.

Sistema multi-agente autónomo de geração de código, inspirado na estrutura
do cérebro humano: um orquestrador (CORTEX), um revisor (CEREBELLUM) e seis
unidades de execução (NEURONS), num ciclo iterativo até 98-100% de
funcionalidade.

A lógica do MIND (CORTEX/CEREBELLUM/NEURONS) é independente de qualquer
interface — não sabe, nem precisa de saber, que interface a está a chamar.
"""

from .state import AgentState, new_state

__all__ = ["AgentState", "new_state"]
__version__ = "0.1.0"
