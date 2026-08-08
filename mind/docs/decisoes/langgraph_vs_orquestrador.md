# Decisão: dualidade LangGraph / orquestrador Python

**Data:** 3 de Agosto de 2026
**Estado:** decidido e implementado

## O problema

O grafo LangGraph estava construído e compilava, mas a execução real corria
por uma reimplementação paralela em Python (`MindGraph.run()`), escrita para
ter controlo explícito do loop e do rollback. Duas descrições do mesmo fluxo,
ambas executáveis, com risco real de divergirem com o tempo.

Manter as duas a fingir que são uma só estava fora de questão. A direcção a
tomar dependia de uma pergunta concreta, não de preferência: **o LangGraph
consegue exprimir o fluxo real do MIND sem distorcer a lógica?**

## O teste técnico

`teste_tecnico_langgraph.py` (neste directório) constrói o fluxo completo do
MIND como `StateGraph` compilado e verifica, **em execução real e todas ao
mesmo tempo**, as cinco decisões que a arquitectura não pode perder:

| # | Decisão | Resultado |
|---|---|---|
| 1 | Rollback: iteração pior parte da melhor versão anterior | Exprime |
| 2 | Regra de divergência: >15pp entre relatórios força 3ª ronda | Exprime |
| 3 | Activação dinâmica: só os NEURONS visados a partir da 2ª iteração | Exprime |
| 4 | Circuit breaker: timeout individual por NEURON sem bloquear o ciclo | Exprime |
| 5 | Limite de iterações com saída para `needs_human` | Exprime |

O teste é reprodutível: `python docs/decisoes/teste_tecnico_langgraph.py`.

Uma execução observada, com a trajectória dos NEURONS por iteração:

```
neurons por iteração: [['neuron_1','neuron_2'], ['neuron_2'], ['neuron_1'], ['neuron_1']]
estado final: needs_human
```

A 1ª passagem corre todos; a 2ª corre só o violador do contrato; as seguintes
correm só o NEURON visado por melhoria. Exactamente o comportamento
especificado.

## Veredicto

**O LangGraph exprime o fluxo completo sem contorções.**

As três acomodações necessárias não são distorções da lógica — são a forma
natural de exprimir num grafo o que estava num `for`:

1. **`recursion_limit` explícito.** Cada iteração atravessa cerca de uma
   dezena de nós, e o valor por omissão do LangGraph (25) não chega. É um
   parâmetro de configuração documentado, calculado a partir de
   `MUNDJI_MAX_ITERATIONS`.
2. **Nó `nova_iteracao`.** O contador de iterações passa a ser um nó em vez
   de uma variável de loop. Num grafo cíclico é onde deve estar.
3. **Nó `so_o_violador`.** Já era lógica existente (estava inline no `run()`),
   apenas passou a ter nome próprio.

## Direcção tomada

Unificação no LangGraph. `MindGraph.run()` deixou de reimplementar o fluxo e
passa a invocar o grafo compilado:

```python
def run(self, state):
    app = self._app or self.build_langgraph()
    self._app = app
    limite = max(60, self.max_iterations * 20 + 60)
    return dict(app.invoke(state, config={"recursion_limit": limite}))
```

Os métodos `_phase1`, `_phase2` e `_phase3` foram removidos. **Existe uma
única implementação executável do ciclo.**

## Notas de implementação

Durante o teste técnico apareceram dois falsos negativos que valem registo,
porque ambos eram defeitos do próprio teste e não do sistema:

- **O circuit breaker bloqueava o ciclo.** O NEURON de ensaio excedia o
  timeout em *todas* as rondas, o que é tratado como violação de contrato —
  correctamente — e o ciclo nunca chegava à Fase 3. Passou a falhar apenas na
  primeira ronda.
- **O rollback é invisível a olho nu.** Quando a melhor versão é idêntica à
  actual, a substituição acontece na mesma mas não se vê comparando o código.
  Passou a ser detectado pelo registo na SYNAPSE DB, que é onde fica a prova.
