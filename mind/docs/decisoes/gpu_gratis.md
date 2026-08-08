# GPU grátis para o piloto — o que existe mesmo

Reavaliação feita a 8 de Agosto de 2026, a pedido, depois de uma primeira
resposta minha demasiado apressada.

## O que eu disse antes, e estava errado

Arrumei as opções gratuitas em «notebooks que não servem» e dei a questão
por fechada. **Estava errado.** Duas plataformas são bastante melhores do
que isso:

- **Lightning AI** — 80 h/mês num *workspace persistente* estilo VS Code,
  sem cartão. Não é notebook; é o que mais se aproxima de «VPS com GPU».
- **Kaggle** — 30 h/semana garantidas, sessões de 9 h. Chega e sobra para
  um piloto.

O que continua a não existir é uma **GPU dedicada sempre ligada e grátis
para sempre**. Isso não é limitação de mercado: uma GPU dedicada custa ao
fornecedor por hora, esteja ou não a ser usada. O que se dá de graça é
tempo limitado ou hardware partilhado.

## As três rotas, sem embelezar

| Rota | Custo | Setup | Adequação ao MIND |
|---|---|---|---|
| **API com nível grátis** (OpenRouter, Groq, NVIDIA NIM) | 0 € | ~2 min | **Melhor.** O modo `openai_compat` já fala este dialecto. Zero alterações de código |
| **GPU grátis + vLLM** (Lightning, Kaggle) | 0 € | ~30-60 min | Funciona. Dá controlo total do modelo, incluindo checkpoints afinados |
| **GPU alugada** (vast.ai) | ~0,2-2 €/h | ~20 min | Só se faz falta hardware maior ou modelos privados |

## Limites que interessam ao MIND

Um ciclo do MIND faz **10 a 20 chamadas** ao modelo (9 medidas ao CORTEX e
ao CEREBELLUM, mais uma por NEURON activo por iteração). Três tarefas do
piloto ficam na ordem das 50-75 chamadas.

- **Pedidos por dia**: 1000-1500 nos níveis grátis típicos. Folgado.
- **Pedidos por minuto**: 15-30. **Isto estrangula** — o MIND dispara os
  NEURONS em paralelo com `asyncio.gather()`. O retry com backoff em 429 já
  está implementado e testado, portanto não parte; fica lento.
- **Atenção ao SambaNova**: 20 pedidos/dia não dá para um ciclo.

## Armadilhas da rota GPU (já resolvidas no script)

`scripts/provisionar_gpu_gratis.py` cobre as quatro que custam mais tempo:

1. **A P100 do Kaggle não serve.** O vLLM exige capacidade de computação
   >= 7.0 e a P100 é 6.0. Tem de se escolher «GPU T4 x2». O script verifica
   e diz-o, em vez de rebentar a meio de uma instalação de 10 minutos.
2. **A T4 não tem bfloat16** (capacidade 7.5). Sem `--dtype half` o vLLM
   aborta ou cai para float32 e não cabe na memória.
3. **16 GB não chegam** para 7B em fp16 com contexto grande. O
   `--max-model-len` é dimensionado ao que sobra depois dos pesos.
4. **As sessões terminam** e o URL do túnel muda a cada arranque.

## Segurança — a diferença face à GPU alugada

Estas plataformas não dão SSH, por isso o túnel é **cloudflared**, que
produz um **URL público**. Isso é uma diferença de segurança, não de
conveniência: sem token, quem descobrir o URL usa a GPU e vê os prompts.

O script arranca sempre com `--api-key` e gera um se não for dado. O URL ser
efémero e não indexado é obscuridade, não segurança.

O princípio da SYNAPSE DB mantém-se intacto em todas as rotas: **os dados
operacionais nunca saem da máquina local.** O que sai são os prompts — tal
como sairiam para a GPU alugada, e tal como saem no modo `anthropic`.

## Recomendação

**Para validar o MIND, a rota da API é objectivamente melhor**: mesmo
resultado, 2 minutos em vez de uma hora, e sem sessões a expirar a meio de
um piloto. Os modelos disponíveis (`cohere/north-mini-code`,
`nvidia/nemotron-3-super-120b`, `google/gemma-4-31b-it`) são open-weight —
ou seja, mais fiéis à especificação original do que o modo `anthropic` era.

**A rota da GPU justifica-se** quando se quiser servir checkpoints afinados
ou adaptadores LoRA próprios, que é o passo seguinte natural depois de
haver dados na SYNAPSE DB. Aí o controlo do servidor deixa de ser luxo.

As duas estão preparadas. A escolha é de conveniência, não de capacidade.
