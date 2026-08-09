# Integrar as IAs no MIND — do zero aos 100%, com o MIND ON

Guia operacional. Leva de uma máquina sem nada a um MIND a responder a
tarefas com modelos reais. Os modelos aqui documentados são a escolha do
operador para esta fase; o MIND em si é agnóstico a eles — troca-se de
modelo mudando o `.env`, sem tocar em código.

> **Princípio que não muda:** a SYNAPSE DB e todos os dados operacionais
> ficam **sempre locais**. O que sai da tua máquina são os prompts e o
> código gerado, que vão para os servidores de inferência. É a mesma cedência
> da GPU alugada — consciente, não descuidada.

---

## 0. Os modelos a usar

Todos verificados por API no HuggingFace (8 Ago 2026). São variantes de
`Qwen2.5-Coder-7B` com as restrições de segurança removidas — escolha de
domínio, para gerar ferramentas de segurança sem recusas.

| Componente | Modelo (ID no HuggingFace) |
|---|---|
| CORTEX | `saidutta69/Qwen2.5-Coder-7B-Instruct-heretic` |
| CEREBELLUM | `huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated` |
| NEURON 1 | `GodsDevProject/qwe2.5-coder-Uncensored` |
| NEURON 2 | `Qwen/Qwen2.5-Coder-7B-Instruct` *(provisório — ver nota)* |
| NEURON 3 | `saidutta69/Qwen2.5-Coder-7B-Instruct-heretic` |
| NEURON 4 | `vanta-research/wraith-coder-7b` |
| NEURON 5 | `Qwen/Qwen2.5-Coder-7B-Instruct` *(provisório — ver nota)* |
| NEURON 6 | `huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated` |
| HIPPOCAMPUS | `sentence-transformers/all-MiniLM-L6-v2` |

**Duas notas verificadas:**
- **NEURON 2 e 5**: o ID pedido `Qwen/Qwen2.5-Coder-7B-PyCDistill` **não
  existe** no HuggingFace. Fica o base `Qwen2.5-Coder-7B-Instruct` até haver
  um ID real.
- **NEURON 4**: o ID real leva a org — `vanta-research/wraith-coder-7b`.

**São 5 modelos distintos** (heretic, abliterated, uncensored, wraith,
base). Cada um ocupa ~15 GB em fp16 → ~75 GB no total. **Não cabem numa só
GPU.** Por isso são **5 servidores**, um por modelo, e os componentes que
partilham modelo apontam para o mesmo. É a decisão "várias contas / VPs com
GPU dedicada".

**Aviso de compatibilidade:** os processos abliterated/heretic/uncensored
podem degradar a aderência ao formato. O MIND depende de duas disciplinas —
os marcadores `[NEURON_N:linguagem]` e o esquema JSON dos relatórios. Se os
modelos divagarem, os ciclos custam mais iterações. O MIND tem redes
(parsing por recurso, degradação graciosa) e **mede** os desvios de formato.
É o principal risco a vigiar no arranque.

---

## 1. Os cinco servidores

| Servidor | Modelo | Serve os componentes |
|---|---|---|
| **A** | `saidutta69/...heretic` | CORTEX, NEURON 3 |
| **B** | `huihui-ai/...abliterated` | CEREBELLUM, NEURON 6 |
| **C** | `GodsDevProject/qwe2.5-coder-Uncensored` | NEURON 1 |
| **D** | `vanta-research/wraith-coder-7b` | NEURON 4 |
| **E** | `Qwen/Qwen2.5-Coder-7B-Instruct` | NEURON 2, NEURON 5 |

Cada servidor precisa de uma GPU com **capacidade de computação ≥ 7.0** e
≥ 16 GB de VRAM (T4, A10, A100…). **A P100 do Kaggle (6.0) não serve** —
escolhe "GPU T4 x2".

---

## 2. Levantar cada servidor

Em cada sessão de GPU (Kaggle, Lightning AI, Colab, ou VPS alugada), corre
`scripts/provisionar_gpu_gratis.py` uma vez, indicando o modelo e os
componentes que ele serve. Exemplo para o **Servidor A**:

```bash
MODELO=saidutta69/Qwen2.5-Coder-7B-Instruct-heretic \
COMPONENTES="CORTEX NEURON_3" \
python scripts/provisionar_gpu_gratis.py
```

O script:
1. verifica a GPU (recusa a P100 com explicação),
2. instala o vLLM,
3. arranca o servidor com `--dtype half` na T4 e `--max-model-len`
   dimensionado à VRAM,
4. abre um túnel cloudflared (URL público — protegido por `--api-key`),
5. **imprime as linhas exactas do `.env`** para este servidor:

```
# Servidor de saidutta69/Qwen2.5-Coder-7B-Instruct-heretic
CORTEX_ENDPOINT=https://xxxx.trycloudflare.com
CORTEX_TOKEN=<gerado>
NEURON_3_ENDPOINT=https://xxxx.trycloudflare.com
NEURON_3_TOKEN=<gerado>
```

Repete para B, C, D, E, mudando `MODELO` e `COMPONENTES`:

| Servidor | `MODELO=` | `COMPONENTES=` |
|---|---|---|
| B | `huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated` | `"CEREBELLUM NEURON_6"` |
| C | `GodsDevProject/qwe2.5-coder-Uncensored` | `"NEURON_1"` |
| D | `vanta-research/wraith-coder-7b` | `"NEURON_4"` |
| E | `Qwen/Qwen2.5-Coder-7B-Instruct` | `"NEURON_2 NEURON_5"` |

**Deixa cada célula/sessão a correr.** Se parar, o túnel fecha e o URL muda.

---

## 3. Ligar ao MIND (`.env`)

Na tua máquina, no `mind/.env`:

```
MODEL_MODE=openai_compat
```

Depois cola os blocos que os 5 servidores imprimiram. No fim, o `.env` tem
os 8 pares endpoint/token preenchidos, mais os modelos:

```
CORTEX_MODEL=saidutta69/Qwen2.5-Coder-7B-Instruct-heretic
CEREBELLUM_MODEL=huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated
NEURON_1_MODEL=GodsDevProject/qwe2.5-coder-Uncensored
NEURON_2_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
NEURON_3_MODEL=saidutta69/Qwen2.5-Coder-7B-Instruct-heretic
NEURON_4_MODEL=vanta-research/wraith-coder-7b
NEURON_5_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
NEURON_6_MODEL=huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated
```

O `.env` **nunca é versionado** (está no `.gitignore`) — os tokens e URLs
ficam só na tua máquina.

---

## 4. Verificar antes de gastar um ciclo

```bash
python main.py verificar
```

Testa a ligação aos 8 componentes com um prompt mínimo. Um URL errado ou um
token em falta descobre-se aqui em segundos. Só se avança quando estiver:

```
cortex       OK   ...  -> 'OK'
cerebellum   OK   ...
neuron_1..6  OK   ...
8 a responder, 0 com falha, 0 sem modelo configurado.
```

Se algum falhar, a mensagem traz a razão real da API (não um "400" cego).
**Reportar com o log antes de mexer** — não ajustar às cegas.

---

## 5. Pôr o MIND a trabalhar — ON

O MIND é uma ferramenta de linha de comando: corre uma tarefa até ao fim e
termina. "ON" significa **os 5 servidores de pé + `verificar` verde + o MIND
pronto a aceitar tarefas**.

**Uma tarefa:**
```bash
python main.py "cria um scanner de portas TCP com timeout por porta"
```
O CORTEX cria e distribui, os NEURONS implementam a sua secção, o CEREBELLUM
audita, a sandbox executa, e o ciclo repete até ao limiar de aprovação. O
resultado fica em `workspace/output/`.

**O piloto (mede vários casos de referência):**
```bash
python main.py piloto --max-tarefas 3    # primeira passagem
python main.py piloto                     # as 10 tarefas
```
Mede duração, iterações, conformidade JSON e resultados de testes; exporta
`datasets/piloto.csv`.

---

## 6. O que fica ON, e o que corre on-demand

| Peça | Estado |
|---|---|
| **5 servidores de modelo** | **ON contínuo** (nas GPUs). Se caem, o MIND não tem com quem falar |
| **SYNAPSE DB** | Ficheiro local; persiste entre execuções. Sempre disponível |
| **Backup (APScheduler)** | Corre em paralelo enquanto o MIND está a executar; escreve cópias locais. Confirma que os dados descem antes de desligar as GPUs |
| **O ciclo do MIND** | **On-demand**: arranca por tarefa (`python main.py "..."`) e termina. Não é um daemon |
| **HIPPOCAMPUS (ML)** | Desligado (`ML_ENABLED=false`) até haver ~100 ciclos reais na SYNAPSE DB. Até lá, o histórico acumula na mesma |

Se quiseres o MIND a atender pedidos em contínuo (um serviço em vez de um
comando por tarefa), isso é uma camada de servidor à volta do `main.py` que
**ainda não existe** — a interface é só CLI nesta fase, por decisão de
âmbito. É o passo natural depois de o piloto validar o comportamento.

---

## 7. Ordem recomendada de arranque

1. Levanta **um** servidor (o A) e faz `verificar` só com o CORTEX e o
   NEURON 3 configurados — prova a cadeia com um modelo antes de acender
   cinco GPUs.
2. Corre **uma** tarefa simples. Observa se os marcadores e o JSON saem
   limpos. É aqui que o risco dos modelos uncensored se revela ou não.
3. Se estável, levanta os outros quatro servidores, preenche o `.env`
   inteiro, `verificar` a 8/8.
4. `piloto --max-tarefas 3`, depois o piloto completo.
5. Consulta a conformidade JSON acumulada:
   ```bash
   python main.py ml-status
   ```
   É o número que diz se estes modelos cumprem o contrato de formato. Só
   depois de medido é que se preenchem as fichas de
   `docs/model_selection_criteria.md` — nunca antes.

O MIND fica ON quando os servidores estão de pé e `verificar` está verde.
Daí em diante, cada `python main.py "<tarefa>"` é um ciclo completo com
modelos reais.
