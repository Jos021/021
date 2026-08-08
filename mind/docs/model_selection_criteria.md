# Critério de Selecção de Modelo — MIND

Antes de fixar o modelo de **qualquer** componente (CORTEX, CEREBELLUM,
NEURON 1-6), seguir e documentar aqui o processo abaixo. Os campos `_MODEL`
ficam **vazios** por defeito no `.env` — só se preenchem depois de a decisão
estar documentada nesta página.

## Processo (obrigatório)

1. **Pesquisar no HuggingFace** se existe modelo já especializado na função
   exacta do componente.
2. **Avaliar contra critérios:**
   - Benchmark relevante à função real do componente
   - Licença de uso comercial
   - Tamanho compatível com o hardware disponível
   - Compatibilidade de idioma/formato (português europeu; formato de output)
   - Comunidade activa (manutenção, issues, downloads)
   - Resultado em teste-piloto próprio (5-10 tarefas reais)
3. **Decidir:**
   - Usar directamente se passar nos critérios; **senão**,
   - Usar modelo genérico forte sem fine-tuning, marcado como **candidato a
     fine-tuning futuro** quando houver dados suficientes na SYNAPSE DB
     (exportáveis via `python main.py export`).

## Contexto de execução (decidido)

Os modelos são **todos do HuggingFace**, alguns **retreinados**, e correm em
**GPU alugada na cloud** (vast.ai e semelhantes). Ver
`docs/decisoes/gpu_alugada.md` para os modos de serviço, a nota de segurança
e a recomendação de usar adaptadores LoRA em vez de checkpoints completos.

Consequências para a escolha de modelo:

- **A licença tem de permitir uso comercial E execução em hardware alugado
  de terceiros.** Algumas licenças restringem onde os pesos podem correr.
- **O tamanho é limitado pela GPU que se aluga**, não por hardware próprio —
  o critério passa a ser custo/hora versus qualidade, e não "cabe ou não
  cabe".
- **Preferir modelos com o mesmo base entre componentes** sempre que possível:
  permite servir variantes afinadas como adaptadores LoRA sobre um só modelo
  carregado, em vez de pagar memória por cada um.
- **Verificar se o modelo respeita instruções de formato** (o esquema JSON dos
  relatórios). É um critério prático que só se mede em teste-piloto, e a
  SYNAPSE DB regista cada desvio automaticamente.

## Princípio de dimensionamento

Os modelos **não** são todos do mesmo tamanho por padrão — cada componente
recebe um modelo dimensionado ao esforço cognitivo real da sua função:

- **CORTEX** = maior modelo disponível (orquestração, decisão, persona)
- **CEREBELLUM** = médio-alto (auditoria, validação cruzada)
- **NEURONS** = dimensionados caso a caso, conforme a complexidade real da
  parte que cada um gera

## Treino local ou cloud

Independente de onde o MIND corre em produção — pode-se treinar na cloud e
servir localmente, ou o inverso. A SYNAPSE DB e todos os dados operacionais
ficam **sempre locais**; só datasets exportados e anonimizados poderiam
alimentar treino externo, por decisão explícita do operador.

---

## Modelos indicados para o piloto (8 de Agosto de 2026)

Escolha do operador: variantes de **Qwen2.5-Coder-7B** com restrições de
segurança removidas (heretic / abliterated / uncensored). É uma decisão de
domínio — a Muñdji gera ferramentas de segurança ofensiva, e um modelo que
recusa esse tipo de código não serve para o efeito. A qualidade destas
variantes comunitárias é desconhecida e é precisamente o que o piloto mede.

Todos os IDs foram **verificados por API** antes de entrarem no `.env`. Duas
correcções face à lista indicada:

| Componente | ID (verificado) | Nota |
|---|---|---|
| CORTEX | `saidutta69/Qwen2.5-Coder-7B-Instruct-heretic` | ok |
| CEREBELLUM | `huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated` | ok |
| NEURON 1 | `GodsDevProject/qwe2.5-coder-Uncensored` | ok (o `qwe2.5` é o nome real do repo) |
| NEURON 2 | `Qwen/Qwen2.5-Coder-7B-Instruct` | **provisório** — `PyCDistill` não existe |
| NEURON 3 | `saidutta69/Qwen2.5-Coder-7B-Instruct-heretic` | ok |
| NEURON 4 | `vanta-research/wraith-coder-7b` | corrigido (faltava a org) |
| NEURON 5 | `Qwen/Qwen2.5-Coder-7B-Instruct` | **provisório** — `PyCDistill` não existe |
| NEURON 6 | `huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated` | ok |
| HIPPOCAMPUS | `sentence-transformers/all-MiniLM-L6-v2` | ok (é o default do código) |

**`Qwen/Qwen2.5-Coder-7B-PyCDistill` não existe** no HuggingFace — a Qwen
nunca o publicou e a busca por "PyCDistill" no Hub inteiro não devolve nada.
Fica o base Instruct como substituto assinalado até haver um ID real.

### Restrição de serviço que decide o arranque do piloto

São **cinco modelos distintos** de 7B (heretic, abliterated, uncensored,
wraith, base). Cada um ocupa ~15 GB em fp16 — no total ~75 GB. **Não cabem
em simultâneo em nenhuma GPU grátis** (T4 tem 16 GB, uma A100 tem 40 GB). O
modo `openai_compat`/vLLM serve **um modelo por servidor**.

Isto tem de se decidir antes de arrancar, e não é limitação do MIND — é
memória de GPU. As opções estão descritas em `docs/decisoes/gpu_gratis.md`.
A regra do dimensionamento por componente mantém-se como objectivo; a
diferenciação real entre os oito só se mede quando houver GPU que os sirva.

---

## Ficha por componente (campos por preencher)

### CORTEX
- Função exacta: orquestração, distribuição, decisão (com persona JARVIS)
- Candidatos HuggingFace pesquisados:
- Benchmark relevante:
- Licença:
- Tamanho / hardware:
- Idioma / formato:
- Comunidade:
- Resultado do teste-piloto (5-10 tarefas):
- Respeita o esquema JSON dos relatórios? (% de respostas válidas):
- Modelo base (para partilha de LoRA entre componentes):
- **Decisão:**
- Candidato a fine-tuning futuro? (sim/não):

### CEREBELLUM
- Função exacta: auditoria técnica, validação cruzada, cálculo de %
- Candidatos HuggingFace pesquisados:
- Benchmark relevante:
- Licença:
- Tamanho / hardware:
- Idioma / formato:
- Comunidade:
- Resultado do teste-piloto (5-10 tarefas):
- Respeita o esquema JSON dos relatórios? (% de respostas válidas):
- Modelo base (para partilha de LoRA entre componentes):
- **Decisão:**
- Candidato a fine-tuning futuro? (sim/não):

### NEURON 1
- Especialidade:
- Candidatos HuggingFace pesquisados:
- Benchmark relevante:
- Licença:
- Tamanho / hardware:
- Idioma / formato:
- Comunidade:
- Resultado do teste-piloto (5-10 tarefas):
- Respeita o esquema JSON dos relatórios? (% de respostas válidas):
- Modelo base (para partilha de LoRA entre componentes):
- **Decisão:**
- Candidato a fine-tuning futuro? (sim/não):

### NEURON 2
- Especialidade:
- Candidatos HuggingFace pesquisados:
- Benchmark relevante:
- Licença:
- Tamanho / hardware:
- Idioma / formato:
- Comunidade:
- Resultado do teste-piloto (5-10 tarefas):
- Respeita o esquema JSON dos relatórios? (% de respostas válidas):
- Modelo base (para partilha de LoRA entre componentes):
- **Decisão:**
- Candidato a fine-tuning futuro? (sim/não):

### NEURON 3
- Especialidade:
- Candidatos HuggingFace pesquisados:
- Benchmark relevante:
- Licença:
- Tamanho / hardware:
- Idioma / formato:
- Comunidade:
- Resultado do teste-piloto (5-10 tarefas):
- Respeita o esquema JSON dos relatórios? (% de respostas válidas):
- Modelo base (para partilha de LoRA entre componentes):
- **Decisão:**
- Candidato a fine-tuning futuro? (sim/não):

### NEURON 4
- Especialidade:
- Candidatos HuggingFace pesquisados:
- Benchmark relevante:
- Licença:
- Tamanho / hardware:
- Idioma / formato:
- Comunidade:
- Resultado do teste-piloto (5-10 tarefas):
- Respeita o esquema JSON dos relatórios? (% de respostas válidas):
- Modelo base (para partilha de LoRA entre componentes):
- **Decisão:**
- Candidato a fine-tuning futuro? (sim/não):

### NEURON 5
- Especialidade:
- Candidatos HuggingFace pesquisados:
- Benchmark relevante:
- Licença:
- Tamanho / hardware:
- Idioma / formato:
- Comunidade:
- Resultado do teste-piloto (5-10 tarefas):
- Respeita o esquema JSON dos relatórios? (% de respostas válidas):
- Modelo base (para partilha de LoRA entre componentes):
- **Decisão:**
- Candidato a fine-tuning futuro? (sim/não):

### NEURON 6
- Especialidade:
- Candidatos HuggingFace pesquisados:
- Benchmark relevante:
- Licença:
- Tamanho / hardware:
- Idioma / formato:
- Comunidade:
- Resultado do teste-piloto (5-10 tarefas):
- Respeita o esquema JSON dos relatórios? (% de respostas válidas):
- Modelo base (para partilha de LoRA entre componentes):
- **Decisão:**
- Candidato a fine-tuning futuro? (sim/não):
