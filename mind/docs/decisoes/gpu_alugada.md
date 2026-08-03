# Modelos HuggingFace em GPU alugada

**Data:** 3 de Agosto de 2026
**Estado:** implementado (router) · avaliação de fine-tuning para decisão

## O contexto que mudou

O Ollama sai de cena. Os modelos passam a ser **todos do HuggingFace**,
alguns **retreinados**, e correm em **GPU alugada na cloud** (vast.ai e
semelhantes). O `model_router.py` assumia Ollama por omissão e foi
reescrito.

## Modos suportados

Os quatro coexistem atrás do mesmo `ModelRouter` e escolhem-se por
`MODEL_MODE`. O resto do sistema continua sem saber qual está activo.

| Modo | Quando usar | Notas |
|---|---|---|
| `openai_compat` **(omissão)** | Servidor vLLM / TGI / llama.cpp na GPU alugada | O servidor gere a GPU; o MIND só fala HTTP. Não precisa de torch. |
| `hf_local` | O MIND corre **na própria máquina** da GPU | `transformers` em processo. Aceita ids do Hub e caminhos de disco. Exige `requirements-hf.txt`. |
| `hf_api` | Inference API / Inference Endpoints da HuggingFace | Não exige GPU; os dados saem para a HF. |
| `ollama` | Instalações locais existentes | Mantido por compatibilidade. |

O campo `*_MODEL` aceita indistintamente **id do Hub**
(`Qwen/Qwen2.5-Coder-32B-Instruct`), **nome de adaptador LoRA servido**
(`cortex-lora-v3`) ou **caminho de disco** (`/models/finetuned/cortex`).

## Segurança: o que muda com GPU alugada

Uma instância alugada é uma máquina que controlas, mas que está no
datacenter de outra pessoa. Consequência directa e inevitável: **os prompts
e o código gerado saem do teu hardware.** O princípio "tudo local" da
SYNAPSE DB mantém-se — a base de dados nunca sai — mas a inferência deixa de
ser local. É uma cedência consciente, não um descuido, e fica registada aqui
como tal.

O que está implementado para reduzir o risco:

- **Token por componente** (`CORTEX_TOKEN`, `NEURON_N_TOKEN`, ou o global
  `MODEL_AUTH_TOKEN`). Um servidor de inferência exposto sem autenticação é
  uma GPU alugada a pagar por ti para servir outra pessoa — e um registo dos
  teus prompts nas mãos dela.
- **Verificação de TLS activa** por omissão (o `httpx` não a desliga).
- **Componentes podem apontar para servidores diferentes**, o que permite
  pôr o CORTEX numa instância maior e os NEURONS noutra mais barata.
- **Retry com backoff**, que passa a ser mais relevante do que era: uma
  instância alugada pode ser reiniciada ou ficar momentaneamente
  indisponível, e isso não deve reprovar um ciclo.

Recomendações operacionais que **não** estão no código porque são de
infraestrutura:

- Não expor o servidor de inferência à internet aberta. Preferir túnel SSH
  ou WireGuard entre a tua máquina e a instância, com o servidor a escutar
  apenas em `localhost` do lado remoto.
- Se tiver de ser exposto, HTTPS com certificado real e token obrigatório.
- Assumir que o disco da instância alugada é **efémero e não confiável**:
  não deixar lá segredos, e considerar que qualquer coisa escrita ali pode
  ser lida por quem opera o hardware.

## Avaliação pedida: onde devem viver os modelos retreinados

O contexto de GPU alugada muda a resposta, porque o disco da instância é
efémero — pode ser reclamado, e cada nova instância começa vazia.

### Recomendação: **HF Hub privado como fonte de verdade, disco como cache**

| Opção | A favor | Contra | Veredicto |
|---|---|---|---|
| Só disco da instância alugada | Rápido a carregar | **Perde-se com a instância.** Cada nova máquina exige voltar a treinar ou a copiar | Inviável como única cópia |
| Só disco da tua máquina | Durável e sob teu controlo | Tens de enviar dezenas de GB para cada instância nova, de cada vez | Viável mas doloroso |
| **Repo privado no HF Hub** | Durável, versionado, e uma instância nova puxa-o com um comando | Os pesos passam pela HF | **Recomendado** |

O argumento decisivo: com GPU alugada vais destruir e recriar instâncias com
frequência. A fonte de verdade tem de estar num sítio de onde uma máquina
nova a consiga puxar sozinha. Um repo privado no Hub faz isso; um disco
efémero não.

Sobre o "os pesos passam pela HF": os pesos de um modelo afinado sobre os
teus ciclos são menos sensíveis do que os dados que os geraram — e esses
(a SYNAPSE DB) continuam a nunca sair. Se mesmo assim for inaceitável, a
alternativa coerente é armazenamento próprio (S3 privado, MinIO na tua
rede), não o disco da instância.

**O router já suporta as três opções sem código adicional**, porque o campo
`*_MODEL` é só uma string que o `transformers` ou o servidor de inferência
resolvem. A decisão é de operação, não de implementação.

### Recomendação técnica: LoRA em vez de checkpoints completos

Com 8 componentes e uma GPU alugada, servir 8 modelos grandes distintos é
caro ou impossível. Se os modelos afinados forem **adaptadores LoRA sobre um
mesmo modelo base**:

- o vLLM serve vários adaptadores em simultâneo sobre um só modelo carregado,
  com custo de memória quase nulo por adaptador;
- cada adaptador é exposto como um nome de modelo distinto, portanto
  `CORTEX_MODEL=cortex-lora-v3` e `NEURON_1_MODEL=neuron1-lora-v2` funcionam
  sem qualquer alteração ao MIND;
- os adaptadores pesam megabytes em vez de gigabytes, o que torna o repo
  privado no Hub barato e rápido de puxar;
- o retreino a partir dos dados da SYNAPSE DB (via `python main.py export`)
  produz adaptadores, não modelos inteiros.

É a combinação que melhor encaixa no que já está construído: o
`export_to_jsonl` existe precisamente para alimentar este ciclo, e o
princípio de dimensionamento por componente continua a poder ser respeitado
escolhendo bases diferentes onde fizer sentido.

## O que isto não resolve

O sistema continua **sem nunca ter corrido com um modelo verdadeiro**. Ter o
router pronto para HuggingFace não substitui o piloto: só com uma instância
a responder é que se sabe se os modelos respeitam o esquema JSON dos
relatórios, quanto tempo demora um ciclo, e se os `NEURON_TIMEOUT_SECONDS`
actuais são realistas. A SYNAPSE DB já regista cada desvio de formato, por
isso o piloto vai produzir essa resposta directamente.
