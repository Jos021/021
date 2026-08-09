# Recusa de gateways de modelos de terceiros

Decisão de segurança, 8 de Agosto de 2026.

## O que foi recusado

Ligar o MIND a um *gateway* de revenda de modelos alojado num domínio de DNS
dinâmico gratuito (`*.hopto.org`), com chave no formato `sk-gtw-...`. A
própria página do serviço avisava: *"DONT USE COMMUNITY TAB IT CONTAINS IP
GRABBER"*.

## Porque é inegociável — não é melindre, é a arquitectura

O risco não é o dos prompts saírem (isso já acontece com qualquer inferência
remota). É específico do que o MIND faz:

1. **A sandbox do MIND não é uma fronteira de segurança.** Corre o código
   gerado como subprocesso normal, no mesmo container, com timeout — não
   isola privilégios, rede nem sistema de ficheiros.
2. **O container tem credenciais.** Chaves em `.env` e acesso de escrita ao
   repositório GitHub da sessão.
3. **Um gateway controla as respostas dos modelos.** É um homem-no-meio por
   desenho — o `sk-gtw-` diz isso mesmo.

Encadeado: o gateway injecta código na resposta de um NEURON → o MIND
compila-o e executa-o → código de um terceiro corre com acesso às
credenciais. É comprometimento da cadeia de fornecimento no pior sítio: no
ponto de execução.

## Porque "conheço o criador" não fecha o risco

A confiança na pessoa cobre a intenção. Não cobre: o servidor dele ser
comprometido, a plataforma ter componentes maliciosos admitidos (o
"IP grabber" da própria página), e — o essencial — que confiar numa pessoa
não é o mesmo que confiar na infraestrutura dela para **executar código com
as tuas credenciais**.

## Critério permanente

O endpoint de inferência tem de ser do **próprio fabricante do modelo**:
`api.deepseek.com`, `api.anthropic.com`, `api.groq.com`, `openrouter.ai`, ou
um servidor **próprio** (vLLM numa GPU controlada pelo operador). Nunca um
revendedor/proxy intermédio, por mais conveniente que seja.

## O que fica permitido, mesmo com fonte não fiável

Um `python main.py verificar` — round-trip de texto que **não executa** o que
recebe — é de baixo risco e serve para diagnosticar conectividade. O que não
se faz com fonte não fiável é correr ciclos, porque é aí que o código
devolvido é executado.
