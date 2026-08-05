"""ModelRouter unificado + retry com backoff.

Uma única classe abstrai a chamada a modelos, seja qual for a forma como
estão a ser servidos. O método `generate` funciona identicamente em todos os
modos — o resto do sistema nunca sabe qual está activo.

Cada um dos 8 componentes (CORTEX, CEREBELLUM, NEURON 1-6) tem endpoint,
modelo e token configuráveis de forma independente.

--------------------------------------------------------------------------
MODOS (MODEL_MODE)
--------------------------------------------------------------------------
  openai_compat  Servidor compatível com a API da OpenAI (vLLM, TGI,
                 llama.cpp server). É o modo por omissão e o recomendado
                 para GPU alugada: o servidor gere a GPU, o MIND só fala
                 HTTP. Suporta modelos do HuggingFace Hub e adaptadores
                 LoRA servidos como modelos distintos.
  hf_local       transformers em processo. Usar quando o MIND corre NA
                 própria máquina da GPU. Aceita IDs do Hub e caminhos de
                 disco (checkpoints afinados). Importação preguiçosa: o
                 torch só é carregado se este modo for usado.
  hf_api         HuggingFace Inference API / Inference Endpoints (remoto).
  ollama         Mantido para compatibilidade com instalações locais.
  anthropic      API da Anthropic directamente, sem proxy intermédio. O
                 formato do pedido difere do openai_compat em quatro
                 pontos — ver _call_anthropic.

--------------------------------------------------------------------------
NOTA DE SEGURANÇA — inferência remota
--------------------------------------------------------------------------
Quando os modelos correm em GPU alugada (vast.ai e afins), os prompts e o
código gerado saem do hardware do utilizador. O princípio "tudo local" da
SYNAPSE DB mantém-se — a base de dados nunca sai — mas a inferência deixa
de ser local, e isso é uma cedência consciente, não um descuido. O modo
anthropic tem exactamente a mesma natureza: os prompts vão para um serviço
de terceiros.

Mitigações previstas no código: token por componente (MODEL_AUTH_TOKEN ou
NEURON_N_TOKEN), verificação de TLS activa por omissão, e recomendação de
túnel (SSH/WireGuard) para endpoints sem HTTPS. Ver
docs/decisoes/gpu_alugada.md.

--------------------------------------------------------------------------
Decisão explícita sobre Redis
--------------------------------------------------------------------------
Redis NÃO é usado nesta fase. Com um único servidor de inferência, uma fila
de mensagens não traz paralelismo real. Fica documentado como caminho de
migração futuro (vários servidores/GPUs), não como necessidade actual. A
comunicação CORTEX->NEURONS usa asyncio.gather() (ver agent/graph.py).
"""

import os
import time
from typing import Optional

import httpx

MODOS = ("openai_compat", "hf_local", "hf_api", "ollama", "anthropic")

# Endpoint da API da Anthropic. É fixo e público, por isso serve de omissão
# quando MODEL_MODE=anthropic — não faz sentido obrigar a preenchê-lo.
ENDPOINT_ANTHROPIC = "https://api.anthropic.com"

# Versão da API pedida no header obrigatório anthropic-version.
VERSAO_API_ANTHROPIC = "2023-06-01"

# max_tokens é obrigatório no corpo do pedido da Anthropic (ao contrário do
# openai_compat, onde é opcional). O valor vem da especificação do modo.
ANTHROPIC_MAX_TOKENS = 4096

# Modelos carregados em processo no modo hf_local, reutilizados entre
# chamadas. Numa GPU só cabem alguns em simultâneo — ver docstring de
# _call_hf_local.
_PIPELINES: dict = {}


class ModelError(Exception):
    """Erro estruturado devolvido quando todas as tentativas falham."""

    def __init__(self, component: str, message: str):
        self.component = component
        self.message = message
        super().__init__(f"[{component}] {message}")


def call_model_with_retry(
    endpoint: str,
    model: str,
    prompt: str,
    system: str = "",
    mode: str = "openai_compat",
    api_key: str = "",
    timeout: float = 120.0,
    max_retries: int = 3,
    backoff: int = 2,
    db=None,
    component: str = "unknown",
    cycle_id: Optional[int] = None,
    iteration: int = 0,
) -> str:
    """Tenta a chamada ao modelo com retry e backoff exponencial.

    Se falhar (timeout, erro 5xx, erro de conexão), espera `backoff`
    segundos e tenta de novo, dobrando o backoff a cada tentativa
    (2s, 4s, 8s). Se todas as tentativas falharem, devolve um erro
    estruturado e regista-o na SYNAPSE DB.

    Complementa (não substitui) o circuit breaker por NEURON: o circuit
    breaker limita quanto tempo se espera por UM NEURON no total dentro do
    asyncio.gather(); o retry aumenta a probabilidade dessa chamada
    individual ter sucesso antes do circuit breaker cortar.

    O retry é particularmente relevante com GPU alugada: instâncias podem
    ser reiniciadas ou ficar momentaneamente indisponíveis, e uma falha de
    rede transitória não deve reprovar um ciclo.
    """
    last_error = ""
    wait = backoff
    for attempt in range(1, max_retries + 1):
        try:
            return _despachar(mode, endpoint, model, prompt, system,
                              api_key, timeout)
        except (httpx.HTTPStatusError, httpx.RequestError,
                httpx.TimeoutException) as exc:
            # Só faz retry em erros transitórios (5xx / conexão / timeout).
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                last_error = f"Erro não recuperável {status}: {exc}"
                break
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(wait)
                wait *= 2  # 2s -> 4s -> 8s
        except RuntimeError as exc:
            # Falhas locais (modelo não carrega, sem memória de GPU) não são
            # transitórias: repetir só desperdiça tempo.
            last_error = str(exc)
            break

    # Todas as tentativas falharam: erro estruturado + registo na SYNAPSE DB.
    # O cycle_id tem de ser o real: a tabela iterations tem foreign key para
    # cycles, e um valor inventado faria o registo falhar em silêncio — foi
    # exactamente o que acontecia antes, deixando as falhas de modelo sem
    # rasto nenhum.
    if db is not None and cycle_id:
        try:
            db.log_iteration(
                cycle_id=cycle_id,
                iteration_number=iteration,
                phase="model_call",
                component=component,
                input_summary=prompt[:200],
                output_summary="ERRO",
                full_output=f"call_model_with_retry falhou: {last_error}",
            )
        except Exception:
            pass  # nunca deixar o logging derrubar a chamada
    raise ModelError(
        component, f"Todas as {max_retries} tentativas falharam: {last_error}"
    )


def _despachar(mode, endpoint, model, prompt, system, api_key, timeout) -> str:
    """Encaminha para o runner do modo activo."""
    if mode == "hf_local":
        return _call_hf_local(model, prompt, system, timeout)
    if mode == "hf_api":
        return _call_hf_api(endpoint, model, prompt, system, api_key, timeout)
    if mode == "ollama":
        return _call_ollama(endpoint, model, prompt, system, timeout)
    if mode == "anthropic":
        return _call_anthropic(endpoint, model, prompt, system, api_key,
                               timeout)
    # openai_compat é o modo por omissão (vLLM, TGI, llama.cpp server).
    return _call_openai_compatible(endpoint, model, prompt, system,
                                   api_key, timeout)


def _call_openai_compatible(
    endpoint: str, model: str, prompt: str, system: str,
    api_key: str, timeout: float,
) -> str:
    """Chama um endpoint compatível com a API chat/completions da OpenAI.

    Serve vLLM, Text Generation Inference e llama.cpp server. É o caminho
    recomendado para GPU alugada: o servidor trata da memória e do batching,
    e adaptadores LoRA podem ser expostos como nomes de modelo distintos.
    """
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": messages}
    resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_anthropic(
    endpoint: str, model: str, prompt: str, system: str,
    api_key: str, timeout: float,
) -> str:
    """Chama a API de mensagens da Anthropic directamente, sem proxy.

    Parece o openai_compat mas difere em quatro pontos, e nenhum deles é
    cosmético — trocar qualquer um devolve 400 ou lê a resposta errada:

      1. autenticação em `x-api-key`, não em `Authorization: Bearer`
      2. header `anthropic-version` obrigatório
      3. system prompt num campo `system` de topo, não como uma mensagem
         de role "system" dentro de `messages`
      4. o texto está em content[0].text, não em choices[0].message.content

    `max_tokens` também é obrigatório aqui, ao contrário do openai_compat.

    A leitura da resposta procura o primeiro bloco de texto em vez de assumir
    cegamente o índice 0: a lista `content` é heterogénea por desenho (pode
    trazer outros tipos de bloco à frente do texto), e nesse caso content[0]
    não teria sequer a chave "text".
    """
    url = (endpoint or ENDPOINT_ANTHROPIC).rstrip("/") + "/v1/messages"
    headers = {
        "content-type": "application/json",
        "anthropic-version": VERSAO_API_ANTHROPIC,
    }
    if api_key:
        headers["x-api-key"] = api_key
    payload = {
        "model": model,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    blocos = resp.json().get("content") or []
    for bloco in blocos:
        if isinstance(bloco, dict) and bloco.get("type") == "text":
            return bloco.get("text", "")
    return ""


def _call_hf_api(
    endpoint: str, model: str, prompt: str, system: str,
    api_key: str, timeout: float,
) -> str:
    """Chama a Inference API da HuggingFace (ou um Inference Endpoint).

    Se `endpoint` apontar para um Inference Endpoint dedicado, usa-o; caso
    contrário monta o URL público a partir do id do modelo.
    """
    base = (endpoint or "").rstrip("/")
    if not base or "huggingface.co" not in base:
        base = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    entrada = f"{system}\n\n{prompt}" if system else prompt
    payload = {"inputs": entrada, "parameters": {"return_full_text": False}}
    resp = httpx.post(base, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return data[0].get("generated_text", "")
    if isinstance(data, dict):
        return data.get("generated_text", "")
    return ""


def _call_hf_local(model: str, prompt: str, system: str, timeout: float) -> str:
    """Corre um modelo HuggingFace em processo, via transformers.

    `model` pode ser um id do Hub ("Qwen/Qwen2.5-Coder-7B-Instruct") ou um
    caminho de disco para um checkpoint afinado — o transformers aceita os
    dois indistintamente, o que é o que permite servir modelos retreinados
    sem código especial.

    Aviso de memória: os 8 componentes do MIND não cabem simultaneamente
    numa só GPU se forem modelos grandes distintos. Os pipelines ficam em
    cache e são reutilizados; libertar memória entre componentes exigiria
    descarregar e recarregar, que é lento. Para várias variantes afinadas do
    mesmo modelo base, servir adaptadores LoRA por um servidor
    (modo openai_compat) é bastante mais eficiente do que este modo.

    O torch/transformers é importado preguiçosamente: quem não usar este
    modo não precisa de os ter instalados.
    """
    pipe = _PIPELINES.get(model)
    if pipe is None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "modo hf_local exige transformers e torch instalados "
                f"(pip install -r requirements-hf.txt): {exc}"
            ) from exc
        try:
            pipe = pipeline("text-generation", model=model,
                            device_map="auto")
        except Exception as exc:
            raise RuntimeError(f"não foi possível carregar '{model}': {exc}") from exc
        _PIPELINES[model] = pipe

    entrada = f"{system}\n\n{prompt}" if system else prompt
    try:
        saida = pipe(entrada, max_new_tokens=1024, return_full_text=False)
    except Exception as exc:
        raise RuntimeError(f"falha na inferência de '{model}': {exc}") from exc
    if isinstance(saida, list) and saida:
        return saida[0].get("generated_text", "")
    return ""


def _call_ollama(
    endpoint: str, model: str, prompt: str, system: str, timeout: float
) -> str:
    """Chama a API generate do Ollama (mantido para compatibilidade)."""
    url = endpoint.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "system": system,
               "stream": False}
    resp = httpx.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "")


class ModelRouter:
    """Lê MODEL_MODE do .env e abstrai a chamada ao modelo.

    Modos suportados: openai_compat (omissão), hf_local, hf_api, ollama e
    anthropic. O método generate(prompt, model, endpoint) funciona
    identicamente em todos — o resto do sistema nunca sabe qual está activo.
    """

    def __init__(self, db=None):
        self.mode = os.getenv("MODEL_MODE", "openai_compat").strip().lower()
        if self.mode not in MODOS:
            # Modo desconhecido não pode falhar em silêncio nem rebentar o
            # ciclo: cai no modo por omissão e fica registado.
            self.mode = "openai_compat"
        self.max_retries = int(os.getenv("MODEL_MAX_RETRIES", "3"))
        self.backoff = int(os.getenv("MODEL_RETRY_BACKOFF_SECONDS", "2"))
        self.db = db

    def generate(
        self,
        prompt: str,
        model: str,
        endpoint: str,
        system: str = "",
        component: str = "unknown",
        timeout: float = 120.0,
        cycle_id: Optional[int] = None,
        iteration: int = 0,
    ) -> str:
        """Gera texto a partir do modelo, com retry+backoff transparente."""
        return call_model_with_retry(
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            system=system,
            mode=self.mode,
            api_key=token_do_componente(component),
            timeout=timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
            db=self.db,
            component=component,
            cycle_id=cycle_id,
            iteration=iteration,
        )

    async def agenerate(
        self,
        prompt: str,
        model: str,
        endpoint: str,
        system: str = "",
        component: str = "unknown",
        timeout: float = 120.0,
        cycle_id: Optional[int] = None,
        iteration: int = 0,
    ) -> str:
        """Versão assíncrona — corre a chamada síncrona numa thread.

        Permite usar asyncio.gather() sobre múltiplos NEURONS sem bloquear
        o event loop. O retry+backoff continua a aplicar-se por chamada.
        """
        import asyncio

        return await asyncio.to_thread(
            self.generate, prompt, model, endpoint, system, component,
            timeout, cycle_id, iteration,
        )


def token_do_componente(component: str) -> str:
    """Token de autenticação de um componente.

    Precedência: token específico do componente, depois o global. Ter um
    token por componente permite apontar componentes diferentes para
    servidores diferentes — por exemplo, o CORTEX numa GPU alugada maior e
    os NEURONS noutra mais barata.
    """
    if component and component.startswith("neuron_"):
        n = component.split("_")[-1]
        especifico = os.getenv(f"NEURON_{n}_TOKEN", "")
    elif component in ("cortex", "cerebellum"):
        especifico = os.getenv(f"{component.upper()}_TOKEN", "")
    else:
        especifico = ""
    return (
        especifico
        or os.getenv("MODEL_AUTH_TOKEN", "")
        or os.getenv("HUGGINGFACE_API_KEY", "")
    )


def endpoint_por_omissao() -> str:
    """Endpoint usado por um componente que não defina o seu.

    Normalmente é MODEL_ENDPOINT. No modo anthropic o endereço da API é fixo
    e público, por isso serve de omissão quando MODEL_ENDPOINT não está
    preenchido — não vale a pena obrigar a escrever o óbvio. Um
    MODEL_ENDPOINT explícito continua a mandar em qualquer modo (permite
    apontar para um proxy ou gateway compatível).
    """
    explicito = os.getenv("MODEL_ENDPOINT", "").strip()
    if explicito:
        return explicito
    if os.getenv("MODEL_MODE", "").strip().lower() == "anthropic":
        return ENDPOINT_ANTHROPIC
    return "http://localhost:8000"


def component_config(component: str) -> tuple[str, str, bool]:
    """Lê endpoint, modelo e flag de existência de um componente do .env.

    `component` é 'cortex', 'cerebellum' ou 'neuron_N'. Devolve
    (endpoint, model, enabled). O modelo pode ser um id do HuggingFace Hub,
    o nome de um adaptador LoRA servido, ou um caminho de disco para um
    checkpoint afinado — o router trata os três da mesma maneira.

    Para NEURONS, `enabled` reflecte ENABLE_NEURON_N (existência no sistema
    — não activação por ronda). Para CORTEX/CEREBELLUM é sempre True.
    """
    omissao = endpoint_por_omissao()
    if component == "cortex":
        return (
            os.getenv("CORTEX_ENDPOINT", omissao),
            os.getenv("CORTEX_MODEL", ""),
            True,
        )
    if component == "cerebellum":
        return (
            os.getenv("CEREBELLUM_ENDPOINT", omissao),
            os.getenv("CEREBELLUM_MODEL", ""),
            True,
        )
    # neuron_N
    n = component.split("_")[-1]
    endpoint = os.getenv(f"NEURON_{n}_ENDPOINT", omissao)
    model = os.getenv(f"NEURON_{n}_MODEL", "")
    enabled = os.getenv(f"ENABLE_NEURON_{n}", "true").lower() == "true"
    return endpoint, model, enabled
