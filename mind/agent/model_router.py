"""ModelRouter unificado + retry com backoff.

Uma única classe abstrai a chamada a modelos locais (Ollama) e em nuvem
(provedor compatível com OpenAI, ex: HuggingFace). O método `generate`
funciona identicamente em ambos os modos — o resto do sistema nunca sabe
qual modo está activo.

Cada um dos 8 componentes (CORTEX, CEREBELLUM, NEURON 1-6) tem endpoint e
modelo configuráveis de forma independente.

--------------------------------------------------------------------------
Decisão explícita sobre Redis
--------------------------------------------------------------------------
Redis NÃO é usado nesta fase. Se os NEURONS correm numa única GPU local,
uma fila de mensagens não traz paralelismo real algum. Fica aqui
documentado como caminho de migração futuro (múltiplas GPUs/máquinas),
não como necessidade actual. A comunicação CORTEX->NEURONS usa
asyncio.gather() (ver agent/graph.py); se os endpoints suportarem
concorrência real, o paralelismo é real; se partilharem GPU, a fila é
natural, sem custo extra de infraestrutura.
"""

import os
import time
from typing import Optional

import httpx


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
    mode: str = "local",
    api_key: str = "",
    timeout: float = 120.0,
    max_retries: int = 3,
    backoff: int = 2,
    db=None,
    component: str = "unknown",
) -> str:
    """Tenta a chamada ao modelo com retry e backoff exponencial.

    Se falhar (timeout, erro 5xx, erro de conexão), espera `backoff`
    segundos e tenta de novo, dobrando o backoff a cada tentativa
    (2s, 4s, 8s). Se todas as tentativas falharem, devolve um erro
    estruturado e regista-o na SYNAPSE DB.

    Complementa (não substitui) o circuit breaker por NEURON: o circuit
    breaker limita quanto tempo se espera por UM NEURON no total dentro
    do asyncio.gather(); o retry aumenta a probabilidade dessa chamada
    individual ter sucesso antes do circuit breaker cortar.
    """
    last_error = ""
    wait = backoff
    for attempt in range(1, max_retries + 1):
        try:
            if mode == "api":
                return _call_openai_compatible(
                    endpoint, model, prompt, system, api_key, timeout
                )
            return _call_ollama(endpoint, model, prompt, system, timeout)
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as exc:
            # Só faz retry em erros transitórios (5xx / conexão / timeout).
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                last_error = f"Erro não recuperável {status}: {exc}"
                break
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(wait)
                wait *= 2  # 2s -> 4s -> 8s

    # Todas as tentativas falharam: erro estruturado + registo na SYNAPSE DB.
    if db is not None:
        try:
            db.log_iteration(
                cycle_id=0,
                iteration_number=0,
                phase="model_call",
                component=component,
                input_summary=prompt[:200],
                output_summary="ERRO",
                full_output=f"call_model_with_retry falhou: {last_error}",
            )
        except Exception:
            pass  # nunca deixar o logging derrubar a chamada
    raise ModelError(component, f"Todas as {max_retries} tentativas falharam: {last_error}")


def _call_ollama(
    endpoint: str, model: str, prompt: str, system: str, timeout: float
) -> str:
    """Chama a API generate do Ollama."""
    url = endpoint.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }
    resp = httpx.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def _call_openai_compatible(
    endpoint: str,
    model: str,
    prompt: str,
    system: str,
    api_key: str,
    timeout: float,
) -> str:
    """Chama um endpoint compatível com a API chat/completions da OpenAI."""
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


class ModelRouter:
    """Lê MODEL_MODE do .env ('local' ou 'api').

    Se local: chama Ollama via endpoint configurado por componente.
    Se api: chama o provedor configurado (HuggingFace ou outro compatível
    com OpenAI).

    O método generate(prompt, model, endpoint) funciona identicamente em
    ambos os modos — o resto do sistema nunca sabe qual modo está activo.
    """

    def __init__(self, db=None):
        self.mode = os.getenv("MODEL_MODE", "local")
        self.api_key = os.getenv("HUGGINGFACE_API_KEY", "")
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
    ) -> str:
        """Gera texto a partir do modelo, com retry+backoff transparente."""
        return call_model_with_retry(
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            system=system,
            mode=self.mode,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
            db=self.db,
            component=component,
        )

    async def agenerate(
        self,
        prompt: str,
        model: str,
        endpoint: str,
        system: str = "",
        component: str = "unknown",
        timeout: float = 120.0,
    ) -> str:
        """Versão assíncrona — corre a chamada síncrona numa thread.

        Permite usar asyncio.gather() sobre múltiplos NEURONS sem bloquear
        o event loop. O retry+backoff continua a aplicar-se por chamada.
        """
        import asyncio

        return await asyncio.to_thread(
            self.generate, prompt, model, endpoint, system, component, timeout
        )


def component_config(component: str) -> tuple[str, str, bool]:
    """Lê endpoint, modelo e flag de existência de um componente do .env.

    `component` é 'cortex', 'cerebellum' ou 'neuron_N'. Devolve
    (endpoint, model, enabled). Para NEURONS, `enabled` reflecte
    ENABLE_NEURON_N (existência no sistema — não activação por ronda).
    Para CORTEX/CEREBELLUM `enabled` é sempre True.
    """
    if component == "cortex":
        return (
            os.getenv("CORTEX_ENDPOINT", "http://localhost:11434"),
            os.getenv("CORTEX_MODEL", ""),
            True,
        )
    if component == "cerebellum":
        return (
            os.getenv("CEREBELLUM_ENDPOINT", "http://localhost:11434"),
            os.getenv("CEREBELLUM_MODEL", ""),
            True,
        )
    # neuron_N
    n = component.split("_")[-1]
    endpoint = os.getenv(f"NEURON_{n}_ENDPOINT", "http://localhost:11434")
    model = os.getenv(f"NEURON_{n}_MODEL", "")
    enabled = os.getenv(f"ENABLE_NEURON_{n}", "true").lower() == "true"
    return endpoint, model, enabled
