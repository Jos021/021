#!/usr/bin/env python3
"""Servidor de ensaio — endpoint falso, em dois formatos.

Serve para ENSAIAR o runbook do piloto sem GPU e sem custo: responde como
um servidor real responderia, com respostas plausíveis para cada momento do
ciclo do MIND. Fala dois protocolos, escolhidos pelo caminho do pedido:

    /v1/chat/completions   formato compatível com OpenAI (vLLM, TGI)
    /v1/messages           formato da API da Anthropic

O modo Anthropic é ESTRITO por desenho: rejeita com 400 qualquer pedido sem
`x-api-key`, sem `anthropic-version`, sem `max_tokens`, ou que meta o system
como mensagem de role "system" em vez de campo de topo. Um servidor
permissivo aceitaria pedidos mal formados e o ensaio não provaria nada. Sendo
estrito, um ciclo completo que corra até ao fim prova que TODOS os pontos de
chamada produzem pedidos válidos — não só o que o teste unitário cobre.

NÃO substitui o piloto. Os modelos são falsos e as respostas são fixas — o
que isto valida é a SEQUÊNCIA OPERACIONAL e a FORMA DOS PEDIDOS, nunca a
qualidade do que um modelo verdadeiro geraria.

Uso:
    python scripts/servidor_de_ensaio.py [porta]         # arranca
    MODEL_MODE=anthropic MODEL_ENDPOINT=http://127.0.0.1:8999 \\
        MODEL_AUTH_TOKEN=ensaio CORTEX_MODEL=ensaio ... python main.py verificar

Opções por variável de ambiente:
    ENSAIO_FALHA_JSON=true    responde fora do esquema, para exercitar o
                              registo de desvios de formato
    ENSAIO_TRUNCAR=true       devolve stop_reason=max_tokens, para exercitar
                              a guarda de truncagem do modo anthropic
    ENSAIO_LATENCIA=0.5       segundos de atraso por resposta
"""

import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

CODIGO_EXEMPLO = (
    "def somar(a, b):\n"
    "    \"\"\"Soma dois numeros.\"\"\"\n"
    "    return a + b\n"
    "\n"
    "print(somar(2, 3))"
)


def _avaliacao(pct=99):
    return json.dumps({
        "functionality_pct": pct,
        "failures": [],
        "improvements": {},
        "auto_reject": False,
    })


def _testes(nivel):
    categoria = {1: "basic", 2: "edge", 3: "error"}[nivel]
    return json.dumps([{
        "neuron_target": "neuron_1",
        "language": "python",
        "level": nivel,
        "category": categoria,
        "description": f"teste de ensaio, nivel {nivel}",
        "code": "assert 2 + 3 == 5",
        "expected_outcome": "pass",
    }])


def responder(system: str, prompt: str) -> str:
    """Escolhe a resposta pelo momento do ciclo que o prompt revela."""
    if os.getenv("ENSAIO_FALHA_JSON", "false").lower() == "true":
        return "A funcionalidade esta nos 99% mais coisa menos coisa."

    # Verificação de ligação.
    if prompt.strip().endswith("OK"):
        return "OK"

    # Geração de testes da sandbox evolutiva.
    m = re.search(r"N[ÍI]VEL (\d)", prompt)
    if m and "array JSON" in prompt:
        return _testes(int(m.group(1)))

    # NEURON: implementar a sua secção.
    m = re.search(r"\[NEURON_(\d)\]", prompt)
    if m and "Implementa APENAS" in prompt:
        return f"# [NEURON_{m.group(1)}:python]\n{CODIGO_EXEMPLO}"

    # CORTEX.
    if "Anota" in prompt:
        return f"# [NEURON_1:python]\n{CODIGO_EXEMPLO}"
    if "Aprimora" in prompt:
        return f"# [NEURON_1:python]\n{CODIGO_EXEMPLO}"
    if "avaliação" in prompt.lower() or "reconcilia" in prompt.lower():
        return _avaliacao()
    if "===CODIGO===" in prompt or "LÓGICA" in prompt.upper():
        return f"Logica: somar dois numeros.\n===CODIGO===\n{CODIGO_EXEMPLO}"
    return "Avaliacao tecnica: sem problemas de maior."


def _erro_anthropic(tipo: str, mensagem: str) -> bytes:
    """Erro no formato da API da Anthropic, para o router o saber ler."""
    return json.dumps({"type": "error",
                       "error": {"type": tipo, "message": mensagem}}).encode()


def validar_pedido_anthropic(cabecalhos, pedido) -> str:
    """Verifica que o pedido respeita o contrato da API da Anthropic.

    É aqui que está o valor deste modo. Um servidor permissivo aceitaria um
    pedido mal formado e o ensaio passaria à mesma, provando nada. Sendo
    estrito, um ciclo completo do MIND que corra até ao fim prova que TODOS
    os pontos de chamada — CORTEX, CEREBELLUM, os 6 NEURONS, o gerador de
    testes — produzem pedidos válidos. Devolve "" se estiver tudo bem.
    """
    if not cabecalhos.get("x-api-key"):
        return "x-api-key header is required"
    if cabecalhos.get("anthropic-version") != "2023-06-01":
        return ("anthropic-version header is required and must be "
                "2023-06-01")
    if "max_tokens" not in pedido:
        return "max_tokens: Field required"
    mensagens = pedido.get("messages") or []
    if not mensagens:
        return "messages: at least one message is required"
    for m in mensagens:
        if m.get("role") == "system":
            return ("Unexpected role 'system'. The Messages API accepts a "
                    "top-level 'system' parameter, not 'system' as an "
                    "input message role.")
    return ""


class Manipulador(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        tamanho = int(self.headers.get("Content-Length", 0))
        try:
            pedido = json.loads(self.rfile.read(tamanho) or b"{}")
        except ValueError:
            pedido = {}

        atraso = float(os.getenv("ENSAIO_LATENCIA", "0"))
        if atraso:
            time.sleep(atraso)

        if self.path.rstrip("/").endswith("/v1/messages"):
            self._responder_anthropic(pedido)
        else:
            self._responder_openai(pedido)

    # -- formato compatível com OpenAI (vLLM, TGI, llama.cpp) -------------
    def _responder_openai(self, pedido):
        mensagens = pedido.get("messages") or []
        system = next((m.get("content", "") for m in mensagens
                       if m.get("role") == "system"), "")
        prompt = next((m.get("content", "") for m in reversed(mensagens)
                       if m.get("role") == "user"), "")
        conteudo = responder(system, prompt)
        self._enviar(200, json.dumps({
            "id": "ensaio", "object": "chat.completion",
            "model": pedido.get("model", "ensaio"),
            "choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": conteudo},
                         "finish_reason": "stop"}],
        }).encode("utf-8"))

    # -- formato da API da Anthropic --------------------------------------
    def _responder_anthropic(self, pedido):
        problema = validar_pedido_anthropic(self.headers, pedido)
        if problema:
            self._enviar(400, _erro_anthropic("invalid_request_error",
                                              problema))
            return

        # O system vem no campo de topo, que é precisamente o ponto a provar.
        system = pedido.get("system", "") or ""
        prompt = next((m.get("content", "") for m in reversed(
            pedido.get("messages") or []) if m.get("role") == "user"), "")
        conteudo = responder(system, prompt)

        # Exercita a guarda de truncagem de ponta a ponta.
        if os.getenv("ENSAIO_TRUNCAR", "false").lower() == "true":
            self._enviar(200, json.dumps({
                "id": "msg_ensaio", "type": "message", "role": "assistant",
                "model": pedido.get("model", "ensaio"),
                "content": [{"type": "text", "text": conteudo[:20]}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 100, "output_tokens": 4096},
            }).encode("utf-8"))
            return

        self._enviar(200, json.dumps({
            "id": "msg_ensaio", "type": "message", "role": "assistant",
            "model": pedido.get("model", "ensaio"),
            "content": [{"type": "text", "text": conteudo}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 200},
        }).encode("utf-8"))

    def _enviar(self, estado: int, corpo: bytes):
        self.send_response(estado)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def log_message(self, *args):
        """Silencioso: o output do MIND é que interessa ver."""


def main():
    porta = int(sys.argv[1]) if len(sys.argv) > 1 else 8999
    servidor = HTTPServer(("127.0.0.1", porta), Manipulador)
    print(f"Servidor de ensaio em http://127.0.0.1:{porta} "
          "(Ctrl-C para parar)")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
