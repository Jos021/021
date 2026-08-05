#!/usr/bin/env python3
"""Servidor de ensaio — endpoint falso compatível com OpenAI.

Serve para ENSAIAR o runbook do piloto sem GPU e sem custo: responde como
um servidor vLLM responderia, com respostas plausíveis para cada momento do
ciclo do MIND.

NÃO substitui o piloto. Os modelos são falsos e as respostas são fixas —
o que isto valida é a SEQUÊNCIA OPERACIONAL: que os comandos existem, que
encadeiam, que o CSV sai, que nada rebenta a meio. Uma discrepância
encontrada aqui é uma que não custa tempo de GPU depois.

Uso:
    python scripts/servidor_de_ensaio.py [porta]         # arranca
    MODEL_MODE=openai_compat MODEL_ENDPOINT=http://127.0.0.1:8999 \\
        CORTEX_MODEL=ensaio ... python main.py verificar

Opções por variável de ambiente:
    ENSAIO_FALHA_JSON=true    responde fora do esquema, para exercitar o
                              registo de desvios de formato
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


class Manipulador(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        tamanho = int(self.headers.get("Content-Length", 0))
        try:
            pedido = json.loads(self.rfile.read(tamanho) or b"{}")
        except ValueError:
            pedido = {}

        mensagens = pedido.get("messages") or []
        system = next((m.get("content", "") for m in mensagens
                       if m.get("role") == "system"), "")
        prompt = next((m.get("content", "") for m in reversed(mensagens)
                       if m.get("role") == "user"), "")

        atraso = float(os.getenv("ENSAIO_LATENCIA", "0"))
        if atraso:
            time.sleep(atraso)

        conteudo = responder(system, prompt)
        corpo = json.dumps({
            "id": "ensaio", "object": "chat.completion",
            "model": pedido.get("model", "ensaio"),
            "choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": conteudo},
                         "finish_reason": "stop"}],
        }).encode("utf-8")

        self.send_response(200)
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
