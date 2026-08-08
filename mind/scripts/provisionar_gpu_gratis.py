#!/usr/bin/env python3
"""Aprovisiona um servidor vLLM numa GPU GRÁTIS (Kaggle, Lightning, Colab).

Corre ISTO na plataforma da GPU — cola como célula de notebook (Kaggle,
Colab) ou executa como script (Lightning AI Studio). No fim imprime as
linhas exactas a pôr no .env do MIND.

--------------------------------------------------------------------------
DIFERENÇA FACE AO provisionar_vllm.sh
--------------------------------------------------------------------------
Aquele é para GPU alugada com acesso SSH, e o túnel é SSH. Estas
plataformas não dão SSH, por isso o túnel é cloudflared, que produz um URL
PÚBLICO. É uma diferença de segurança, não de conveniência — ver a secção
de segurança em baixo.

--------------------------------------------------------------------------
ARMADILHAS QUE ESTE SCRIPT JÁ EVITA
--------------------------------------------------------------------------
1. O vLLM exige capacidade de computação >= 7.0. A P100 do Kaggle é 6.0:
   NÃO SERVE. Tem de se escolher o acelerador "T4 x2" e não "P100" — o
   script verifica e diz-o em vez de rebentar a meio da instalação.
2. A T4 (7.5) não suporta bfloat16. Sem --dtype half o vLLM aborta ou cai
   para float32 e não cabe na memória.
3. 16 GB de VRAM não chegam para um modelo de 7B em fp16 com contexto
   grande. O script dimensiona --max-model-len ao que sobra.
4. Sessões terminam (Kaggle 9h, Colab menos). Um piloto tem de caber, e o
   URL do túnel muda a cada arranque.

--------------------------------------------------------------------------
SEGURANÇA — ler antes de usar
--------------------------------------------------------------------------
O túnel cloudflared expõe o servidor à internet inteira. Sem token, quem
descobrir o URL usa a GPU e vê os prompts. Por isso o servidor arranca
SEMPRE com --api-key, e o script gera um se não for dado.

O URL é efémero e não indexado, mas isso é obscuridade, não segurança.
Fecha a sessão quando acabares.
"""

import os
import secrets
import shutil
import subprocess
import sys
import time

PORTA = int(os.environ.get("PORTA", "8000"))
MODELO = os.environ.get(
    "MODELO", "Qwen/Qwen2.5-Coder-7B-Instruct"
)


def executar(cmd, **kw):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=False, **kw)


def verificar_gpu() -> tuple[str, float, int]:
    """Devolve (nome, memoria_gb, capacidade_major). Aborta se não servir."""
    try:
        import torch
    except ImportError:
        print("torch não está instalado — esta plataforma tem GPU?")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERRO: não há GPU visível. No Kaggle: Settings -> Accelerator "
              "-> GPU T4 x2. No Colab: Runtime -> Change runtime type.")
        sys.exit(1)

    nome = torch.cuda.get_device_name(0)
    maior, menor = torch.cuda.get_device_capability(0)
    memoria = torch.cuda.get_device_properties(0).total_memory / 1e9
    n = torch.cuda.device_count()
    print(f"GPU: {nome} x{n} | {memoria:.1f} GB | capacidade {maior}.{menor}")

    if maior < 7:
        print(f"\nERRO: o vLLM exige capacidade >= 7.0 e esta é {maior}.{menor}.")
        if "P100" in nome:
            print("É a P100 do Kaggle. Muda o acelerador para 'GPU T4 x2' "
                  "nas Settings do notebook e volta a correr.")
        sys.exit(1)
    return nome, memoria, maior


def main():
    nome, memoria, capacidade = verificar_gpu()

    # A T4 (7.5) não tem bfloat16. A A100/H100 (8.0+) têm.
    dtype = "half" if capacidade < 8 else "bfloat16"

    # Deixar margem para o cache KV: quanto menor a VRAM, menor o contexto.
    if memoria < 17:
        max_len = 8192
    elif memoria < 25:
        max_len = 16384
    else:
        max_len = 32768

    token = os.environ.get("MODEL_AUTH_TOKEN") or secrets.token_urlsafe(24)

    print("\n=== 1. instalar vLLM (demora vários minutos) ===")
    executar(f"{sys.executable} -m pip install -q -U vllm")

    print("\n=== 2. obter o cloudflared ===")
    if not shutil.which("cloudflared"):
        executar("curl -sSL -o /tmp/cloudflared "
                 "https://github.com/cloudflare/cloudflared/releases/latest/"
                 "download/cloudflared-linux-amd64 && chmod +x /tmp/cloudflared")
        cloudflared = "/tmp/cloudflared"
    else:
        cloudflared = "cloudflared"

    print(f"\n=== 3. arrancar o vLLM ({MODELO}) ===")
    print(f"    dtype={dtype}  max_model_len={max_len}")
    servidor = subprocess.Popen(
        f"{sys.executable} -m vllm.entrypoints.openai.api_server "
        f"--model {MODELO} --port {PORTA} --host 127.0.0.1 "
        f"--dtype {dtype} --max-model-len {max_len} "
        f"--gpu-memory-utilization 0.90 --api-key {token} "
        f"> /tmp/vllm.log 2>&1",
        shell=True,
    )

    print("    a esperar que o modelo carregue (pode levar 5-15 min)...")
    pronto = False
    for _ in range(180):
        time.sleep(10)
        r = subprocess.run(
            f"curl -sf -o /dev/null -H 'Authorization: Bearer {token}' "
            f"http://127.0.0.1:{PORTA}/v1/models",
            shell=True,
        )
        if r.returncode == 0:
            pronto = True
            break
        if servidor.poll() is not None:
            print("\nERRO: o vLLM terminou. Últimas linhas do log:")
            executar("tail -30 /tmp/vllm.log")
            sys.exit(1)
    if not pronto:
        print("\nERRO: o servidor não respondeu a tempo. Log:")
        executar("tail -30 /tmp/vllm.log")
        sys.exit(1)
    print("    servidor a responder.")

    print("\n=== 4. abrir o túnel ===")
    subprocess.Popen(
        f"{cloudflared} tunnel --url http://127.0.0.1:{PORTA} "
        f"--no-autoupdate > /tmp/tunel.log 2>&1", shell=True,
    )
    url = ""
    for _ in range(30):
        time.sleep(3)
        try:
            with open("/tmp/tunel.log", encoding="utf-8") as fh:
                texto = fh.read()
        except FileNotFoundError:
            continue
        for parte in texto.split():
            if parte.startswith("https://") and "trycloudflare.com" in parte:
                url = parte.strip()
                break
        if url:
            break
    if not url:
        print("ERRO: o túnel não deu URL. Log:")
        executar("tail -20 /tmp/tunel.log")
        sys.exit(1)

    print("\n" + "=" * 74)
    print("PRONTO — põe isto no .env do MIND (na tua máquina):")
    print("=" * 74)
    print(f"MODEL_MODE=openai_compat")
    print(f"MODEL_ENDPOINT={url}")
    print(f"MODEL_AUTH_TOKEN={token}")
    print(f"CORTEX_MODEL={MODELO}")
    print(f"CEREBELLUM_MODEL={MODELO}")
    for n in range(1, 7):
        print(f"NEURON_{n}_MODEL={MODELO}")
    print("=" * 74)
    print("Verifica com:  python main.py verificar")
    print("\nDEIXA ESTA CÉLULA A CORRER. Se parar, o túnel fecha e o URL "
          "muda no arranque seguinte.")
    try:
        servidor.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
