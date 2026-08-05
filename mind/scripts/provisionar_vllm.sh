#!/usr/bin/env bash
# ==========================================================================
# Aprovisionamento de um servidor vLLM numa GPU alugada (vast.ai e afins)
# ==========================================================================
# Corre ESTE script NA INSTÂNCIA alugada, não na tua máquina.
#
# Depois de correr, o MIND fala com ele pondo no .env:
#     MODEL_MODE=openai_compat
#     MODEL_ENDPOINT=http://127.0.0.1:8000      (através do túnel)
#     MODEL_AUTH_TOKEN=<o token que este script imprime>
#     CORTEX_MODEL=<o id do modelo>
#
# SEGURANÇA — ler antes de usar:
#   O servidor escuta em 127.0.0.1 de propósito. NÃO o exponhas à internet.
#   Da tua máquina, abre um túnel SSH:
#       ssh -N -L 8000:127.0.0.1:8000 -p <porta> root@<host-da-instancia>
#   Assim o tráfego vai cifrado e o servidor não fica acessível a quem
#   descobrir o IP. Um servidor de inferência aberto é uma GPU que estás a
#   pagar para servir outra pessoa — e os teus prompts nas mãos dela.
#
# Uso:
#   ./provisionar_vllm.sh <modelo-do-hub> [porta]
#   ./provisionar_vllm.sh Qwen/Qwen2.5-Coder-7B-Instruct
#
# Variáveis opcionais:
#   HUGGINGFACE_API_KEY   para modelos privados (checkpoints afinados)
#   MODEL_AUTH_TOKEN      token a exigir; se vazio, é gerado um
#   LORA_MODULES          adaptadores LoRA: "nome=/caminho nome2=/caminho2"
#   MAX_MODEL_LEN         contexto máximo (default 8192)
# ==========================================================================

set -euo pipefail

MODELO="${1:-}"
PORTA="${2:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

if [[ -z "$MODELO" ]]; then
    echo "Uso: $0 <modelo-do-hub> [porta]" >&2
    echo "Ex:  $0 Qwen/Qwen2.5-Coder-7B-Instruct" >&2
    exit 1
fi

# --- Token de autenticação -------------------------------------------------
if [[ -z "${MODEL_AUTH_TOKEN:-}" ]]; then
    MODEL_AUTH_TOKEN="$(head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 40)"
    echo ">> Token gerado (guarda-o, vai para o .env do MIND):"
    echo "   MODEL_AUTH_TOKEN=${MODEL_AUTH_TOKEN}"
fi

# --- Verificação de GPU ----------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "AVISO: nvidia-smi não encontrado. Sem GPU, o vLLM não arranca." >&2
else
    echo ">> GPU disponível:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# --- Instalação ------------------------------------------------------------
if ! python3 -c "import vllm" >/dev/null 2>&1; then
    echo ">> A instalar vLLM (demora alguns minutos)..."
    pip install --quiet --upgrade pip
    pip install --quiet vllm
else
    echo ">> vLLM já instalado."
fi

# --- Adaptadores LoRA ------------------------------------------------------
# Vários componentes do MIND podem partilhar um modelo base e diferir apenas
# no adaptador — é o que torna 8 componentes viáveis numa só GPU.
ARGS_LORA=()
if [[ -n "${LORA_MODULES:-}" ]]; then
    echo ">> Adaptadores LoRA: ${LORA_MODULES}"
    ARGS_LORA+=(--enable-lora)
    for modulo in ${LORA_MODULES}; do
        ARGS_LORA+=(--lora-modules "${modulo}")
    done
fi

# --- Arranque --------------------------------------------------------------
echo ">> A arrancar o servidor em 127.0.0.1:${PORTA} com ${MODELO}"
echo ">> (escuta só em localhost — usa um túnel SSH a partir da tua máquina)"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODELO}" \
    --host 127.0.0.1 \
    --port "${PORTA}" \
    --api-key "${MODEL_AUTH_TOKEN}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    "${ARGS_LORA[@]}"
