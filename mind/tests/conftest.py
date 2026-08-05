"""Fixtures partilhadas da suite de testes do MIND.

Princípio: todos os testes são herméticos. Cada teste recebe uma SYNAPSE DB
própria e um workspace próprio em directório temporário — nada toca no
estado real do projecto, e a ordem de execução nunca importa.
"""

import json
import os
import sys

import pytest

# Permite importar `agent` e `main` a partir da raiz do projecto.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.database import SynapseDB  # noqa: E402


# Variáveis de ambiente que os testes controlam explicitamente. São limpas
# antes de cada teste para que a .env do utilizador nunca influencie o
# resultado.
_ENV_CONTROLADAS = [
    "MODEL_MODE", "MODEL_ENDPOINT",
    "CORTEX_MODEL", "CEREBELLUM_MODEL", "CORTEX_ENDPOINT",
    "CEREBELLUM_ENDPOINT", "APPROVAL_THRESHOLD", "DIVERGENCE_THRESHOLD",
    "MUNDJI_WORKSPACE", "MUNDJI_MAX_ITERATIONS", "MUNDJI_SANDBOX_TIMEOUT",
    "NEURON_TIMEOUT_SECONDS", "ENABLE_ROLLBACK", "GIT_PERMANENT_THRESHOLD",
    "OUTPUT_SANITIZER_ENABLED", "ML_ENABLED", "ML_MIN_TRAINING_SAMPLES",
    "ML_AUTO_REJECT_CONFIDENCE", "ML_RETRAIN_DEVIATION_THRESHOLD",
    "SANDBOX_TESTS_ENABLED", "SANDBOX_ACCUMULATE_LEVELS",
    "SANDBOX_MAX_ITER_PER_LEVEL", "SANDBOX_TESTS_PER_LEVEL",
    "SANDBOX_MIN_LIBRARY_SIMILARITY",
] + [f"NEURON_{n}_MODEL" for n in range(1, 7)] \
  + [f"ENABLE_NEURON_{n}" for n in range(1, 7)]


@pytest.fixture(autouse=True)
def ambiente_limpo(monkeypatch, tmp_path):
    """Isola cada teste do ambiente do utilizador.

    Aplicada automaticamente a todos os testes: limpa as variáveis que o
    MIND lê e aponta o workspace para um directório temporário.
    """
    for var in _ENV_CONTROLADAS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MUNDJI_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("OUTPUT_SANITIZER_ENABLED", "true")
    monkeypatch.setenv("APPROVAL_THRESHOLD", "98")
    monkeypatch.setenv("MUNDJI_SANDBOX_TIMEOUT", "15")
    monkeypatch.setenv("ML_ENABLED", "false")
    monkeypatch.setenv("SANDBOX_TESTS_ENABLED", "false")
    yield


@pytest.fixture
def db(tmp_path):
    """SYNAPSE DB temporária, já com o schema criado."""
    base = SynapseDB(str(tmp_path / "synapse.db"))
    yield base
    base.close()


@pytest.fixture
def cycle_id(db):
    """Um ciclo criado, para satisfazer as foreign keys."""
    return db.create_cycle("tarefa de teste")


class RouterFalso:
    """Router determinístico que substitui as chamadas a modelos.

    Devolve respostas coerentes por componente, o que permite exercitar o
    ciclo completo sem depender de endpoints externos. Regista as chamadas
    recebidas para que os testes possam afirmar quem foi (ou não) chamado.
    """

    def __init__(self, respostas=None, codigo_neuron=None):
        self.respostas = respostas or {}
        self.chamadas = []
        self.chamadas_async = []
        self.codigo_neuron = codigo_neuron or (
            "def somar(a, b):\n    return a + b\nprint(somar(2, 3))"
        )

    def generate(self, prompt, model, endpoint, system="", component="",
                 timeout=120, cycle_id=None, iteration=0):
        self.chamadas.append(component)
        if component in self.respostas:
            resposta = self.respostas[component]
            return resposta(prompt) if callable(resposta) else resposta
        if component == "cortex":
            if "Anota" in prompt:
                return f"# [NEURON_1:python]\n{self.codigo_neuron}"
            if "Aprimora" in prompt:
                return "# código base aprimorado"
            if "avaliação" in prompt.lower():
                return self.avaliacao(99)
            return "Lógica: somar.\n===CODIGO===\n# base\n"
        if component == "cerebellum":
            if "avaliação" in prompt.lower() or "reconcilia" in prompt.lower():
                return self.avaliacao(99)
            return "Avaliação técnica: ok."
        return ""

    @staticmethod
    def avaliacao(pct, failures=None, improvements=None, auto_reject=False):
        """Resposta no esquema JSON fixo que os prompts agora pedem."""
        return json.dumps({
            "functionality_pct": pct,
            "failures": failures or [],
            "improvements": improvements or {},
            "auto_reject": auto_reject,
        })

    async def agenerate(self, prompt, model, endpoint, system="",
                        component="", timeout=120, cycle_id=None, iteration=0):
        self.chamadas_async.append(component)
        n = component.split("_")[-1]
        return f"# [NEURON_{n}:python]\n{self.codigo_neuron}"


@pytest.fixture
def router():
    """Router falso com comportamento por omissão (ciclo que aprova)."""
    return RouterFalso()


@pytest.fixture
def com_modelos(monkeypatch):
    """Marca todos os componentes como tendo modelo configurado.

    Sem isto, os agentes degradam para o caminho 'sem modelo' e não chamam
    o router.
    """
    monkeypatch.setenv("CORTEX_MODEL", "modelo-de-teste")
    monkeypatch.setenv("CEREBELLUM_MODEL", "modelo-de-teste")
    for n in range(1, 7):
        monkeypatch.setenv(f"NEURON_{n}_MODEL", "modelo-de-teste")
