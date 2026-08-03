"""ModelRouter — modos de serviço, autenticação e retry.

Os modelos são todos do HuggingFace e correm em GPU alugada. O router tem
de abstrair as formas de os servir sem que o resto do sistema saiba qual
está activa, e tem de aguentar as falhas transitórias que uma instância
alugada produz.
"""

import httpx
import pytest

from agent.model_router import (
    MODOS,
    ModelError,
    ModelRouter,
    call_model_with_retry,
    component_config,
    token_do_componente,
)


# --- Selecção de modo -----------------------------------------------------
def test_modo_por_omissao_e_openai_compat():
    """O recomendado para GPU alugada é o que fica activo sem configurar."""
    assert ModelRouter().mode == "openai_compat"


@pytest.mark.parametrize("modo", MODOS)
def test_modos_suportados_sao_aceites(modo, monkeypatch):
    monkeypatch.setenv("MODEL_MODE", modo)
    assert ModelRouter().mode == modo


def test_modo_desconhecido_cai_no_por_omissao(monkeypatch):
    """Um MODEL_MODE errado não pode rebentar o ciclo nem falhar em silêncio."""
    monkeypatch.setenv("MODEL_MODE", "inventado")
    assert ModelRouter().mode == "openai_compat"


def test_modo_e_normalizado(monkeypatch):
    monkeypatch.setenv("MODEL_MODE", "  HF_Local  ")
    assert ModelRouter().mode == "hf_local"


# --- Configuração por componente -----------------------------------------
def test_endpoint_por_omissao_e_herdado(monkeypatch):
    """Um componente sem endpoint próprio usa MODEL_ENDPOINT."""
    monkeypatch.setenv("MODEL_ENDPOINT", "https://gpu-alugada:8000")
    endpoint, _, _ = component_config("cortex")
    assert endpoint == "https://gpu-alugada:8000"


def test_endpoint_do_componente_tem_precedencia(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://barata:8000")
    monkeypatch.setenv("CORTEX_ENDPOINT", "https://grande:8000")
    endpoint, _, _ = component_config("cortex")
    assert endpoint == "https://grande:8000"


@pytest.mark.parametrize("valor", [
    "Qwen/Qwen2.5-Coder-32B-Instruct",   # id do Hub
    "cortex-lora-v3",                     # adaptador LoRA servido
    "/models/finetuned/cortex",           # checkpoint afinado em disco
])
def test_modelo_aceita_hub_lora_e_caminho(valor, monkeypatch):
    """As três formas são tratadas igualmente — é só uma string."""
    monkeypatch.setenv("CORTEX_MODEL", valor)
    _, modelo, _ = component_config("cortex")
    assert modelo == valor


def test_neuron_desactivado(monkeypatch):
    monkeypatch.setenv("ENABLE_NEURON_3", "false")
    _, _, activo = component_config("neuron_3")
    assert activo is False


# --- Autenticação ---------------------------------------------------------
def test_token_do_componente_tem_precedencia(monkeypatch):
    monkeypatch.setenv("MODEL_AUTH_TOKEN", "global")
    monkeypatch.setenv("CORTEX_TOKEN", "so-do-cortex")
    assert token_do_componente("cortex") == "so-do-cortex"


def test_token_global_e_usado_quando_nao_ha_especifico(monkeypatch):
    monkeypatch.setenv("MODEL_AUTH_TOKEN", "global")
    assert token_do_componente("neuron_4") == "global"


def test_token_de_neuron_especifico(monkeypatch):
    """Componentes podem apontar para servidores diferentes."""
    monkeypatch.setenv("MODEL_AUTH_TOKEN", "global")
    monkeypatch.setenv("NEURON_2_TOKEN", "outra-gpu")
    assert token_do_componente("neuron_2") == "outra-gpu"


def test_chave_huggingface_serve_de_recurso(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_xxx")
    assert token_do_componente("cerebellum") == "hf_xxx"


def test_sem_token_nenhum(monkeypatch):
    for var in ("MODEL_AUTH_TOKEN", "HUGGINGFACE_API_KEY", "CORTEX_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert token_do_componente("cortex") == ""


def test_token_e_enviado_no_cabecalho(monkeypatch):
    enviados = {}

    def falso_post(url, json=None, headers=None, timeout=None):
        enviados["headers"] = headers or {}
        return _resposta_openai("resultado")

    monkeypatch.setattr(httpx, "post", falso_post)
    call_model_with_retry("https://gpu:8000", "m", "p", api_key="segredo",
                          mode="openai_compat")
    assert enviados["headers"]["Authorization"] == "Bearer segredo"


# --- Chamadas por modo ----------------------------------------------------
class _Resposta:
    def __init__(self, dados, status=200):
        self._dados = dados
        self.status_code = status

    def json(self):
        return self._dados

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("erro", request=None, response=self)


def _resposta_openai(texto):
    return _Resposta({"choices": [{"message": {"content": texto}}]})


def test_modo_openai_compat_le_a_resposta(monkeypatch):
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _resposta_openai("gerado por vLLM"))
    assert call_model_with_retry("https://gpu:8000", "m", "p",
                                 mode="openai_compat") == "gerado por vLLM"


def test_modo_openai_compat_monta_o_url(monkeypatch):
    visto = {}

    def falso_post(url, **k):
        visto["url"] = url
        return _resposta_openai("x")

    monkeypatch.setattr(httpx, "post", falso_post)
    call_model_with_retry("https://gpu:8000/", "m", "p", mode="openai_compat")
    assert visto["url"] == "https://gpu:8000/v1/chat/completions"


def test_modo_hf_api_monta_o_url_do_hub(monkeypatch):
    visto = {}

    def falso_post(url, **k):
        visto["url"] = url
        return _Resposta([{"generated_text": "gerado"}])

    monkeypatch.setattr(httpx, "post", falso_post)
    saida = call_model_with_retry("", "Qwen/Qwen2.5-Coder-7B-Instruct", "p",
                                  mode="hf_api")
    assert saida == "gerado"
    assert visto["url"].endswith("/models/Qwen/Qwen2.5-Coder-7B-Instruct")


def test_modo_ollama_continua_a_funcionar(monkeypatch):
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: _Resposta({"response": "do ollama"}))
    assert call_model_with_retry("http://localhost:11434", "m", "p",
                                 mode="ollama") == "do ollama"


def test_modo_hf_local_sem_transformers_da_erro_claro(monkeypatch):
    """Sem as dependências, a mensagem tem de dizer o que instalar."""
    import builtins

    real = builtins.__import__

    def bloquear(nome, *a, **k):
        if nome.startswith("transformers"):
            raise ImportError("sem transformers")
        return real(nome, *a, **k)

    monkeypatch.setattr(builtins, "__import__", bloquear)
    with pytest.raises(ModelError) as exc:
        call_model_with_retry("", "algum/modelo", "p", mode="hf_local",
                              max_retries=1)
    assert "requirements-hf.txt" in str(exc.value)


# --- Retry e backoff ------------------------------------------------------
def test_retry_em_erro_transitorio(monkeypatch):
    """Uma instância alugada que reinicia não pode reprovar um ciclo."""
    tentativas = []

    def falso_post(*a, **k):
        tentativas.append(1)
        if len(tentativas) < 3:
            raise httpx.ConnectError("instância indisponível")
        return _resposta_openai("recuperou")

    monkeypatch.setattr(httpx, "post", falso_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    saida = call_model_with_retry("https://gpu:8000", "m", "p",
                                  mode="openai_compat", max_retries=3)
    assert saida == "recuperou"
    assert len(tentativas) == 3


def test_erro_nao_recuperavel_nao_repete(monkeypatch):
    """Um 401 não melhora com insistência."""
    tentativas = []

    def falso_post(*a, **k):
        tentativas.append(1)
        return _Resposta({}, status=401)

    monkeypatch.setattr(httpx, "post", falso_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ModelError):
        call_model_with_retry("https://gpu:8000", "m", "p",
                              mode="openai_compat", max_retries=3)
    assert len(tentativas) == 1


def test_erro_429_repete(monkeypatch):
    """Servidor saturado é transitório."""
    tentativas = []

    def falso_post(*a, **k):
        tentativas.append(1)
        if len(tentativas) < 2:
            return _Resposta({}, status=429)
        return _resposta_openai("ok")

    monkeypatch.setattr(httpx, "post", falso_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert call_model_with_retry("https://gpu:8000", "m", "p",
                                 mode="openai_compat", max_retries=3) == "ok"


def test_backoff_duplica(monkeypatch):
    esperas = []
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: (_ for _ in ()).throw(
                            httpx.ConnectError("falhou")))
    monkeypatch.setattr("time.sleep", esperas.append)
    with pytest.raises(ModelError):
        call_model_with_retry("https://gpu:8000", "m", "p",
                              mode="openai_compat", max_retries=3, backoff=2)
    assert esperas == [2, 4]


def test_falha_e_registada_na_synapse_db(db, cycle_id, monkeypatch):
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: (_ for _ in ()).throw(
                            httpx.ConnectError("sem rede")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ModelError):
        call_model_with_retry("https://gpu:8000", "m", "p",
                              mode="openai_compat", max_retries=1, db=db,
                              component="neuron_1", cycle_id=cycle_id)
    linhas = db._conn.execute(
        "SELECT * FROM iterations WHERE component='neuron_1' "
        "AND phase='model_call'").fetchall()
    assert linhas and "falhou" in linhas[0]["full_output"]


def test_sem_cycle_id_nao_tenta_registar(db, monkeypatch):
    """Sem ciclo real não há registo — mas também não há excepção."""
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: (_ for _ in ()).throw(
                            httpx.ConnectError("x")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ModelError):
        call_model_with_retry("https://gpu:8000", "m", "p",
                              mode="openai_compat", max_retries=1, db=db,
                              component="cortex")
    assert db._conn.execute(
        "SELECT COUNT(*) AS n FROM iterations").fetchone()["n"] == 0


def test_erro_estruturado_nomeia_o_componente(monkeypatch):
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: (_ for _ in ()).throw(
                            httpx.ConnectError("x")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ModelError) as exc:
        call_model_with_retry("https://gpu:8000", "m", "p",
                              mode="openai_compat", max_retries=1,
                              component="cerebellum")
    assert exc.value.component == "cerebellum"
