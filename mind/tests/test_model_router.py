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


def test_endpoint_do_componente_vazio_cai_no_global(monkeypatch):
    """O .env.example traz os endpoints por componente vazios de propósito.

    Sem esta regra, um CORTEX_ENDPOINT= vazio devolvia "" (a variável está
    definida, logo o default do getenv não entra) e os 8 componentes
    reprovavam à primeira chamada — foi o que o ensaio local apanhou.
    """
    monkeypatch.setenv("MODEL_ENDPOINT", "https://global:8000")
    monkeypatch.setenv("CORTEX_ENDPOINT", "")        # vazio, como no template
    monkeypatch.setenv("NEURON_2_ENDPOINT", "   ")   # só espaços
    assert component_config("cortex")[0] == "https://global:8000"
    assert component_config("neuron_2")[0] == "https://global:8000"


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


# --- Modo anthropic -------------------------------------------------------
# O formato da Anthropic difere do openai_compat em quatro pontos, e nenhum
# deles falha de forma barulhenta se estiver errado: ou é um 400, ou lê-se
# uma resposta vazia. Cada um tem o seu teste.
def _resposta_anthropic(texto):
    return _Resposta({"content": [{"type": "text", "text": texto}]})


def _capturar_pedido_anthropic(monkeypatch, texto="resposta da Anthropic"):
    """Intercepta o POST e devolve o dicionário com url/headers/json."""
    visto = {}

    def falso_post(url, json=None, headers=None, timeout=None):
        visto.update(url=url, json=json or {}, headers=headers or {})
        return _resposta_anthropic(texto)

    monkeypatch.setattr(httpx, "post", falso_post)
    return visto


def test_modo_anthropic_autentica_com_x_api_key(monkeypatch):
    """A chave vai em x-api-key, não em Authorization: Bearer."""
    visto = _capturar_pedido_anthropic(monkeypatch)
    call_model_with_retry("https://api.anthropic.com", "claude-sonnet-4-6",
                          "p", api_key="sk-ant-xxx", mode="anthropic")
    assert visto["headers"]["x-api-key"] == "sk-ant-xxx"
    assert "Authorization" not in visto["headers"]


def test_modo_anthropic_envia_o_header_de_versao(monkeypatch):
    """anthropic-version é obrigatório: sem ele a API recusa o pedido."""
    visto = _capturar_pedido_anthropic(monkeypatch)
    call_model_with_retry("https://api.anthropic.com", "claude-sonnet-4-6",
                          "p", api_key="sk-ant-xxx", mode="anthropic")
    assert visto["headers"]["anthropic-version"] == "2023-06-01"


def test_modo_anthropic_poe_o_system_no_campo_proprio(monkeypatch):
    """O system prompt é um campo de topo, não uma mensagem de role system."""
    visto = _capturar_pedido_anthropic(monkeypatch)
    call_model_with_retry("https://api.anthropic.com", "claude-sonnet-4-6",
                          "p", system="és o CORTEX", mode="anthropic")
    corpo = visto["json"]
    assert corpo["system"] == "és o CORTEX"
    assert [m["role"] for m in corpo["messages"]] == ["user"]
    assert all(m["role"] != "system" for m in corpo["messages"])


def test_modo_anthropic_le_a_resposta_de_content_text(monkeypatch):
    """O texto está em content[0].text, não em choices[0].message.content."""
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _resposta_anthropic("gerado pelo Claude"))
    assert call_model_with_retry("https://api.anthropic.com", "m", "p",
                                 mode="anthropic") == "gerado pelo Claude"


def test_modo_anthropic_monta_o_url_das_mensagens(monkeypatch):
    visto = _capturar_pedido_anthropic(monkeypatch)
    call_model_with_retry("https://api.anthropic.com/", "m", "p",
                          mode="anthropic")
    assert visto["url"] == "https://api.anthropic.com/v1/messages"


def test_modo_anthropic_envia_max_tokens(monkeypatch):
    """max_tokens é obrigatório aqui, ao contrário do openai_compat."""
    visto = _capturar_pedido_anthropic(monkeypatch)
    call_model_with_retry("https://api.anthropic.com", "m", "p",
                          mode="anthropic")
    assert visto["json"]["max_tokens"] == 4096


def test_modo_anthropic_ignora_blocos_que_nao_sao_texto(monkeypatch):
    """content é uma lista heterogénea: assumir o índice 0 dá KeyError."""
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resposta({
        "content": [{"type": "thinking", "thinking": "..."},
                    {"type": "text", "text": "o que interessa"}]}))
    assert call_model_with_retry("https://api.anthropic.com", "m", "p",
                                 mode="anthropic") == "o que interessa"


def test_modo_anthropic_recusa_resposta_truncada(monkeypatch):
    """stop_reason=max_tokens vem com 200 e texto cortado — não é sucesso.

    Se passasse, o código incompleto ia à sandbox e o CEREBELLUM reprovaria
    a iteração pela razão errada.
    """
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resposta({
        "stop_reason": "max_tokens",
        "content": [{"type": "text", "text": "def f(x):\n    retur"}]}))
    with pytest.raises(ModelError) as exc:
        call_model_with_retry("https://api.anthropic.com", "m", "p",
                              mode="anthropic", max_retries=3)
    assert "truncada" in str(exc.value)
    assert "4096" in str(exc.value)


def test_truncagem_nao_e_repetida(monkeypatch):
    """Repetir uma truncagem dá exactamente a mesma truncagem."""
    tentativas = []

    def falso_post(*a, **k):
        tentativas.append(1)
        return _Resposta({"stop_reason": "max_tokens",
                          "content": [{"type": "text", "text": "cortado"}]})

    monkeypatch.setattr(httpx, "post", falso_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ModelError):
        call_model_with_retry("https://api.anthropic.com", "m", "p",
                              mode="anthropic", max_retries=3)
    assert len(tentativas) == 1


def test_stop_reason_normal_passa(monkeypatch):
    """end_turn é o caso bom e não pode ser afectado."""
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resposta({
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "completo"}]}))
    assert call_model_with_retry("https://api.anthropic.com", "m", "p",
                                 mode="anthropic") == "completo"


def test_endpoint_por_omissao_do_modo_anthropic(monkeypatch):
    """Com MODEL_MODE=anthropic e MODEL_ENDPOINT ausente, usa a API pública."""
    monkeypatch.setenv("MODEL_MODE", "anthropic")
    monkeypatch.delenv("MODEL_ENDPOINT", raising=False)
    endpoint, _, _ = component_config("cortex")
    assert endpoint == "https://api.anthropic.com"


def test_endpoint_explicito_manda_mesmo_no_modo_anthropic(monkeypatch):
    """Permite apontar a um proxy compatível sem mudar de modo."""
    monkeypatch.setenv("MODEL_MODE", "anthropic")
    monkeypatch.setenv("MODEL_ENDPOINT", "https://proxy-interno:8443")
    endpoint, _, _ = component_config("neuron_1")
    assert endpoint == "https://proxy-interno:8443"


def test_omissao_dos_outros_modos_nao_muda(monkeypatch):
    """Regressão: a omissão só passa a ser a Anthropic no modo anthropic."""
    monkeypatch.delenv("MODEL_ENDPOINT", raising=False)
    for modo in ("openai_compat", "hf_api", "ollama"):
        monkeypatch.setenv("MODEL_MODE", modo)
        endpoint, _, _ = component_config("cortex")
        assert endpoint == "http://localhost:8000"


def test_modo_anthropic_repete_em_529(monkeypatch):
    """529 (overloaded) é transitório e tem de entrar no retry."""
    tentativas = []

    def falso_post(*a, **k):
        tentativas.append(1)
        if len(tentativas) < 2:
            return _Resposta({}, status=529)
        return _resposta_anthropic("recuperou")

    monkeypatch.setattr(httpx, "post", falso_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert call_model_with_retry("https://api.anthropic.com", "m", "p",
                                 mode="anthropic", max_retries=3) == "recuperou"


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


# --- Diagnóstico: o que a mensagem de erro tem de dizer --------------------
# Estes dois testes nasceram de uma falha real. Um 400 da Anthropic por falta
# de saldo apareceu ao operador como "Client error '400 Bad Request'" e um
# link para a documentação da Mozilla — a razão verdadeira estava no corpo da
# resposta, que era deitado fora. O `verificar` existe para dizer o que está
# errado antes de se gastar dinheiro.
class _RespostaComCorpo(_Resposta):
    def __init__(self, dados, status=400, texto=""):
        super().__init__(dados, status)
        self.text = texto


def test_erro_reporta_a_mensagem_da_api(monkeypatch):
    """O corpo do erro é a única informação que interessa — não se descarta."""
    corpo = {"type": "error", "error": {
        "type": "invalid_request_error",
        "message": "Your credit balance is too low to access the "
                   "Anthropic API."}}
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: _RespostaComCorpo(corpo))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ModelError) as exc:
        call_model_with_retry("https://api.anthropic.com", "m", "p",
                              mode="anthropic", max_retries=3)
    assert "credit balance is too low" in str(exc.value)


def test_erro_em_formato_ollama_tambem_e_lido(monkeypatch):
    """O Ollama usa {"error": "..."} em vez de {"error": {"message": ...}}."""
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _RespostaComCorpo(
        {"error": "model 'inexistente' not found"}))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ModelError) as exc:
        call_model_with_retry("http://localhost:11434", "inexistente", "p",
                              mode="ollama", max_retries=1)
    assert "not found" in str(exc.value)


def test_corpo_sem_json_cai_no_texto_cru(monkeypatch):
    class _Html(_Resposta):
        text = "<html>502 Bad Gateway</html>"

        def json(self):
            raise ValueError("não é JSON")

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Html({}, status=400))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ModelError) as exc:
        call_model_with_retry("https://gpu:8000", "m", "p",
                              mode="openai_compat", max_retries=1)
    assert "502 Bad Gateway" in str(exc.value)


def test_mensagem_conta_as_tentativas_realmente_feitas(monkeypatch):
    """Um 400 sai à primeira: dizer "todas as 3 falharam" seria mentira."""
    tentativas = []

    def falso_post(*a, **k):
        tentativas.append(1)
        return _Resposta({}, status=400)

    monkeypatch.setattr(httpx, "post", falso_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ModelError) as exc:
        call_model_with_retry("https://gpu:8000", "m", "p",
                              mode="openai_compat", max_retries=3)
    assert len(tentativas) == 1
    assert "1 tentativa falhou" in str(exc.value)
    assert "3 tentativas" not in str(exc.value)


def test_mensagem_conta_as_tres_quando_houve_tres(monkeypatch):
    """O contrário também: um erro transitório insiste, e diz que insistiu."""
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: (_ for _ in ()).throw(
                            httpx.ConnectError("sem rede")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ModelError) as exc:
        call_model_with_retry("https://gpu:8000", "m", "p",
                              mode="openai_compat", max_retries=3)
    assert "3 tentativas" in str(exc.value)


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
