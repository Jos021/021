"""Filtro de sanitização de output, aplicado antes da compilação final."""

import pytest

from agent.sanitizer import sanitize_output


def test_desligado_devolve_codigo_intacto(monkeypatch):
    monkeypatch.setenv("OUTPUT_SANITIZER_ENABLED", "false")
    codigo = 'API = "{CHAVE_API}"'
    assert sanitize_output(codigo) == codigo


# --- 1. Placeholders por valores realistas por tipo ----------------------
@pytest.mark.parametrize("placeholder,esperado", [
    ("{CHAVE_API}", "sk-exemplo0000000000000000000000"),
    ("{API_KEY}", "sk-exemplo0000000000000000000000"),
    ("{IP}", "10.0.0.1"),
    ("{DOMINIO}", "exemplo.local"),
    ("{DOMAIN}", "exemplo.local"),
])
def test_placeholders_substituidos(placeholder, esperado):
    assert esperado in sanitize_output(f'x = "{placeholder}"')


def test_valor_substituido_mantem_o_formato_esperado():
    """O valor não pode ser genérico ao ponto de quebrar o formato."""
    saida = sanitize_output('chave = "{CHAVE_API}"')
    assert saida.startswith('chave = "sk-')


# --- 2. IPs e domínios reais não anotados como exemplo -------------------
def test_ip_publico_e_substituido():
    assert "8.8.8.8" not in sanitize_output('dns = "8.8.8.8"')


@pytest.mark.parametrize("ip", ["10.1.2.3", "192.168.1.1", "172.16.0.5",
                                "127.0.0.1"])
def test_ips_privados_sao_preservados(ip):
    """Gamas privadas e loopback já são exemplos aceitáveis."""
    assert ip in sanitize_output(f'host = "{ip}"')


def test_ip_anotado_como_exemplo_e_preservado():
    assert "8.8.8.8" in sanitize_output('host = "8.8.8.8"  # exemplo')


def test_dominio_real_e_substituido():
    assert "exemplo.local" in sanitize_output('url = "https://empresa.com"')


def test_dominio_local_e_preservado():
    assert "servidor.local" in sanitize_output('url = "servidor.local"')


# --- 3. Acesso a recursos fora do workspace ------------------------------
@pytest.mark.parametrize("linha", [
    'open("/etc/passwd")',
    'f = open("/root/.ssh/id_rsa")',
    'caminho = "../../segredo"',
    'os.system("rm -rf /")',
    'shutil.rmtree("/dados")',
])
def test_acesso_externo_e_comentado(linha):
    saida = sanitize_output(linha)
    assert "SANITIZADO" in saida
    assert saida.lstrip().startswith(("#", "//")), \
        "a linha tem de ficar comentada, não activa"


def test_codigo_legitimo_nao_e_comentado():
    codigo = "def somar(a, b):\n    return a + b"
    assert "SANITIZADO" not in sanitize_output(codigo)


# --- 4. Comentários com palavras sensíveis -------------------------------
@pytest.mark.parametrize("comentario", [
    "# a senha do admin e 1234",
    "# TODO: guardar o token aqui",
    "// the password is hunter2",
    "# chave_privada em falta",
])
def test_comentario_sensivel_e_removido(comentario):
    saida = sanitize_output(comentario)
    assert "SANITIZADO" in saida
    for termo in ("1234", "hunter2"):
        assert termo not in saida


def test_codigo_com_palavra_sensivel_fora_de_comentario_e_mantido():
    """Só comentários são apagados — apagar código partiria o programa."""
    codigo = 'password = obter_password()'
    assert "password" in sanitize_output(codigo)


def test_sanitizacao_e_idempotente():
    uma_vez = sanitize_output('k = "{CHAVE_API}"\n# a senha e 1234')
    assert sanitize_output(uma_vez) == uma_vez
