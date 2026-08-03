"""Sandbox multi-linguagem.

Os testes de linguagens que exigem toolchain externo (rustc, go, gcc,
dotnet) são saltados quando a ferramenta não existe — o que se valida
sempre é que a ausência de toolchain é reportada, nunca propagada como
excepção.
"""

import shutil

import pytest

from agent.sandbox import get_language_for_section, run_section


def test_python_executa_e_captura_stdout():
    res = run_section("print(2 + 2)", "python")
    assert res.success
    assert res.stdout.strip() == "4"
    assert res.language == "python"


def test_python_com_erro_reporta_falha():
    res = run_section("raise ValueError('rebentou')", "python")
    assert not res.success
    assert "ValueError" in res.stderr


def test_timeout_e_assinalado(monkeypatch):
    monkeypatch.setenv("MUNDJI_SANDBOX_TIMEOUT", "1")
    res = run_section("import time; time.sleep(5)", "python")
    assert res.timed_out
    assert not res.success


def test_linguagem_nao_suportada_e_reportada_sem_excepcao():
    res = run_section("codigo", "cobol")
    assert not res.success
    assert "não suportada" in res.stderr


def test_linguagem_vazia_assume_python():
    assert run_section("print('ok')", "").success


def test_ambiente_nao_herda_variaveis_sensiveis(monkeypatch):
    """O runner não pode ver segredos do processo pai."""
    monkeypatch.setenv("SEGREDO_DO_PAI", "nao-devia-passar")
    res = run_section(
        "import os; print(os.environ.get('SEGREDO_DO_PAI', 'AUSENTE'))",
        "python",
    )
    assert res.stdout.strip() == "AUSENTE"


def test_execucao_ocorre_dentro_do_workspace(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    monkeypatch.setenv("MUNDJI_WORKSPACE", str(ws))
    res = run_section("import os; print(os.getcwd())", "python")
    assert str(ws.resolve()) in res.stdout


# Sinais de que a ferramenta existe mas não está utilizável (ex: rustc
# instalado via rustup sem toolchain por omissão). Não é falha do sandbox.
_TOOLCHAIN_INDISPONIVEL = (
    "rustup could not choose", "no default toolchain",
    "toolchain não disponível", "command not found",
    "cannot find", "no such file",
)


def _toolchain_utilizavel(ferramenta: str, linguagem: str, codigo: str) -> bool:
    """A ferramenta existe E consegue mesmo compilar/executar algo trivial."""
    if shutil.which(ferramenta) is None:
        return False
    res = run_section(codigo, linguagem)
    if res.success:
        return True
    erro = (res.stderr or "").lower()
    return not any(sinal in erro for sinal in _TOOLCHAIN_INDISPONIVEL)


@pytest.mark.parametrize("linguagem,ferramenta,codigo", [
    ("rust", "rustc", 'fn main() { println!("42"); }'),
    ("go", "go", 'package main\nimport "fmt"\nfunc main() { fmt.Println("42") }'),
    ("c", "gcc", '#include <stdio.h>\nint main(){printf("42");return 0;}'),
])
def test_linguagens_compiladas(linguagem, ferramenta, codigo):
    if not _toolchain_utilizavel(ferramenta, linguagem, codigo):
        pytest.skip(f"toolchain de {linguagem} não utilizável neste ambiente")
    res = run_section(codigo, linguagem)
    assert res.success, res.stderr
    assert "42" in res.stdout


@pytest.mark.parametrize("linguagem,ferramenta", [
    ("rust", "rustc"), ("go", "go"), ("c", "gcc"), ("csharp", "dotnet"),
])
def test_toolchain_em_falta_degrada_sem_rebentar(linguagem, ferramenta):
    """Sem toolchain, o runner reporta o problema em vez de lançar."""
    if shutil.which(ferramenta) is not None:
        pytest.skip(f"{ferramenta} está instalado — caso não aplicável")
    res = run_section("codigo qualquer", linguagem)
    assert not res.success
    assert res.stderr, "a razão da falha tem de ser reportada"


# --- Detecção de linguagem: sempre pelo marcador -------------------------
def test_get_language_usa_o_marcador():
    markers = {"neuron_1": {"language": "rust"}}
    assert get_language_for_section("neuron_1", markers) == "rust"


def test_get_language_default_python_quando_nao_anotado():
    assert get_language_for_section("neuron_1", {"neuron_1": {}}) == "python"
    assert get_language_for_section("neuron_9", {}) == "python"
