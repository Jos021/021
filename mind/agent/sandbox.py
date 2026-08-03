"""Sandbox multi-linguagem de testes.

Execução isolada, restrita ao workspace/, com timeout por runner
(MUNDJI_SANDBOX_TIMEOUT). A detecção de linguagem usa a ANOTAÇÃO DO
MARCADOR, nunca heurística sobre o conteúdo do código.

Justificação: matching de strings (ex: "def " in code) é frágil — a mesma
string pode aparecer dentro de comentários, docstrings ou strings literais
sem relação com a linguagem real. A linguagem de cada secção já é conhecida
desde a Fase 1, através da anotação [NEURON_N:linguagem] feita pelo CORTEX.

Runners suportados (cada um respeita MUNDJI_SANDBOX_TIMEOUT):
    Python  -> subprocess.run(["python3", "-c", code], timeout=...)
    Rust    -> escreve main.rs, rustc main.rs && ./main
    Go      -> escreve main.go, go run main.go
    C       -> escreve main.c, gcc main.c -o main && ./main
    C#      -> escreve script.csx, dotnet script script.csx

Cada runner corre isolado, restrito ao workspace/, sem herdar variáveis de
ambiente sensíveis do processo pai.

--------------------------------------------------------------------------
DECISÃO CONSCIENTE DE NÃO-FAZER: isolamento por contentores
--------------------------------------------------------------------------
O isolamento actual é por subprocess, com ambiente mínimo e directório
restrito ao workspace/. NÃO se usam contentores, namespaces nem limites de
recursos, e isso é uma decisão tomada, não uma lacuna por resolver.

Razão: o MIND gera e executa o seu próprio código, não código hostil de
terceiros. Reforçar com contentores agora seria complexidade prematura — o
mesmo tipo que foi recusado com o Redis, com o ML embutido nos agentes e
com as oito bases de dados separadas.

Caminho futuro, se e quando o MIND passar a correr código não confiável
(por exemplo, código submetido por terceiros ou obtido da internet):
contentores efémeros ou namespaces com limites de CPU, memória e rede. Até
lá, os princípios de segurança aplicados são os documentados acima.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class SandboxResult:
    """Resultado da execução de uma secção de código."""

    language: str
    success: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


# Variáveis de ambiente mínimas — não herdamos o ambiente sensível do pai.
def _minimal_env(workdir: str) -> dict:
    """Ambiente mínimo e seguro para os runners."""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": "C.UTF-8",
    }


def get_language_for_section(neuron_id: str, markers: dict) -> str:
    """Lê a linguagem declarada no marcador correspondente.

    Default: python, se a secção não tiver anotação de linguagem.
    """
    return markers.get(neuron_id, {}).get("language", "python")


def _sandbox_root() -> str:
    """Raiz onde os runners escrevem/executam — sempre dentro do workspace."""
    ws = os.getenv("MUNDJI_WORKSPACE", "./workspace")
    root = os.path.join(ws, ".sandbox")
    os.makedirs(root, exist_ok=True)
    return os.path.abspath(root)


def _timeout() -> int:
    return int(os.getenv("MUNDJI_SANDBOX_TIMEOUT", "30"))


def run_section(code: str, language: str) -> SandboxResult:
    """Executa uma secção de código na linguagem indicada."""
    language = (language or "python").lower()
    runners = {
        "python": _run_python,
        "rust": _run_rust,
        "go": _run_go,
        "c": _run_c,
        "csharp": _run_csharp,
        "c#": _run_csharp,
        "cs": _run_csharp,
    }
    runner = runners.get(language)
    if runner is None:
        return SandboxResult(
            language=language,
            success=False,
            stdout="",
            stderr=f"Linguagem não suportada pelo sandbox: {language}",
            returncode=-1,
        )
    return runner(code)


def _exec(cmd: list, workdir: str, code_via_stdin: bool = False) -> SandboxResult:
    """Executa um comando isolado, capturando saída e respeitando timeout."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            env=_minimal_env(workdir),
            capture_output=True,
            text=True,
            timeout=_timeout(),
        )
        return SandboxResult(
            language="",
            success=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(
            language="",
            success=False,
            stdout=exc.stdout or "",
            stderr=f"Timeout após {_timeout()}s",
            returncode=-1,
            timed_out=True,
        )
    except FileNotFoundError as exc:
        # Toolchain ausente (ex: rustc não instalado) — reporta, não rebenta.
        return SandboxResult(
            language="",
            success=False,
            stdout="",
            stderr=f"Toolchain não disponível: {exc}",
            returncode=-1,
        )


def _run_python(code: str) -> SandboxResult:
    root = _sandbox_root()
    res = _exec(["python3", "-c", code], root)
    res.language = "python"
    return res


def _run_rust(code: str) -> SandboxResult:
    root = _sandbox_root()
    with tempfile.TemporaryDirectory(dir=root) as d:
        src = os.path.join(d, "main.rs")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(code)
        compile_res = _exec(["rustc", "main.rs"], d)
        if not compile_res.success:
            compile_res.language = "rust"
            return compile_res
        run_res = _exec([os.path.join(d, "main")], d)
        run_res.language = "rust"
        return run_res


def _run_go(code: str) -> SandboxResult:
    root = _sandbox_root()
    with tempfile.TemporaryDirectory(dir=root) as d:
        src = os.path.join(d, "main.go")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(code)
        res = _exec(["go", "run", "main.go"], d)
        res.language = "go"
        return res


def _run_c(code: str) -> SandboxResult:
    root = _sandbox_root()
    with tempfile.TemporaryDirectory(dir=root) as d:
        src = os.path.join(d, "main.c")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(code)
        compile_res = _exec(["gcc", "main.c", "-o", "main"], d)
        if not compile_res.success:
            compile_res.language = "c"
            return compile_res
        run_res = _exec([os.path.join(d, "main")], d)
        run_res.language = "c"
        return run_res


def _run_csharp(code: str) -> SandboxResult:
    root = _sandbox_root()
    with tempfile.TemporaryDirectory(dir=root) as d:
        src = os.path.join(d, "script.csx")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(code)
        res = _exec(["dotnet", "script", "script.csx"], d)
        res.language = "csharp"
        return res
