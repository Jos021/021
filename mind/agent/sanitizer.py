"""Filtro de sanitização de output.

Aplicado antes da compilação final (cortex_approve), só quando
OUTPUT_SANITIZER_ENABLED=true. Substitui placeholders por valores realistas
por tipo, detecta IPs/domínios reais não anotados como exemplo, remove/comenta
acessos a recursos fora do workspace/ e comenta linhas com palavras sensíveis.
"""

import os
import re

# 1. Placeholders -> valores realistas por tipo (não genéricos que quebrem
#    o formato esperado pelo código que os consome).
PLACEHOLDER_MAP = {
    "{CHAVE_API}": "sk-exemplo0000000000000000000000",
    "{API_KEY}": "sk-exemplo0000000000000000000000",
    "{IP}": "10.0.0.1",              # exemplo documentado (RFC1918)
    "{DOMINIO}": "exemplo.local",
    "{DOMAIN}": "exemplo.local",
}

# 2. Detecção de IPs reais (não RFC1918/loopback) e domínios não anotados.
_IP_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
_DOMAIN_RE = re.compile(
    r"\b(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|io|dev|ao|pt)\b"
)

# 4. Palavras sensíveis em comentários.
_SENSITIVE_WORDS = [
    "password", "passwd", "senha", "secret", "segredo", "token",
    "private_key", "chave_privada", "credential", "credencial",
]

# 3. Padrões de acesso a recursos fora do workspace/.
_OUTSIDE_ACCESS_RE = re.compile(
    r"""(open\s*\(\s*['"]/(?!.*workspace)"""   # open("/...") absoluto
    r"""|['"]\.\./"""                             # caminhos ../
    r"""|/etc/|/root/|/home/(?!.*workspace)"""    # dirs sensíveis
    r"""|os\.system|subprocess\.|shutil\.rmtree)""",
    re.IGNORECASE,
)


def _is_example_ip(match: re.Match) -> bool:
    """True se o IP for de uma gama de exemplo/privada aceitável."""
    a, b = int(match.group(1)), int(match.group(2))
    if a == 10:                       # 10.0.0.0/8
        return True
    if a == 192 and b == 168:         # 192.168.0.0/16
        return True
    if a == 172 and 16 <= b <= 31:    # 172.16.0.0/12
        return True
    if a == 127:                      # loopback
        return True
    if a == 0 or a > 255 or b > 255:  # inválido -> ignorar
        return True
    return False


def _comment_prefix_for(line: str) -> str:
    """Escolhe o prefixo de comentário adequado à linha (heurística leve)."""
    stripped = line.lstrip()
    if stripped.startswith(("//", "/*")) or "fn " in line or "func " in line:
        return "//"
    return "#"


def sanitize_output(code: str) -> str:
    """Aplica todo o filtro de sanitização e devolve o código limpo."""
    if os.getenv("OUTPUT_SANITIZER_ENABLED", "true").lower() != "true":
        return code

    # 1. Substituir placeholders por valores realistas.
    for placeholder, value in PLACEHOLDER_MAP.items():
        code = code.replace(placeholder, value)

    out_lines = []
    for line in code.splitlines():
        new_line = line

        # 2. IPs reais não anotados como exemplo -> substituir por exemplo.
        def _ip_sub(m: re.Match) -> str:
            if _is_example_ip(m):
                return m.group(0)
            if "exemplo" in line.lower() or "example" in line.lower():
                return m.group(0)
            return "10.0.0.1"

        new_line = _IP_RE.sub(_ip_sub, new_line)

        # Domínios reais não anotados -> exemplo.local.
        def _dom_sub(m: re.Match) -> str:
            dom = m.group(0)
            if dom.endswith(".local") or "exemplo" in line.lower():
                return dom
            return "exemplo.local"

        new_line = _DOMAIN_RE.sub(_dom_sub, new_line)

        # 3. Acesso a recursos fora do workspace/ -> comentar.
        if _OUTSIDE_ACCESS_RE.search(new_line) and not new_line.lstrip().startswith(
            ("#", "//")
        ):
            prefix = _comment_prefix_for(new_line)
            new_line = f"{prefix} [SANITIZADO: acesso externo removido] {new_line}"

        # 4. Comentários com palavras sensíveis -> remover conteúdo sensível.
        lowered = new_line.lower()
        is_comment = new_line.lstrip().startswith(("#", "//"))
        if is_comment and any(w in lowered for w in _SENSITIVE_WORDS):
            prefix = _comment_prefix_for(new_line)
            new_line = f"{prefix} [SANITIZADO: comentário sensível removido]"

        out_lines.append(new_line)

    return "\n".join(out_lines)
