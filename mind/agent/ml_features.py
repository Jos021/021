"""Extracção de features para o HIPPOCAMPUS.

Traduz o estado operacional do MIND (tarefa, código, resultados de testes,
histórico da SYNAPSE DB) em vectores numéricos utilizáveis por modelos
clássicos de ML.

Princípio: tudo local e tolerante a dependências em falta. As bibliotecas
pesadas (sentence-transformers, radon) são importadas de forma preguiçosa e
têm sempre um fallback determinístico — a ausência de uma dependência nunca
lança excepção nem impede o ciclo do MIND de correr.

Features por consumidor (ver config/ml_config.yaml):
  cortex_support     -> task_embedding, task_keywords, history_success_rate
  cerebellum_support -> code_complexity, test_coverage, failure_patterns,
                        history_approval_rate
"""

import hashlib
import math
import re

# Dimensão do embedding de fallback (hashing determinístico).
_FALLBACK_EMBEDDING_DIM = 32

# Palavras-chave técnicas cuja presença na tarefa é sinal útil.
_KEYWORDS = [
    "api", "auth", "autenticacao", "base de dados", "cli", "cripto",
    "encriptacao", "http", "json", "jwt", "parser", "pentest", "rede",
    "scanner", "sql", "teste", "token", "web", "payload", "socket",
]

# Padrões de falha reconhecíveis na saída dos testes.
_FAILURE_PATTERNS = [
    "traceback", "error", "erro", "exception", "failed", "falhou",
    "timeout", "segmentation fault", "panic", "cannot find",
    "não suportada", "toolchain não disponível",
]

_sentence_model = None
_sentence_model_tried = False


def _get_sentence_model():
    """Carrega o modelo de embeddings uma única vez (se disponível).

    sentence-transformers puxa torch, que compete com os LLMs pela GPU. Se
    não estiver instalado, devolvemos None e usamos o fallback por hashing.
    """
    global _sentence_model, _sentence_model_tried
    if _sentence_model_tried:
        return _sentence_model
    _sentence_model_tried = True
    try:
        from sentence_transformers import SentenceTransformer

        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _sentence_model = None
    return _sentence_model


def task_embedding(task: str) -> list:
    """Embedding da descrição da tarefa.

    Usa sentence-transformers se disponível; caso contrário, um embedding
    determinístico por hashing de tokens (bag-of-hashes normalizado), que
    não é tão expressivo mas é estável, local e sem dependências.
    """
    model = _get_sentence_model()
    if model is not None:
        try:
            return [float(x) for x in model.encode(task or "")]
        except Exception:
            pass  # cai para o fallback
    return _hash_embedding(task or "")


def _hash_embedding(text: str, dim: int = _FALLBACK_EMBEDDING_DIM) -> list:
    """Bag-of-hashes normalizado — determinístico e sem dependências."""
    vec = [0.0] * dim
    for token in re.findall(r"[0-9a-zà-ÿ_]+", text.lower()):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


# Dimensão fixa dos embeddings de tarefa guardados na test_library. É a do
# all-MiniLM-L6-v2; o fallback determinístico enche o resto com zeros para
# que a distância cosseno funcione entre vectores das duas origens.
EMBEDDING_DIM = 384


def embed_task(task: str) -> list | None:
    """Gera um embedding da descrição da tarefa, com 384 dimensões.

    COM sentence-transformers disponível:
        modelo leve (all-MiniLM-L6-v2), vector de 384 dimensões.

    SEM sentence-transformers (caso inicial, ML_ENABLED=false):
        usa o hashing determinístico já existente neste módulo, enchendo
        as posições restantes com zeros para manter a compatibilidade com
        a distância cosseno. Menos preciso semanticamente, mas funcional
        para recuperação aproximada — e é o que permite a biblioteca de
        testes crescer antes de haver camada de ML.

    Se tudo falhar, devolve None sem lançar. O código chamador guarda
    task_embedding=None nesse caso, e get_tests_by_embedding trata None
    como lista vazia.
    """
    if not task:
        return None
    model = _get_sentence_model()
    if model is not None:
        try:
            vector = [float(x) for x in model.encode(task)]
            if vector:
                return _ajustar_dimensao(vector)
        except Exception:
            pass  # cai para o fallback determinístico
    try:
        return _ajustar_dimensao(_hash_embedding(task, dim=EMBEDDING_DIM))
    except Exception:
        return None


def _ajustar_dimensao(vector: list, dim: int = EMBEDDING_DIM) -> list:
    """Trunca ou enche com zeros até à dimensão fixa."""
    if len(vector) >= dim:
        return vector[:dim]
    return vector + [0.0] * (dim - len(vector))


def cosine_similarity(a: list, b: list) -> float:
    """Similaridade cosseno entre dois vectores.

    Devolve 0.0 se qualquer dos vectores for None, vazio, de comprimentos
    incompatíveis, ou de norma nula — nunca lança.
    """
    if not a or not b:
        return 0.0
    try:
        n = min(len(a), len(b))
        produto = sum(float(a[i]) * float(b[i]) for i in range(n))
        norma_a = math.sqrt(sum(float(x) * float(x) for x in a[:n]))
        norma_b = math.sqrt(sum(float(x) * float(x) for x in b[:n]))
        if norma_a == 0 or norma_b == 0:
            return 0.0
        return produto / (norma_a * norma_b)
    except (TypeError, ValueError):
        return 0.0


def task_keywords(task: str) -> dict:
    """Presença (0/1) de cada palavra-chave técnica na tarefa."""
    lowered = (task or "").lower()
    return {kw: (1.0 if kw in lowered else 0.0) for kw in _KEYWORDS}


def code_complexity(code: str) -> float:
    """Complexidade ciclomática média do código.

    Usa radon quando disponível (só faz sentido para Python); caso
    contrário, aproxima pela densidade de estruturas de controlo.
    """
    if not code:
        return 0.0
    try:
        from radon.complexity import cc_visit

        blocks = cc_visit(code)
        if blocks:
            return float(sum(b.complexity for b in blocks) / len(blocks))
    except Exception:
        pass
    # Fallback: densidade de palavras de controlo de fluxo por linha.
    control = len(
        re.findall(
            r"\b(if|else|elif|for|while|match|case|try|except|switch)\b", code
        )
    )
    lines = max(1, len(code.splitlines()))
    return float(control) / lines * 10.0


def test_coverage(test_results: str) -> float:
    """Proporção de secções testadas com sucesso (0.0-1.0).

    Lida directamente com o formato produzido por Cortex.test(), onde cada
    secção reporta 'success=True|False'.
    """
    if not test_results:
        return 0.0
    successes = len(re.findall(r"success=True", test_results))
    failures = len(re.findall(r"success=False", test_results))
    total = successes + failures
    return float(successes) / total if total else 0.0


def failure_patterns(test_results: str) -> float:
    """Número de padrões de falha conhecidos presentes na saída dos testes."""
    if not test_results:
        return 0.0
    lowered = test_results.lower()
    return float(sum(1 for p in _FAILURE_PATTERNS if p in lowered))


def history_success_rate(db) -> float:
    """Taxa histórica de ciclos aprovados (0.0-1.0)."""
    try:
        with db._lock:  # noqa: SLF001 - acesso interno controlado
            row = db._conn.execute(
                """SELECT
                       SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) AS ok,
                       COUNT(*) AS total
                   FROM cycles"""
            ).fetchone()
        if row and row["total"]:
            return float(row["ok"] or 0) / float(row["total"])
    except Exception:
        pass
    return 0.0


def history_approval_rate(db) -> float:
    """Taxa histórica de relatórios que atingiram o threshold de aprovação."""
    import os

    threshold = float(os.getenv("APPROVAL_THRESHOLD", "98"))
    try:
        with db._lock:  # noqa: SLF001
            row = db._conn.execute(
                """SELECT
                       SUM(CASE WHEN functionality_pct >= ? THEN 1 ELSE 0 END) AS ok,
                       COUNT(*) AS total
                   FROM reports""",
                (threshold,),
            ).fetchone()
        if row and row["total"]:
            return float(row["ok"] or 0) / float(row["total"])
    except Exception:
        pass
    return 0.0


# --------------------------------------------------------------------------
# Construção dos dicionários de features por consumidor
# --------------------------------------------------------------------------
def extract_cortex_features(task: str, db=None) -> dict:
    """Features para o apoio ao CORTEX (que abordagem funcionou no passado)."""
    return {
        "task_embedding": task_embedding(task),
        "task_keywords": task_keywords(task),
        "history_success_rate": history_success_rate(db) if db else 0.0,
    }


def extract_cerebellum_features(code: str, test_results: str, db=None) -> dict:
    """Features para o apoio ao CEREBELLUM (zonas de risco, probabilidade)."""
    return {
        "code_complexity": code_complexity(code),
        "test_coverage": test_coverage(test_results),
        "failure_patterns": failure_patterns(test_results),
        "history_approval_rate": history_approval_rate(db) if db else 0.0,
    }


def vectorize(features: dict, feature_names: list) -> list:
    """Achata um dicionário de features num vector numérico estável.

    A ordem é determinada por `feature_names` (vinda do ml_config.yaml) e,
    dentro de cada feature composta, pela ordem alfabética das chaves — o
    que garante que treino e inferência produzem sempre o mesmo layout.
    """
    vector: list = []
    for name in feature_names:
        value = features.get(name, 0.0)
        if isinstance(value, dict):
            vector.extend(float(value[k]) for k in sorted(value))
        elif isinstance(value, (list, tuple)):
            vector.extend(float(x) for x in value)
        elif isinstance(value, bool):
            vector.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            vector.append(float(value))
        else:
            vector.append(0.0)
    return vector
