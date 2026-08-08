"""Geração de testes pelo CEREBELLUM — sandbox evolutiva.

O CEREBELLUM gera os testes. O CORTEX NÃO gera os testes pelos quais vai ser
avaliado — esta separação é deliberada e inegociável.

Quando gera
-----------
Na transição Fase 1 -> Fase 2: depois do código base estar aprimorado e
ANTES dos NEURONS desenvolverem. Gerar neste momento — e não no fim — é uma
decisão de arquitectura: os testes são informados pela INTENÇÃO da tarefa,
não moldados ao código que foi gerado.

Os três níveis
--------------
  Nível 1 — BÁSICO ("basic"): caminho feliz, inputs normais e válidos.
  Nível 2 — LIMITE ("edge"): extremos e casos não óbvios.
  Nível 3 — ERRO ("error"): inputs inválidos e falhas esperadas.

Princípio de robustez: se a geração falhar por qualquer razão, devolve
lista vazia sem lançar. O ciclo nunca falha por causa da geração de testes.
"""

import json
import os
import uuid

from .model_router import ModelError
from .report_schema import _BLOCO_JSON

NIVEIS = {
    1: ("basic", "caminho feliz, com inputs normais e válidos"),
    2: ("edge", "extremos e casos não óbvios (vazios, um só elemento, "
                "valores None, entradas no limite)"),
    3: ("error", "inputs inválidos e falhas esperadas (tipos errados, "
                 "excepções, recursos inexistentes)"),
}

SYSTEM_GERACAO = (
    "És o CEREBELLUM do MIND, a gerar os testes pelos quais o código vai ser "
    "avaliado. Sem personalidade. Produzes apenas testes executáveis, "
    "concretos e verificáveis. Cada teste tem de correr isoladamente na "
    "sandbox e falhar de forma clara quando o código estiver errado. "
    "Descrições em português europeu."
)


def sandbox_tests_enabled() -> bool:
    """A extensão só entra em acção quando explicitamente activada."""
    return os.getenv("SANDBOX_TESTS_ENABLED", "false").lower() == "true"


class TestGenerator:
    """Gera testes nos três níveis e herda testes de ciclos anteriores."""

    # O nome começa por 'Test' (exigido pela especificação), mas isto
    # não é uma classe de testes do pytest — evita a recolha indevida.
    __test__ = False


    def __init__(self, model_router, db, hippocampus=None):
        self.router = model_router
        self.db = db
        self.hippocampus = hippocampus
        self.por_nivel = int(os.getenv("SANDBOX_TESTS_PER_LEVEL", "8"))
        self.min_similaridade = float(
            os.getenv("SANDBOX_MIN_LIBRARY_SIMILARITY", "0.75")
        )

    # ------------------------------------------------------------------
    def generate(self, task: str, base_code: str, markers: dict,
                 cycle_id: int) -> list:
        """Gera testes para os 3 níveis e devolve a lista completa.

        Inclui testes herdados de ciclos anteriores aprovados. Guarda todos
        na test_library. Se a geração falhar, devolve lista vazia sem lançar.
        """
        try:
            return self._generate_inner(task, base_code, markers, cycle_id)
        except Exception:
            # Nunca bloquear o ciclo por causa da geração de testes.
            return []

    def _generate_inner(self, task, base_code, markers, cycle_id) -> list:
        from .ml_features import embed_task

        embedding = embed_task(task)
        herdados = self._herdar(embedding)

        gerados = []
        for nivel in (1, 2, 3):
            gerados.extend(
                self._gerar_nivel(task, base_code, markers, nivel, herdados)
            )

        # Completar e persistir os testes novos.
        resumo = (task or "")[:200]
        for teste in gerados:
            teste["cycle_id"] = cycle_id
            teste["task_summary"] = resumo
            teste["task_embedding"] = embedding
            self.db.save_test(teste)

        # Os herdados já estão na biblioteca; entram na lista devolvida para
        # serem executados, mas não são regravados.
        return gerados + herdados

    def _herdar(self, embedding) -> list:
        """Testes de ciclos anteriores aprovados com tarefa semelhante."""
        if self.hippocampus is None:
            try:
                return self.db.get_tests_by_embedding(
                    embedding, self.min_similaridade, self.por_nivel
                )
            except Exception:
                return []
        try:
            return self.hippocampus.recommend_tests(
                task_embedding=embedding,
                limit=self.por_nivel,
                min_similarity=self.min_similaridade,
            )
        except Exception:
            return []

    def _gerar_nivel(self, task, base_code, markers, nivel, herdados) -> list:
        """Pede ao modelo os testes de um nível. Devolve [] se falhar."""
        categoria, descricao_nivel = NIVEIS[nivel]
        alvos = list(markers) or ["all"]
        linguagens = {
            nid: (markers.get(nid) or {}).get("language", "python")
            for nid in alvos
        }

        # Os herdados entram como contexto para o modelo não gerar
        # duplicados e poder partir de uma base já validada.
        contexto = ""
        ja_existentes = [t["description"] for t in herdados
                         if t.get("level") == nivel]
        if ja_existentes:
            contexto = (
                "\n\nJá existem estes testes deste nível, herdados de ciclos "
                "aprovados anteriores. NÃO os repitas; gera testes novos e "
                "complementares:\n- " + "\n- ".join(ja_existentes[:20]) + "\n"
            )

        prompt = (
            f"Tarefa a implementar:\n{task}\n\n"
            f"Código base com marcadores:\n{base_code}\n\n"
            f"Gera até {self.por_nivel} testes de NÍVEL {nivel} "
            f"(categoria '{categoria}'): {descricao_nivel}.\n"
            f"Secções a cobrir e respectivas linguagens: {linguagens}\n"
            + contexto +
            "\nResponde APENAS com um array JSON de objectos, sem texto antes "
            "nem depois e sem blocos de código. Cada objecto exactamente "
            'assim: {"neuron_target": "neuron_N ou all", "language": "...", '
            f'"level": {nivel}, "category": "{categoria}", '
            '"description": "...", "code": "<código executável do teste>", '
            '"expected_outcome": "pass ou fail"}'
        )

        bruto = self._pedir_ao_modelo(prompt)
        return self._parse(bruto, nivel, categoria, linguagens)

    def _pedir_ao_modelo(self, prompt: str) -> str:
        """Chama o modelo do CEREBELLUM. Devolve '' em caso de erro."""
        from .model_router import component_config

        endpoint, modelo, _ = component_config("cerebellum")
        if not modelo:
            return ""
        try:
            return self.router.generate(
                prompt=prompt, model=modelo, endpoint=endpoint,
                system=SYSTEM_GERACAO, component="cerebellum",
            )
        except ModelError:
            return ""

    def _parse(self, bruto: str, nivel: int, categoria: str,
               linguagens: dict) -> list:
        """Lê o array JSON de testes. Usa o mesmo fallback do relatório.

        Se o JSON não puder ser lido, devolve lista vazia — o ciclo continua
        sem testes gerados desse nível, e isso fica visível no breakdown.
        """
        if not bruto:
            return []
        dados = _extrair_array(bruto)
        if not isinstance(dados, list):
            return []

        omissao = next(iter(linguagens.values()), "python") if linguagens else "python"
        testes = []
        for item in dados:
            if not isinstance(item, dict) or not item.get("code"):
                continue
            alvo = str(item.get("neuron_target", "all"))
            testes.append({
                "test_id": str(uuid.uuid4()),
                "neuron_target": alvo,
                "language": str(item.get("language")
                                or linguagens.get(alvo, omissao)),
                "level": nivel,
                "category": categoria,
                "description": str(item.get("description", ""))[:500],
                "code": str(item["code"]),
                "expected_outcome": (
                    "fail" if str(item.get("expected_outcome", "pass")).lower()
                    == "fail" else "pass"
                ),
            })
        return testes[: self.por_nivel]


def _extrair_array(texto: str):
    """Extrai um array JSON, directo ou embrulhado em prosa/blocos."""
    try:
        return json.loads(texto.strip())
    except (ValueError, TypeError):
        pass
    inicio = texto.find("[")
    fim = texto.rfind("]")
    if inicio != -1 and fim > inicio:
        try:
            return json.loads(texto[inicio:fim + 1])
        except (ValueError, TypeError):
            pass
    # Um só objecto em vez de array também é aceitável.
    bloco = _BLOCO_JSON.search(texto)
    if bloco:
        try:
            unico = json.loads(bloco.group(0))
            return [unico] if isinstance(unico, dict) else None
        except (ValueError, TypeError):
            pass
    return None
