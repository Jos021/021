"""Esquema fixo dos relatórios de avaliação.

Activa a decisão que tinha sido conscientemente adiada na concepção: em vez
de ler percentagens com expressões regulares sobre texto livre, o modelo
devolve um objecto JSON com esquema fixo.

    {
      "functionality_pct": 87,
      "failures": ["..."],
      "improvements": { "neuron_2": "...", "neuron_5": "..." },
      "auto_reject": false
    }

O regex mantém-se apenas como recurso de último caso: se o JSON não fizer
parse, cai-se nele e regista-se um aviso na SYNAPSE DB de que o modelo não
respeitou o formato — informação útil para o piloto com modelos reais, que
é onde se vai descobrir que modelos cumprem o contrato e quais não.
"""

import json
import re

# Instrução acrescentada aos prompts que pedem uma avaliação.
INSTRUCAO_JSON = (
    "Responde APENAS com um objecto JSON, sem texto antes nem depois, sem "
    "blocos de código, exactamente com este formato:\n"
    '{"functionality_pct": <numero 0-100>, '
    '"failures": ["<falha>", ...], '
    '"improvements": {"neuron_N": "<melhoria específica>", ...}, '
    '"auto_reject": <true|false>}'
)

_PCT_ROTULADA = re.compile(r"PCT:\s*([\d.]+)")
_PCT_PERCENTAGEM = re.compile(r"([\d.]+)\s*%")
_MELHORIA = re.compile(r"neuron_(\d+)\s*:\s*(.+)", re.IGNORECASE)
# Bloco JSON mais exterior, para o caso de o modelo o embrulhar em texto.
_BLOCO_JSON = re.compile(r"\{.*\}", re.DOTALL)


# Breakdown vazio: é o que se usa quando a sandbox evolutiva está desligada
# ou quando o modelo não reporta nada. Nulos, não zeros — zero significaria
# "correu e falhou tudo", que é diferente de "não correu".
BREAKDOWN_VAZIO = {"level_1": None, "level_2": None, "level_3": None}


class Relatorio:
    """Avaliação já normalizada, venha ela de JSON ou do regex de recurso.

    Os campos da sandbox evolutiva são opcionais e retrocompatíveis: um
    relatório sem eles continua a ser válido e usa os defaults.
    """

    def __init__(self, functionality_pct=0.0, failures=None, improvements=None,
                 auto_reject=False, via="json", bruto="",
                 test_breakdown=None, new_tests_generated=0,
                 inherited_tests_used=0, tests_to_persist=None):
        self.functionality_pct = functionality_pct
        self.failures = failures or []
        self.improvements = improvements or {}
        self.auto_reject = auto_reject
        self.via = via              # 'json' | 'json_embrulhado' | 'regex'
        self.bruto = bruto
        # --- sandbox evolutiva ---
        self.test_breakdown = (
            test_breakdown if test_breakdown is not None
            else dict(BREAKDOWN_VAZIO)
        )
        self.new_tests_generated = new_tests_generated
        self.inherited_tests_used = inherited_tests_used
        self.tests_to_persist = tests_to_persist or []

    @property
    def formato_respeitado(self) -> bool:
        return self.via == "json"


def parse_relatorio(texto: str) -> Relatorio:
    """Lê a avaliação de um modelo. Nunca lança.

    Ordem de tentativas:
      1. JSON directo — o formato pedido
      2. JSON embrulhado em texto ou em blocos de código
      3. Regex sobre texto livre — recurso de último caso
    """
    if not texto:
        return Relatorio(via="regex", bruto="")

    dados, via = _tentar_json(texto)
    if dados is not None:
        return _do_json(dados, via, texto)
    return _do_regex(texto)


def _tentar_json(texto: str):
    """Devolve (dados, via) ou (None, None)."""
    try:
        return json.loads(texto.strip()), "json"
    except (ValueError, TypeError):
        pass
    # O modelo pode ter embrulhado o JSON em ```json ... ``` ou em prosa.
    bloco = _BLOCO_JSON.search(texto)
    if bloco:
        try:
            return json.loads(bloco.group(0)), "json_embrulhado"
        except (ValueError, TypeError):
            pass
    return None, None


def _do_json(dados, via: str, bruto: str) -> Relatorio:
    if not isinstance(dados, dict):
        return _do_regex(bruto)

    pct = dados.get("functionality_pct")
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        # JSON válido mas sem a percentagem: o campo essencial falta,
        # portanto o formato não foi respeitado.
        return _do_regex(bruto)
    pct = max(0.0, min(100.0, pct))

    falhas = dados.get("failures") or []
    if isinstance(falhas, str):
        falhas = [falhas]
    falhas = [str(f) for f in falhas] if isinstance(falhas, list) else []

    melhorias = dados.get("improvements") or {}
    if not isinstance(melhorias, dict):
        melhorias = {}
    melhorias = {str(k).lower(): str(v) for k, v in melhorias.items()}

    # --- campos da sandbox evolutiva (opcionais, com defaults) ---
    breakdown = dados.get("test_breakdown")
    if not isinstance(breakdown, dict):
        breakdown = dict(BREAKDOWN_VAZIO)
    else:
        breakdown = {**BREAKDOWN_VAZIO, **breakdown}
    persistir = dados.get("tests_to_persist")
    persistir = [str(t) for t in persistir] if isinstance(persistir, list) else []

    return Relatorio(
        functionality_pct=pct,
        failures=falhas,
        improvements=melhorias,
        auto_reject=bool(dados.get("auto_reject", False)),
        via=via,
        bruto=bruto,
        test_breakdown=breakdown,
        new_tests_generated=_inteiro(dados.get("new_tests_generated")),
        inherited_tests_used=_inteiro(dados.get("inherited_tests_used")),
        tests_to_persist=persistir,
    )


def _inteiro(valor) -> int:
    """Converte para inteiro sem lançar; 0 quando não é possível."""
    try:
        return int(valor)
    except (TypeError, ValueError):
        return 0


def _do_regex(texto: str) -> Relatorio:
    """Recurso de último caso, para modelos que não respeitam o formato."""
    pct = 0.0
    m = _PCT_ROTULADA.search(texto) or _PCT_PERCENTAGEM.search(texto)
    if m:
        try:
            pct = max(0.0, min(100.0, float(m.group(1))))
        except ValueError:
            pct = 0.0
    melhorias = {
        f"neuron_{m.group(1)}": m.group(2).strip()
        for m in _MELHORIA.finditer(texto)
    }
    return Relatorio(functionality_pct=pct, improvements=melhorias,
                     via="regex", bruto=texto)


def registar_desvio_de_formato(db, cycle_id: int, iteration: int,
                               component: str, relatorio: Relatorio) -> None:
    """Regista na SYNAPSE DB que o modelo não respeitou o esquema.

    Silencioso em caso de falha: um aviso de formato nunca pode derrubar o
    ciclo. Fica como informação para o piloto com modelos reais.
    """
    if relatorio.formato_respeitado:
        return
    try:
        db.log_decision(
            cycle_id, iteration, component,
            f"Modelo não respeitou o esquema JSON (via '{relatorio.via}') — "
            "avaliação lida por recurso.",
        )
    except Exception:
        pass
