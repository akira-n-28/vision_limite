"""C9 di MAVIS-Geometry: valori esatti, visualizzati e richiesti separati; shape_type con opzioni costruite per istanza.

Solo libreria standard. La costruzione storica delle domande (f1_mavis.mavis_questions) resta quella predefinita e non
cambia: i manifest esistenti la ricalcolano con gli stessi digest. Questa costruzione, `mavis-c9/1`, si usa solo se il
manifest la dichiara nel record (`costruzione_qa`, letta da data.mavis_f1_sample).

Fatti, in tre livelli che non si mescolano:
- **annotazioni** (nascoste al modello): tipo dichiarato da diagram_generation (mavis_gen.figures), misure esatte
  (mavis_gen.measures, confermate dalla didascalia: mavis_gen.compare senza segnalazioni), vertice degli isosceli dalla
  frase di costruzione;
- **stampato** (disponibile al modello per la regola del renderer): le misure della didascalia, scritte come
  f1_schema.latex_value (razionali senza radice come decimali con al più due cifre, job 30049; un angolo di 90° come
  segno). È la rappresentazione prevista, non una lettura dei pixel;
- **da verificare a vista**: fedeltà del disegno, leggibilità, segni non dichiarati come misure (il segno dell'angolo
  retto di un triangolo rettangolo, osservato in contact_run1 ma non garantito), forma disegnata.

Valori: per ogni misura il valore esatto delle annotazioni (`44/3`), la rappresentazione visualizzata prevista (`14.67`)
e la risposta richiesta dai compiti di lettura (il visualizzato: i target storici non cambiano). Un esatto non si
ricostruisce da un decimale arrotondato.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from .f1_schema import angle_name, latex_value
from .mavis_gen import figures, measures

CONSTRUCTION = "mavis-c9/1"
HISTORICAL = "storica"

QUAD_KINDS = ("square", "rectangle", "rhombus", "parallelogram", "kite", "trapezoid")
TRIANGLE_KINDS = ("triangle", "right triangle", "isosceles triangle", "equilateral triangle")
ROUND_KINDS = ("sector", "semicircle", "circle")

VALUE_SOURCE = ("diagram_generation (mavis_gen.measures, forma di facts.norm_value), confermato dalla didascalia "
                "(mavis_gen.compare senza segnalazioni)")
DISPLAY_RULE = ("regola del renderer di f1_schema.latex_value (job 30049 e contact_run1): razionali senza radice come "
                "decimali con al più due cifre, radici e frazioni con radice in forma simbolica; prevista, non letta dai "
                "pixel")
HALF_ULP = Fraction(1, 200)  # metà dell'ultima cifra di un decimale a due cifre: intervallo di un valore arrotondato

_RAT = re.compile(r"^(\d+(?:\.\d+)?)(?:/(\d+))?$")
_ROOT = re.compile(r"^(\d+(?:\.\d+)?)?√(\d+)(?:/(\d+))?$")


# ---------------------------------------------------------------- valori esatti

def _squarefree(n: int) -> tuple[int, int]:
    """n = k² · m con m senza quadrati: (k, m)."""
    k, m, d = 1, n, 2
    while d * d <= m:
        while m % (d * d) == 0:
            m //= d * d
            k *= d
        d += 1
    return k, m


@dataclass(frozen=True)
class Exact:
    """Numero positivo coef·√rad, con coef razionale e rad intero senza quadrati (1 per i razionali)."""

    coef: Fraction
    rad: int

    @property
    def square(self) -> Fraction:
        return self.coef * self.coef * self.rad

    @property
    def rational(self) -> bool:
        return self.rad == 1


def parse_exact(value: str) -> Exact | None:
    """Valore esatto da una forma di facts.norm_value (`42`, `12.5`, `3/2`, `√2`, `2√3`, `10√3/3`); None se la forma
    è un'altra (radicando decimale, zero, testo)."""
    m = _RAT.fullmatch(value)
    if m:
        x = Fraction(m.group(1)) / (int(m.group(2)) if m.group(2) else 1)
        return Exact(x, 1) if x > 0 else None
    m = _ROOT.fullmatch(value)
    if m:
        coef = Fraction(m.group(1) or 1) / (int(m.group(3)) if m.group(3) else 1)
        k, rad = _squarefree(int(m.group(2)))
        return Exact(coef * k, rad) if coef > 0 and rad > 0 else None
    return None


def representation(value: str) -> str:
    """Come la regola del renderer scrive l'esatto: intero, decimale_esatto (3/2 -> 1.5), decimale_arrotondato
    (44/3 -> 14.67, non uguale all'esatto), radice (forma simbolica esatta) o non_analizzabile."""
    x = parse_exact(value)
    if x is None:
        return "non_analizzabile"
    if not x.rational:
        return "radice"
    if x.coef.denominator == 1:
        return "intero"
    shown = latex_value(value)
    return "decimale_esatto" if _RAT.fullmatch(shown) and Fraction(shown) == x.coef else "decimale_arrotondato"


def value_record(value: str, unit: str = "") -> dict:
    """Esatto, visualizzato previsto e risposta di lettura di una misura, con la provenienza dei primi due."""
    shown = latex_value(value)
    return {"esatto": value, "visualizzato_previsto": shown, "rappresentazione": representation(value),
            "risposta_lettura": shown + unit}


def measure_values(generation: str) -> dict[str, dict]:
    """{"side:AB" | "angle:BAD": record di value_record} dalle misure di diagram_generation. Le chiavi sono quelle di
    f1_mavis.mavis_truth (lato con le lettere ordinate, angolo col vertice al centro e gli estremi ordinati)."""
    gm = measures(generation)
    out = {f"side:{k.upper()}": value_record(v) for k, v in gm["side"].items()}
    out.update({f"angle:{angle_name(k[0].upper(), k[2:].upper())}": value_record(v, "°")
                for k, v in gm["angle"].items()})
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- stampato: valori come li legge il modello

@dataclass(frozen=True)
class Shown:
    """Una misura stampata come la legge il modello: un punto esatto (intero, decimale esatto, radice) o, per un
    decimale arrotondato, l'intervallo [d - 0.005, d + 0.005] dei valori che hanno quella scrittura."""

    text: str
    point: Exact | None
    lo: Fraction | None = None
    hi: Fraction | None = None

    @property
    def exact(self) -> bool:
        return self.point is not None


def shown_of(value: str) -> Shown | None:
    """La misura stampata dalla regola del renderer per l'esatto `value`; None se non è analizzabile."""
    x = parse_exact(value)
    if x is None:
        return None
    text = latex_value(value)
    if representation(value) == "decimale_arrotondato":
        d = Fraction(text)
        return Shown(text, None, max(d - HALF_ULP, Fraction(0)), d + HALF_ULP)
    return Shown(text, x)


def _inside(x: Exact, lo: Fraction, hi: Fraction) -> bool:
    return lo * lo <= x.square <= hi * hi  # tutti positivi: si confrontano i quadrati, razionali


def provably_different(a: Shown, b: Shown) -> bool:
    """Le due scritture escludono che i valori siano uguali (vale per ogni esatto compatibile con ciascuna)."""
    if a.exact and b.exact:
        return a.point.square != b.point.square
    if a.exact or b.exact:
        p, r = (a, b) if a.exact else (b, a)
        return not _inside(p.point, r.lo, r.hi)
    return a.hi < b.lo or b.hi < a.lo


def provably_equal(a: Shown, b: Shown) -> bool:
    """Uguaglianza dimostrata dalle scritture: solo fra punti esatti. Due decimali arrotondati uguali (14.67 e 14.67)
    non dimostrano nulla."""
    return a.exact and b.exact and a.point.square == b.point.square


def angle_degrees(s: Shown) -> Fraction | None:
    """Un angolo stampato come numero esatto di gradi, se lo è (razionale scritto in modo esatto)."""
    return s.point.coef if s.exact and s.point.rational else None


# ---------------------------------------------------------------- figure e misure per figura

ISOSCELES_APEX = (
    (re.compile(r"\btwo (?:congruent|equal-length|equal) sides ([a-z]{2}) and ([a-z]{2})\b"), "lati_uguali"),
    (re.compile(r"\bwhere ([a-z]{2}) = ([a-z]{2})\b"), "uguaglianza"),
    (re.compile(r"\b([a-z]{2}) as its base\b"), "base"),
    (re.compile(r"\bangle ([a-z]{3}) as its vertex angle\b"), "angolo_al_vertice"),
)


def isosceles_apex(sentence: str, name: str) -> tuple[str | None, str | None]:
    """(vertice, regola) dell'isoscele `name` dalla sua frase di diagram_generation, o (None, motivo)."""
    low, letters = sentence.lower(), set(name.lower())
    found = set()
    for pattern, rule in ISOSCELES_APEX:
        for m in pattern.finditer(low):
            if rule in ("lati_uguali", "uguaglianza"):
                a, b = m.group(1), m.group(2)
                common = set(a) & set(b)
                if len(common) == 1 and set(a) | set(b) <= letters:
                    found.add((common.pop().upper(), rule))
            elif rule == "base":
                rest = letters - set(m.group(1))
                if set(m.group(1)) <= letters and len(rest) == 1:
                    found.add((rest.pop().upper(), rule))
            elif set(m.group(1)) == letters:
                found.add((m.group(1)[1].upper(), rule))
    apexes = {a for a, _ in found}
    if len(apexes) == 1:
        return apexes.pop(), "+".join(sorted(r for _, r in found))
    return None, "vertice_non_trovato" if not found else "vertici_diversi"


def family(kind: str, name: str) -> str:
    if kind in QUAD_KINDS and len(name) == 4:
        return "quadrilateral"
    if kind in TRIANGLE_KINDS and len(name) == 3:
        return "triangle"
    if kind == "sector" and len(name) == 3:
        return "sector"
    return "other"


@dataclass
class ShapeFacts:
    """Una figura con domanda shape_type (senza lato sostituito): tipo dichiarato e misure sue, esatte e stampate.
    Lati per nome nell'ordine del poligono (AB, BC, ...); angoli interni per vertice."""

    name: str
    kind: str
    family: str
    sentence: str
    sides: dict[str, str] = field(default_factory=dict)  # lato -> esatto
    angles: dict[str, str] = field(default_factory=dict)  # vertice -> esatto (gradi)
    apex: str | None = None  # vertice degli isosceli, dalla frase di costruzione
    apex_rule: str | None = None
    inscribed_circle: bool = False


def _polygon_sides(name: str) -> list[str]:
    return [name[i] + name[(i + 1) % len(name)] for i in range(len(name))]


def shape_facts(truth: dict, generation: str) -> list[ShapeFacts]:
    """Le figure di shape_type della verità (f1_mavis.mavis_truth), nell'ordine di costruzione, con le loro misure."""
    figs, _ = figures(generation)
    sentence = {f.name.upper(): f.sentence for f in figs}
    vals = measure_values(generation)
    out = []
    for s in truth["shapes"]:
        if s.get("modification"):
            continue
        name, kind = s["name"], s["kind"]
        fam = family(kind, name)
        f = ShapeFacts(name, kind, fam, sentence.get(name, ""), inscribed_circle=bool(s.get("inscribed_circle")))
        if fam == "sector":  # centro al centro del nome (ABC: B), raggi AB e BC
            sides = [name[0] + name[1], name[1] + name[2]]
            corners = [(name[1], name[0] + name[2])]
        else:
            sides = _polygon_sides(name)
            corners = [(name[i], name[i - 1] + name[(i + 1) % len(name)]) for i in range(len(name))]
        for xy in sides:
            v = vals.get("side:" + "".join(sorted(xy)))
            if v:
                f.sides[xy] = v["esatto"]
        for vertex, ends in corners:
            v = vals.get("angle:" + angle_name(vertex, ends))
            if v:
                f.angles[vertex] = v["esatto"]
        if kind == "isosceles triangle":
            f.apex, f.apex_rule = isosceles_apex(f.sentence, name)
        out.append(f)
    return out


# ---------------------------------------------------------------- shape_type: insiemi di opzioni e verità esatta

# Un insieme fisso per famiglia, mescolato per istanza. Le classi escluse lo sono per i fatti della sorgente (sonda del
# job 30773, outputs/mavis_c9/sonda-1): un'opzione mai confutabile dalle misure stampate comparirebbe solo come risposta
# e un insieme scelto per istanza ne rivelerebbe la presenza.
OPTIONS = {"triangle": ("acute triangle", "right triangle", "obtuse triangle"),
           "quadrilateral": ("rectangle", "rhombus", "square")}
NOUN = {"triangle": "triangle", "quadrilateral": "quadrilateral"}
EXCLUDED = {
    "parallelogram": "vera per ogni quadrilatero di MAVIS (quadrati, rettangoli e parallelogrammi dichiarati): offerta, "
                     "sarebbe sempre la risposta",
    "kite": "con la definizione inclusiva vera per rombi e quadrati, con quella esclusiva falsa: ambigua; e nessuna "
            "misura stampata la confuta",
    "trapezoid": "vera o falsa per ogni parallelogramma secondo la definizione (inclusiva o esclusiva): ambigua",
    "isosceles triangle": "nei triangoli di MAVIS mai confutabile dalle misure stampate (un solo angolo stampato): "
                          "offerta, sarebbe sempre la risposta; equilatero ⊂ isoscele con la definizione inclusiva",
    "equilateral triangle": "equilatero ⊂ isoscele e ⊂ acutangolo: con le classi degli angoli non è una partizione",
    "scalene triangle": "mai dimostrabile dalle misure stampate nei triangoli di MAVIS",
    "sector": "vera per ogni figura circolare di MAVIS, mai confutabile; semicerchio e cerchio sempre falsi (angolo al "
              "centro stampato fra 30° e 120°): nessun insieme informativo",
}
RIGHT = Fraction(90)


def _exact_or_none(values: dict[str, str]) -> dict[str, Exact] | None:
    out = {k: parse_exact(v) for k, v in values.items()}
    return None if any(x is None for x in out.values()) else out


def _tri3(*xs):
    """AND a tre valori (True, False, None = ignoto)."""
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def exact_classes(f: ShapeFacts) -> tuple[dict[str, bool | None] | None, str | None]:
    """(verità delle opzioni della famiglia dalle annotazioni, contraddizione). Verità: True, False o None (ignota: mai
    trattata come falsa). Usa il tipo dichiarato e le misure esatte; mai i valori visualizzati."""
    sides, angles = _exact_or_none(f.sides), _exact_or_none(f.angles)
    if sides is None or angles is None:
        return None, "valori_non_analizzabili"
    if any(not a.rational or not 0 < a.coef < 180 for a in angles.values()):
        return None, "angolo_fuori_dominio"
    if f.family == "quadrilateral":
        order = _polygon_sides(f.name)
        got = [sides.get(s) for s in order]
        for i in (0, 1):  # parallelogrammi (anche rettangoli e quadrati): lati opposti uguali
            if got[i] and got[i + 2] and got[i].square != got[i + 2].square:
                return None, "lati_opposti_diversi"
        degs = [angles.get(v) for v in f.name]
        for i in range(4):
            a, b = degs[i], degs[(i + 1) % 4]
            if a and b and a.coef + b.coef != 180:
                return None, "angoli_consecutivi_non_supplementari"
        if f.kind in ("square", "rectangle"):
            if any(a.coef != RIGHT for a in angles.values()):
                return None, "angolo_non_retto_in_un_rettangolo"
            right = True
        else:
            right = (next(iter(angles.values())).coef == RIGHT) if angles else None
        adjacent = [(got[i], got[(i + 1) % 4]) for i in range(4) if got[i] and got[(i + 1) % 4]]
        if f.kind == "square":
            if len({x.square for x in sides.values()}) > 1:
                return None, "lati_diversi_in_un_quadrato"
            equal = True
        elif f.kind in ("rectangle", "parallelogram"):
            equal = (adjacent[0][0].square == adjacent[0][1].square) if adjacent else None
        else:
            return None, "tipo_non_gestito"
        return {"rectangle": right, "rhombus": equal, "square": _tri3(right, equal)}, None
    if f.family == "triangle":
        deg = {v: a.coef for v, a in angles.items()}
        if f.kind == "isosceles triangle":
            if f.apex is None:
                return None, "vertice_dell_isoscele_ignoto"
            legs = [sides[s] for s in sides if f.apex in s]
            if len(legs) == 2 and legs[0].square != legs[1].square:
                return None, "lati_uguali_diversi"
            if f.apex in deg:
                top = deg[f.apex]
            elif deg:  # angolo alla base
                top = 180 - 2 * next(iter(deg.values()))
                if top <= 0:
                    return None, "angolo_alla_base_troppo_grande"
            else:
                return {k: None for k in OPTIONS["triangle"]}, None
            if any(d != top and d != (180 - top) / 2 for d in deg.values()):
                return None, "angoli_incoerenti"
            return {"acute triangle": top < RIGHT, "right triangle": top == RIGHT, "obtuse triangle": top > RIGHT}, None
        if f.kind == "right triangle":  # un angolo retto; gli altri due acuti e complementari
            d = list(deg.values())
            if any(x > RIGHT for x in d) or d.count(RIGHT) > 1 \
                    or (len(d) == 2 and RIGHT not in d and sum(d) != RIGHT) \
                    or (len(d) == 3 and (sum(d) != 180 or RIGHT not in d)):
                return None, "angoli_incoerenti"
            return {"acute triangle": False, "right triangle": True, "obtuse triangle": False}, None
        if f.kind == "equilateral triangle":
            if any(d != 60 for d in deg.values()) or len({x.square for x in sides.values()}) > 1:
                return None, "angoli_o_lati_incoerenti"
            return {"acute triangle": True, "right triangle": False, "obtuse triangle": False}, None
        if len(deg) >= 2:  # triangolo generico: servono due angoli
            third = 180 - sum(deg.values())
            if third <= 0:
                return None, "angoli_incoerenti"
            top = max([*deg.values(), third])
            return {"acute triangle": top < RIGHT, "right triangle": top == RIGHT, "obtuse triangle": top > RIGHT}, None
        return {k: None for k in OPTIONS["triangle"]}, None
    return None, "famiglia_non_gestita"


# ---------------------------------------------------------------- stampato: opzioni confutate dalle misure scritte

def _printed(f: ShapeFacts) -> tuple[dict[str, Shown], dict[str, Shown]] | None:
    sides = {k: shown_of(v) for k, v in f.sides.items()}
    angles = {k: shown_of(v) for k, v in f.angles.items()}
    if any(s is None for s in [*sides.values(), *angles.values()]):
        return None
    return sides, angles


def _known_triangle_angles(name: str, sides: dict[str, Shown], angles: dict[str, Shown]) -> dict[str, Fraction]:
    """Angoli noti dalle sole misure stampate: quelli scritti in modo esatto; due lati scritti uguali in modo esatto
    danno angoli opposti uguali (con il terzo angolo noto, entrambi (180 - terzo) / 2; tre lati uguali, 60° ciascuno);
    due angoli noti danno il terzo. Un decimale arrotondato non dimostra uguaglianze."""
    known = {v: d for v, s in angles.items() if (d := angle_degrees(s)) is not None}
    pairs = []
    items = list(sides.items())
    for i, (s1, a) in enumerate(items):
        for s2, b in items[i + 1:]:
            if provably_equal(a, b):
                pairs.append(((set(name) - set(s1)).pop(), (set(name) - set(s2)).pop()))
    if len({v for p in pairs for v in p}) == 3 and len(pairs) >= 2:
        return {v: Fraction(60) for v in name} if all(known.get(v, 60) == 60 for v in name) else known
    for _ in range(3):
        for o1, o2 in pairs:
            if o1 in known and o2 not in known:
                known[o2] = known[o1]
            elif o2 in known and o1 not in known:
                known[o1] = known[o2]
            elif o1 not in known and o2 not in known:
                (third,) = set(name) - {o1, o2}
                if third in known:
                    known[o1] = known[o2] = (180 - known[third]) / 2
        if len(known) == 2:
            (missing,) = set(name) - set(known)
            known[missing] = 180 - sum(known.values())
    return known


def printed_refutations(f: ShapeFacts, right_mark: bool = False) -> dict[str, str] | None:
    """{opzione: motivo} delle opzioni della famiglia confutate dalle misure stampate (rappresentazione prevista dalla
    regola del renderer), senza tipo dichiarato né valori esatti nascosti. `right_mark` aggiunge il segno dell'angolo
    retto di un triangolo rettangolo dichiarato: convenzione osservata, da verificare a vista (non usato per le domande
    emesse). None se una misura non è analizzabile."""
    got = _printed(f)
    if got is None:
        return None
    sides, angles = got
    out = {}
    if f.family == "quadrilateral":
        for v, s in angles.items():
            if (angle_degrees(s) or 0) != RIGHT and (s.exact or not s.lo <= RIGHT <= s.hi):
                out["rectangle"] = f"angolo {v} stampato {s.text}°, non retto"
                break
        items = list(sides.items())
        for i, (s1, a) in enumerate(items):
            hit = next(((s2, b) for s2, b in items[i + 1:] if provably_different(a, b)), None)
            if hit:
                out["rhombus"] = f"lati {s1} = {a.text} e {hit[0]} = {hit[1].text} stampati diversi"
                break
        if "rectangle" in out or "rhombus" in out:
            out["square"] = out.get("rectangle") or out["rhombus"]
        return out
    if f.family == "triangle":
        known = _known_triangle_angles(f.name, sides, angles)
        vals = list(known.values()) + ([RIGHT] if right_mark else [])
        full = len(known) == 3
        if any(d >= RIGHT for d in vals):
            out["acute triangle"] = f"angolo di {max(vals)}° noto dalle misure stampate"
        if any(d > RIGHT for d in vals):
            out["right triangle"] = f"angolo ottuso di {max(vals)}° noto"
        elif full and not right_mark and RIGHT not in vals:
            out["right triangle"] = f"angoli noti {sorted(known.values())}, nessuno retto"
        if any(d == RIGHT for d in vals):
            out["obtuse triangle"] = "un angolo retto noto: gli altri due sommano 90°"
        elif full and max(vals) < RIGHT:
            out["obtuse triangle"] = f"angoli noti {sorted(known.values())}, nessuno ottuso"
        return out
    return {}


# ---------------------------------------------------------------- shape_type: domanda o omissione

def key_shuffle(key: str, name: str, options: tuple[str, ...]) -> list[str]:
    from .f1c import key_rng  # stessa famiglia di scelte deterministiche per chiave (f1c.key_rng)

    out = list(options)
    key_rng(f"{key}/mavis-c9/{name}").shuffle(out)
    return out


def question_text(family_: str, name: str, options: list[str]) -> str:
    return f"Which of these describes {NOUN[family_]} {name}: {', '.join(options[:-1])} or {options[-1]}?"


def shape_outcome(key: str, f: ShapeFacts) -> dict:
    """Esito di shape_type per una figura: domanda con le opzioni della famiglia (una sola vera secondo le annotazioni,
    le altre confutate dalle misure stampate) oppure omissione col motivo."""
    base = {"figura": f.name, "tipo_dichiarato": f.kind}
    if f.family == "sector":
        return {**base, "esito": "omessa", "motivo": "settore_nessun_insieme_informativo"}
    if f.family not in OPTIONS:
        return {**base, "esito": "omessa", "motivo": "famiglia_non_gestita"}
    truth, contradiction = exact_classes(f)
    if truth is None:
        return {**base, "esito": "omessa", "motivo": "misure_contraddittorie" if contradiction not in (
            "valori_non_analizzabili", "famiglia_non_gestita", "tipo_non_gestito", "vertice_dell_isoscele_ignoto")
            else contradiction, "dettaglio": contradiction}
    options = OPTIONS[f.family]
    if any(v is None for v in truth.values()):
        return {**base, "esito": "omessa", "motivo": "verita_ignota",
                "ignote": sorted(k for k, v in truth.items() if v is None)}
    true = [k for k in options if truth[k]]
    if len(true) != 1:
        return {**base, "esito": "omessa", "motivo": "nessuna_opzione_vera" if not true else "piu_opzioni_vere",
                "vere": true}
    answer = true[0]
    refuted = printed_refutations(f)
    if refuted is None:
        return {**base, "esito": "omessa", "motivo": "valori_non_analizzabili"}
    wrong = [k for k in options if k != answer]
    if any(k in refuted and truth[k] for k in options):
        raise ValueError(f"{key} {f.name}: opzione vera confutata dalle misure stampate ({refuted})")
    open_ = [k for k in wrong if k not in refuted]
    if open_:
        out = {**base, "esito": "omessa", "motivo": "non_deciso_dalle_misure_stampate", "non_confutate": open_}
        if f.kind == "right triangle":
            with_mark = printed_refutations(f, right_mark=True)
            if all(k in with_mark for k in wrong):
                out["motivo"] = "richiede_segno_angolo_retto"
        return out
    shuffled = key_shuffle(key, f.name, options)
    return {**base, "esito": "recuperabile", "opzioni": shuffled, "risposta": answer,
            "confutazioni": {k: refuted[k] for k in wrong},
            "domanda": {"type": "shape_type", "q": question_text(f.family, f.name, shuffled), "answer": answer}}


# ---------------------------------------------------------------- pool delle domande della costruzione C9

_SIDE_Q = re.compile(r"^What is the length of ([A-Z]{2})\?$")
_ANGLE_Q = re.compile(r"^What is the measure of angle ([A-Z]{3})\?$")


def questions(key: str, truth: dict, generation: str) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """(pool, esiti di shape_type, valori) della costruzione mavis-c9/1. Il pool è quello storico di
    f1_mavis.mavis_questions con le domande shape_type sostituite dalla domanda con opzioni o tolte (omesse); le altre
    domande sono identiche, e quelle di lettura dicono la misura che chiedono (`elemento`, fuori dal digest)."""
    from .f1_mavis import mavis_questions

    pool = mavis_questions(key, truth)
    values = measure_values(generation)
    outcomes = [shape_outcome(key, f) for f in shape_facts(truth, generation)]
    it, out = iter(outcomes), []
    for q in pool:
        if q["type"] == "shape_type":
            o = next(it)
            if q["q"] != f"What type of shape is {o['figura']}?":
                raise ValueError(f"{key}: domanda shape_type storica {q['q']!r} non allineata alla figura {o['figura']}")
            if o["esito"] == "recuperabile":
                out.append(o["domanda"])
            continue
        m = _SIDE_Q.fullmatch(q["q"]) if q["type"] == "side_len" else _ANGLE_Q.fullmatch(q["q"]) \
            if q["type"] == "angle_val" else None
        if m:
            elem = ("side:" if q["type"] == "side_len" else "angle:") + m.group(1)
            if values.get(elem, {}).get("risposta_lettura") != q["answer"]:
                raise ValueError(f"{key}: {elem}: risposta storica {q['answer']!r} diversa dal visualizzato previsto")
            q = {**q, "elemento": elem}
        elif q["type"] in ("side_len", "angle_val"):
            raise ValueError(f"{key}: domanda di lettura non riconosciuta: {q['q']!r}")
        out.append(q)
    if next(it, None) is not None:
        raise ValueError(f"{key}: figure di shape_type senza domanda storica")
    return out, outcomes, values


# ---------------------------------------------------------------- controllo indipendente di una domanda shape_type
#
# Scritto a parte dal generatore: la verità esatta viene da un modello numerico della figura (coordinate costruite dal
# tipo dichiarato e dalle misure esatte; una lunghezza non data si prova con due valori diversi, e una classe che cambia
# fra i due è ignota), le confutazioni da un problema di ammissibilità sugli angoli e da intervalli sui lati stampati.

TOL = 1e-9


def _num(x: Exact) -> float:
    return float(x.coef) * x.rad ** 0.5


def _angles_of(pts: list[tuple[float, float]]) -> list[float]:
    import math

    out = []
    n = len(pts)
    for i in range(n):
        (ax, ay), (bx, by), (cx, cy) = pts[i - 1], pts[i], pts[(i + 1) % n]
        u, v = (ax - bx, ay - by), (cx - bx, cy - by)
        cos = (u[0] * v[0] + u[1] * v[1]) / (math.hypot(*u) * math.hypot(*v))
        out.append(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
    return out


def _lengths_of(pts: list[tuple[float, float]]) -> list[float]:
    import math

    return [math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def _quad_classes(pts) -> dict[str, bool]:
    ang, lens = _angles_of(pts), _lengths_of(pts)
    rect = all(_close(a, 90.0) for a in ang)
    rhom = all(_close(x, lens[0]) for x in lens)
    return {"rectangle": rect, "rhombus": rhom, "square": rect and rhom}


def _tri_classes(pts) -> dict[str, bool]:
    top = max(_angles_of(pts))
    return {"acute triangle": top < 90 - 1e-7, "right triangle": _close(top, 90.0), "obtuse triangle": top > 90 + 1e-7}


def numeric_classes(f: ShapeFacts) -> dict[str, bool | None] | None:
    """Verità delle opzioni da un modello numerico della figura dichiarata con le misure esatte; None se il tipo non è
    modellato o le misure non si leggono."""
    import math

    sides, angles = _exact_or_none(f.sides), _exact_or_none(f.angles)
    if sides is None or angles is None:
        return None
    witnesses = []
    if f.family == "quadrilateral" and f.kind in ("square", "rectangle", "parallelogram"):
        p = f.name
        by_side = {s: _num(x) for s, x in sides.items()}
        order = _polygon_sides(p)
        l01 = by_side.get(order[0]) or by_side.get(order[2])
        l12 = by_side.get(order[1]) or by_side.get(order[3])
        if f.kind == "square":
            l01 = l12 = next(iter(by_side.values()), 1.0)
        if f.kind in ("square", "rectangle"):
            phi = 90.0
        elif angles:
            v, a = next(iter(angles.items()))
            phi = float(a.coef) if p.index(v) in (1, 3) else 180.0 - float(a.coef)
        else:
            return {k: None for k in OPTIONS["quadrilateral"]}
        if l01 and l12:
            pairs = [(l01, l12)]
        elif l01:  # lato adiacente ignoto: due testimoni, uguale e diverso
            pairs = [(l01, l01), (l01, 2 * l01 + 1)]
        elif l12:
            pairs = [(l12, l12), (2 * l12 + 1, l12)]
        else:
            pairs = [(1.0, 1.0), (3.0, 1.0)]
        for a, b in pairs:
            t = math.radians(phi)
            pts = [(a, 0.0), (0.0, 0.0), (b * math.cos(t), b * math.sin(t))]
            pts.append((pts[0][0] + pts[2][0], pts[0][1] + pts[2][1]))
            witnesses.append(_quad_classes(pts))
    elif f.family == "triangle":
        deg = {v: float(a.coef) for v, a in angles.items()}
        tops = []
        if f.kind == "isosceles triangle" and f.apex:
            if f.apex in deg:
                tops = [deg[f.apex]]
            elif deg:
                tops = [180.0 - 2 * next(iter(deg.values()))]
            else:
                tops = [30.0, 120.0]
            for top in tops:
                t = math.radians(top)
                witnesses.append(_tri_classes([(0.0, 0.0), (1.0, 0.0), (math.cos(t), math.sin(t))]))
        elif f.kind == "right triangle":
            acute = [d for d in deg.values() if d < 90] or [30.0, 60.0]
            for a in acute:
                witnesses.append(_tri_classes([(0.0, 0.0), (1.0, 0.0), (0.0, math.tan(math.radians(a)))]))
        elif f.kind == "equilateral triangle":
            witnesses.append(_tri_classes([(0.0, 0.0), (1.0, 0.0), (0.5, 3 ** 0.5 / 2)]))
        elif len(deg) >= 2:
            (a, b) = list(deg.values())[:2]
            ta, tb = math.radians(a), math.radians(b)
            # A=(0,0), B=(1,0); C dall'intersezione dei raggi con gli angoli a in A e b in B
            k = math.sin(tb) / math.sin(ta + tb)
            witnesses.append(_tri_classes([(0.0, 0.0), (1.0, 0.0), (k * math.cos(ta), k * math.sin(ta))]))
        else:
            return {k: None for k in OPTIONS["triangle"]}
    else:
        return None
    return {k: (witnesses[0][k] if all(w[k] == witnesses[0][k] for w in witnesses) else None) for k in witnesses[0]}


def _interval_sq(s: Shown) -> tuple[Fraction, Fraction]:
    return (s.point.square, s.point.square) if s.exact else (s.lo * s.lo, s.hi * s.hi)


def _angle_feasible(name: str, fixed: dict[str, Fraction], equal: list[tuple[str, str]]) -> dict[str, bool]:
    """Classi degli angoli ammissibili: esiste una terna di angoli positivi con somma 180, uguale a `fixed` dove è dato e
    con gli angoli di `equal` uguali fra loro, che sia acutangola, rettangola o ottusangola?"""
    parent = {v: v for v in name}

    def find(v):
        while parent[v] != v:
            v = parent[v]
        return v

    for a, b in equal:
        parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for v in name:
        groups.setdefault(find(v), []).append(v)
    values, free = [], []
    for members in groups.values():
        given = {fixed[v] for v in members if v in fixed}
        if len(given) > 1:
            return {"acute triangle": False, "right triangle": False, "obtuse triangle": False}
        if given:
            values += [given.pop()] * len(members)
        else:
            free.append(len(members))
    rest = 180 - sum(values)

    def fits(r, sizes) -> bool:  # gruppi liberi positivi con somma r
        return r == 0 if not sizes else r > 0

    ok = fits(rest, free) and all(x > 0 for x in values)
    right = ok and (RIGHT in values or any(fits(rest - s * RIGHT, free[:i] + free[i + 1:])
                                           for i, s in enumerate(free)))
    # un gruppo libero arriva fino a rest / s (esattamente, se è l'unico; per difetto, se gli altri restano positivi)
    obtuse = ok and (any(x > RIGHT for x in values) or any(rest / s > RIGHT for s in free))
    if not ok or any(x >= RIGHT for x in values):
        acute = False
    elif len(free) == 0:
        acute = True
    elif len(free) == 1:
        acute = 0 < rest / free[0] < RIGHT
    elif len(free) == 2:
        s1, s2 = free
        lo, hi = max(Fraction(0), (rest - RIGHT * s2) / s1), min(rest / s1, RIGHT)
        acute = lo < hi
    else:
        acute = True
    return {"acute triangle": acute, "right triangle": right, "obtuse triangle": obtuse}


def printed_consistent(f: ShapeFacts) -> dict[str, bool] | None:
    """Per ogni opzione della famiglia: esiste una figura di quella classe compatibile con le misure stampate?"""
    got = _printed(f)
    if got is None:
        return None
    sides, angles = got
    if f.family == "quadrilateral":
        order = _polygon_sides(f.name)
        rect = all(s.lo <= RIGHT <= s.hi if not s.exact else (s.point.rational and s.point.coef == RIGHT)
                   for s in angles.values())
        for i in (0, 1):
            a, b = sides.get(order[i]), sides.get(order[i + 2])
            if a and b:
                (al, ah), (bl, bh) = _interval_sq(a), _interval_sq(b)
                rect = rect and max(al, bl) <= min(ah, bh)
        spans = [_interval_sq(s) for s in sides.values()]
        rhom = not spans or max(lo for lo, _ in spans) <= min(hi for _, hi in spans)
        return {"rectangle": rect, "rhombus": rhom, "square": rect and rhom}
    if f.family == "triangle":
        fixed = {v: s.point.coef for v, s in angles.items() if s.exact and s.point.rational}
        equal, items = [], list(sides.items())
        for i, (s1, a) in enumerate(items):
            for s2, b in items[i + 1:]:
                if a.exact and b.exact and _interval_sq(a) == _interval_sq(b):
                    equal.append(((set(f.name) - set(s1)).pop(), (set(f.name) - set(s2)).pop()))
        return _angle_feasible(f.name, fixed, equal)
    return None


def check_question(key: str, f: ShapeFacts, options: list[str], answer: str, text: str) -> list[str]:
    """Problemi di una domanda shape_type (lista vuota se valida): opzioni della famiglia distinte e nell'ordine
    deterministico, testo coerente, risposta vera e unica vera nel modello numerico, risposta compatibile con le misure
    stampate e ogni altra opzione incompatibile."""
    problems = []
    vocab = OPTIONS.get(f.family)
    if vocab is None:
        return [f"famiglia {f.family} senza insieme di opzioni"]
    if len(set(options)) != len(options) or len(options) < 2:
        problems.append(f"opzioni ripetute o meno di due: {options}")
    if set(options) - set(vocab):
        problems.append(f"opzioni fuori dall'insieme della famiglia: {sorted(set(options) - set(vocab))}")
    if options.count(answer) != 1:
        problems.append(f"risposta {answer!r} presente {options.count(answer)} volte fra le opzioni")
    if text != question_text(f.family, f.name, options):
        problems.append("testo della domanda diverso da quello costruito dalle opzioni")
    if sorted(options) == sorted(vocab) and options != key_shuffle(key, f.name, vocab):
        problems.append("ordine delle opzioni diverso da quello deterministico per chiave")
    truth = numeric_classes(f)
    seen = printed_consistent(f)
    if truth is None or seen is None:
        return problems + ["modello numerico o misure stampate non disponibili"]
    for opt in options:
        t, c = truth.get(opt), seen.get(opt)
        if opt == answer:
            if t is not True:
                problems.append(f"risposta {opt!r} non vera nel modello numerico ({t})")
            if c is not True:
                problems.append(f"risposta {opt!r} incompatibile con le misure stampate")
        else:
            if t is None:
                problems.append(f"opzione {opt!r} con verità ignota: non è dimostrata falsa")
            elif t:
                problems.append(f"opzione {opt!r} vera: più di una opzione vera")
            if c:
                problems.append(f"opzione {opt!r} compatibile con le misure stampate: non confutata dall'immagine")
    return problems


def check_outcome(key: str, f: ShapeFacts, outcome: dict) -> list[str]:
    """check_question su un esito recuperabile di shape_outcome, con la coerenza fra esito e domanda."""
    if outcome.get("esito") != "recuperabile":
        return []
    q = outcome["domanda"]
    problems = check_question(key, f, outcome["opzioni"], outcome["risposta"], q["q"])
    if q["answer"] != outcome["risposta"] or q["type"] != "shape_type" or outcome["figura"] != f.name:
        problems.append("domanda e esito non coerenti")
    if sorted(outcome["confutazioni"]) != sorted(o for o in outcome["opzioni"] if o != outcome["risposta"]):
        problems.append("confutazioni diverse dalle opzioni sbagliate")
    return problems
