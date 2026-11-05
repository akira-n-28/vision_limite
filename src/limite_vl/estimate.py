"""Stime della qualità da un campione, con limiti esatti (C8; docs/specs/S4_CAMPIONAMENTO_BUDGET.md §§3, 8, 9).

Nucleo tecnico di S4 adottato dall'utente il 2026-11-05: limiti esatti, compatibilità fra disegno e quantità stimata,
verifiche di copertura. Soglie di ammissione, numerosità e budget degli audit restano decisioni aperte (D4, D5) e qui
non compaiono. Il modulo non estrae campioni, non unisce strati, non legge i fogli di revisione e non certifica: nessun
record diventa gold per effetto di un calcolo. Solo libreria standard.

**Limiti** (superiori unilaterali al livello 1 - alpha; alpha = 1/20 se non indicato, tenuto come frazione esatta):
- `hypergeometric_upper(N, n, k)`: n unità estratte senza reinserimento con probabilità uguali fra N, k non approvate.
  D_U è il più grande D intero in [k, N - n + k] con P(X <= k | N, D, n) > alpha, calcolata con interi esatti e
  ricerca monotona (P non cresce con D); il limite della quota è D_U / N. Con k = n vale D_U = N, senza ricerca; con
  n = N (censimento) D_U = k, cioè la quota esatta.
- `clopper_pearson_upper(n, k)`: n estrazioni indipendenti con la stessa probabilità di errore (con reinserimento, per
  esempio QA estratte con probabilità proporzionale ai target-token). È il più piccolo p con P(X <= k | n, p) <= alpha,
  cercato per bisezione sulla griglia p = m / 2**64 con la probabilità binomiale esatta e arrotondato per eccesso al
  float: mai sotto il valore vero. Con k = n vale 1; con k = 0 vale 1 - alpha^(1/n).
- `bonferroni(alpha, m)`: m affermazioni simultanee, ciascuna al livello alpha / m. Le affermazioni restano separate;
  un limite per l'unione di più strati non si calcola qui.
Input non validi (N, n, k non interi o bool, n < 1, n > N senza reinserimento, k < 0, k > n, alpha fuori da (0, 1))
danno ValueError, mai un valore corretto in silenzio.

**Affermazione e disegno.** Una stima riguarda una quantità di `QUANTITIES`, una popolazione e uno strato dichiarati
prima dell'estrazione (`Claim`), con il disegno registrato all'estrazione (`Design`: unità estratta, modo
d'estrazione, N, n). `check_compatible` rifiuta, con `Refused` (sottoclasse di ValueError):
- i casi mirati (estrazione "mirata"): scelti per regola, non stimano un tasso generale; restano i conteggi (`describe`);
- un'unità estratta diversa da quella della quantità, per esempio la quota di QA da un campione di gruppi o di coppie
  (S4 §4: dipende dalle dimensioni dei gruppi non estratti);
- un modo d'estrazione diverso da quello della quantità (quota dei target-token solo con reinserimento e probabilità
  proporzionale ai token; quote di unità solo senza reinserimento con probabilità uguali);
- popolazione, strato o N diversi fra affermazione e disegno.

**Esiti** (`OUTCOMES`), uno per estrazione:
- `approvato`: giudicato corretto per la proprietà, con primo giudizio indipendente (S4 §5);
- `approvato_non_indipendente`: giudicato corretto, ma il primo giudizio non era indipendente dal target;
- `errato`: errore confermato;
- `incerto`, `non_giudicabile`: il revisore non decide, oppure il caso non si giudica;
- `mancante`: unità estratta e non giudicata.
Nel limite conta come errore tutto ciò che non è `approvato` (caso peggiore): errati, incerti, non giudicabili,
mancanti e approvazioni non indipendenti. Nessuno di questi esiti si toglie o vale come approvazione, e ciascuno si
riporta nei conteggi; la quota non risolta (incerti, non giudicabili, mancanti) si riporta sempre. Contarli come
errori lascia valido il limite, che diventa solo più largo.

**Il calcolo si rifiuta** (`Refused`, oltre ai casi di `check_compatible`) se gli esiti non corrispondono al disegno:
- numero di esiti diverso da n, perché la numerosità non si estende né si riduce dopo l'estrazione;
- con `tally`: unità ripetute in un'estrazione senza reinserimento, unità estratte senza esito (un'unità non giudicata
  si segna `mancante`), esiti di unità non estratte, esiti ripetuti o sconosciuti;
- con `estimate_simultaneous`: la stessa affermazione ripetuta nella famiglia.

**Risultato** di `estimate`, in tre parti separate:
1. `conteggi`: descrittivi, valgono per le unità guardate e non per la popolazione;
2. `limite`: valido per il disegno dichiarato. Con probabilità almeno 1 - alpha sull'estrazione, la quota della
   popolazione che la revisione non approverebbe non supera il limite;
3. `conclusione`: condizionata all'attendibilità del revisore. Lo stesso numero limita la quota di unità errate solo se
   il revisore non approva unità errate; la sua sensibilità non è misurata (S4 §4) e non si applica alcuna correzione.
`certificazione` vale sempre "nessuna".
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from numbers import Integral

ALPHA = Fraction(1, 20)
CP_BITS = 64  # risoluzione della bisezione di Clopper-Pearson: p = m / 2**64

UNIFORM = "uniforme_senza_reinserimento"
PPS = "proporzionale_ai_target_token_con_reinserimento"
TARGETED = "mirata"
EXTRACTIONS = (UNIFORM, PPS, TARGETED)
UNITS = ("qa", "gruppo", "coppia")
# quantità -> (unità estratta, estrazione, metodo del limite); un gruppo o una coppia estratti si controllano per intero
QUANTITIES = {
    "quota_qa_errate": ("qa", UNIFORM, "ipergeometrico"),
    "quota_gruppi_con_errore": ("gruppo", UNIFORM, "ipergeometrico"),
    "quota_coppie_con_errore": ("coppia", UNIFORM, "ipergeometrico"),
    "quota_target_token_errati": ("qa", PPS, "clopper_pearson"),
}
UNIT_NAMES = {"qa": "QA", "gruppo": "gruppi", "coppia": "coppie"}
OUTCOMES = ("approvato", "approvato_non_indipendente", "errato", "incerto", "non_giudicabile", "mancante")


class Refused(ValueError):
    """Stima rifiutata: il disegno non stima la quantità chiesta, oppure gli esiti non corrispondono al disegno."""


# ---------------------------------------------------------------- controlli degli input

def _int(name: str, x, low: int | None = None) -> int:
    if isinstance(x, bool) or not isinstance(x, Integral):
        raise ValueError(f"{name} deve essere un intero, non {x!r}")
    x = int(x)
    if low is not None and x < low:
        raise ValueError(f"{name} = {x}: serve almeno {low}")
    return x


def _alpha(alpha) -> Fraction:
    """alpha come frazione esatta: un float vale il decimale con cui si scrive (0.05 -> 1/20)."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, str, Fraction)):
        raise ValueError(f"alpha non valido: {alpha!r}")
    try:
        a = Fraction(repr(alpha)) if isinstance(alpha, float) else Fraction(alpha)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"alpha non valido: {alpha!r}") from None
    if not 0 < a < 1:
        raise ValueError(f"alpha = {alpha}: deve stare in (0, 1)")
    return a


def _sample(N, n, k) -> tuple[int, int, int]:
    N, n, k = _int("N", N, 1), _int("n", n, 1), _int("k", k, 0)
    if n > N:
        raise ValueError(f"n = {n} > N = {N}: senza reinserimento non si estraggono più unità della popolazione")
    if k > n:
        raise ValueError(f"k = {k} > n = {n}")
    return N, n, k


# ---------------------------------------------------------------- limiti esatti

def _hyper_cdf_numerator(N: int, D: int, n: int, k: int) -> int:
    """Somma su x <= k di C(D, x) C(N - D, n - x): numeratore di P(X <= k) con denominatore C(N, n)."""
    lo, hi = max(0, n - (N - D)), min(k, D, n)
    if hi < lo:
        return 0
    term = math.comb(D, lo) * math.comb(N - D, n - lo)
    total = term
    for x in range(lo, hi):  # rapporto esatto fra termini consecutivi, divisione intera senza resto
        term = term * (D - x) * (n - x) // ((x + 1) * (N - D - n + x + 1))
        total += term
    return total


def hypergeometric_cdf(N, D, n, k) -> Fraction:
    """P(X <= k) esatta per X ipergeometrica: n estratte senza reinserimento da N unità, di cui D errate."""
    N, n = _int("N", N, 1), _int("n", n, 1)
    D, k = _int("D", D, 0), _int("k", k)
    if D > N or n > N:
        raise ValueError(f"serve D <= N e n <= N (N = {N}, D = {D}, n = {n})")
    return Fraction(_hyper_cdf_numerator(N, D, n, k), math.comb(N, n))


def hypergeometric_upper(N, n, k, alpha=ALPHA) -> int:
    """D_U: il più grande numero di unità non approvate nella popolazione che il campione non esclude (S4 §3)."""
    N, n, k = _sample(N, n, k)
    a = _alpha(alpha)
    if k == n:
        return N  # P(X <= n) = 1 per ogni D: nessun D è escluso
    if n == N:
        return k  # censimento: X = D
    total = math.comb(N, n)

    def not_excluded(D: int) -> bool:  # P(X <= k | D) > alpha, con interi
        return _hyper_cdf_numerator(N, D, n, k) * a.denominator > a.numerator * total

    lo, hi = k, N - n + k  # vale sempre not_excluded(k): con D = k, P(X <= k) = 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if not_excluded(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _binomial_cdf_exceeds(n: int, k: int, m: int, bits: int, a: Fraction) -> bool:
    """P(X <= k | n, p = m / 2**bits) > a, con interi esatti."""
    q = (1 << bits) - m
    acc, power, c = 0, 1, 1  # c = C(n, x), power = m**x
    for x in range(k + 1):
        acc = acc * q + c * power  # somma di C(n, i) m**i q**(x - i) per i <= x
        power *= m
        c = c * (n - x) // (x + 1)
    return (acc * q ** (n - k)) * a.denominator > (a.numerator << (bits * n))


def clopper_pearson_upper(n, k, alpha=ALPHA) -> float:
    """Limite superiore unilaterale esatto di Clopper-Pearson per k errori in n prove indipendenti (con reinserimento)."""
    n, k = _int("n", n, 1), _int("k", k, 0)
    if k > n:
        raise ValueError(f"k = {k} > n = {n}")
    a = _alpha(alpha)
    if k == n:
        return 1.0
    lo, hi = 0, 1 << CP_BITS  # P(X <= k | p = 0) = 1 > alpha; P(X <= k | p = 1) = 0 <= alpha
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _binomial_cdf_exceeds(n, k, mid, CP_BITS, a):
            lo = mid
        else:
            hi = mid
    upper = Fraction(hi, 1 << CP_BITS)
    value = float(upper)
    if Fraction(value) < upper:
        value = math.nextafter(value, 1.0)  # per eccesso: il limite non scende sotto il valore esatto
    return value


def bonferroni(alpha, m) -> Fraction:
    """Livello di ciascuna di m affermazioni simultanee con livello di famiglia alpha."""
    return _alpha(alpha) / _int("m", m, 1)


# ---------------------------------------------------------------- affermazione, disegno, esiti

def _text(owner: str, name: str, value) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner}: {name} deve essere un testo non vuoto")


@dataclass(frozen=True)
class Design:
    """Disegno di uno strato registrato all'estrazione: unità estratta, modo, unità della popolazione (N) ed estrazioni
    (n), fissate prima di guardare. Con l'estrazione "mirata" N può mancare."""
    popolazione: str
    strato: str
    unita: str
    estrazione: str
    N: int | None
    n: int

    def __post_init__(self):
        _text("disegno", "popolazione", self.popolazione)
        _text("disegno", "strato", self.strato)
        if self.unita not in UNITS:
            raise ValueError(f"disegno: unità {self.unita!r} sconosciuta; note: {UNITS}")
        if self.estrazione not in EXTRACTIONS:
            raise ValueError(f"disegno: estrazione {self.estrazione!r} sconosciuta; note: {EXTRACTIONS}")
        n = _int("n", self.n, 1)
        if self.estrazione == TARGETED and self.N is None:
            return
        N = _int("N", self.N, 1)
        if self.estrazione == UNIFORM and n > N:
            raise ValueError(f"disegno: n = {n} > N = {N} senza reinserimento")


@dataclass(frozen=True)
class Claim:
    """Affermazione dichiarata prima dell'estrazione: proprietà giudicata, quantità (QUANTITIES), popolazione, strato e
    sua dimensione N nell'unità della quantità."""
    proprieta: str
    quantita: str
    popolazione: str
    strato: str
    N: int

    def __post_init__(self):
        for name in ("proprieta", "quantita", "popolazione", "strato"):
            _text("affermazione", name, getattr(self, name))
        _int("N", self.N, 1)


@dataclass(frozen=True)
class Counts:
    """Esiti delle estrazioni di uno strato, uno per estrazione (OUTCOMES)."""
    approvato: int = 0
    approvato_non_indipendente: int = 0
    errato: int = 0
    incerto: int = 0
    non_giudicabile: int = 0
    mancante: int = 0

    def __post_init__(self):
        for f in fields(self):
            _int(f.name, getattr(self, f.name), 0)

    @property
    def n(self) -> int:
        return sum(getattr(self, name) for name in OUTCOMES)

    @property
    def non_approvati(self) -> int:
        return self.n - self.approvato

    @property
    def non_risolti(self) -> int:
        return self.incerto + self.non_giudicabile + self.mancante


def check_compatible(claim: Claim, design: Design) -> str:
    """Metodo del limite per l'affermazione col disegno dato; Refused se il disegno non stima quella quantità."""
    if design.estrazione == TARGETED:
        raise Refused("casi mirati: scelti per regola, non stimano un tasso generale; si riportano solo i conteggi "
                      "(describe)")
    if claim.quantita not in QUANTITIES:
        raise Refused(f"quantità sconosciuta {claim.quantita!r}; note: {sorted(QUANTITIES)}")
    unit, extraction, method = QUANTITIES[claim.quantita]
    if design.unita != unit:
        why = (" (S4 §4: la quota di QA da un campione di gruppi dipende dalle dimensioni dei gruppi non estratti, e nel "
               "controesempio il limite copre nel 30% dei casi)") if unit == "qa" else ""
        raise Refused(f"{claim.quantita} chiede un campione di unità {unit!r}, il disegno estrae {design.unita!r}{why}")
    if design.estrazione != extraction:
        raise Refused(f"{claim.quantita} chiede l'estrazione {extraction!r}, il disegno è {design.estrazione!r}")
    if (claim.popolazione, claim.strato) != (design.popolazione, design.strato):
        raise Refused(f"popolazione o strato diversi: affermazione {claim.popolazione}/{claim.strato}, disegno "
                      f"{design.popolazione}/{design.strato}")
    if claim.N != design.N:
        raise Refused(f"N diverso fra affermazione ({claim.N}) e disegno ({design.N}): la popolazione è cambiata dopo "
                      "l'estrazione?")
    return method


def tally(design: Design, draws: Sequence, outcomes: Mapping | Iterable[tuple]) -> Counts:
    """Conteggi degli esiti delle estrazioni, senza perdere nulla in silenzio. draws: le unità estratte nell'ordine
    dell'estrazione (con reinserimento un'unità può ripetersi e conta per ogni estrazione); outcomes: un dizionario o
    coppie (unità, esito), un esito per unità."""
    draws = list(draws)
    if len(draws) != design.n:
        raise Refused(f"{len(draws)} estrazioni, il disegno ne registra {design.n}: la numerosità non cambia dopo "
                      "l'estrazione")
    if design.estrazione != PPS and len(set(draws)) != len(draws):
        raise Refused("unità ripetute in un'estrazione senza reinserimento")
    found = {}
    for unit, label in (outcomes.items() if isinstance(outcomes, Mapping) else outcomes):
        if unit in found:
            raise Refused(f"{unit}: esito ripetuto")
        if label not in OUTCOMES:
            raise Refused(f"{unit}: esito {label!r} sconosciuto; ammessi: {OUTCOMES}")
        found[unit] = label
    drawn = set(draws)
    missing = [u for u in dict.fromkeys(draws) if u not in found]
    if missing:
        raise Refused(f"{len(missing)} unità estratte senza esito, per esempio {missing[:3]}: un'unità non giudicata si "
                      "segna 'mancante', non si toglie")
    extra = [u for u in found if u not in drawn]
    if extra:
        raise Refused(f"esiti di {len(extra)} unità non estratte, per esempio {extra[:3]}")
    return Counts(**Counter(found[u] for u in draws))


# ---------------------------------------------------------------- risultati

def describe(counts: Counts) -> dict:
    """Conteggi descrittivi delle unità guardate, casi mirati compresi: non sono una stima della popolazione."""
    n = counts.n
    if n < 1:
        raise ValueError("nessun esito")
    return {"tipo": "descrittivo", "nota": "conteggi delle unità guardate, non una stima della popolazione", "n": n,
            **asdict(counts), "non_approvati": counts.non_approvati, "non_risolti": counts.non_risolti,
            "quota_non_risolta": f"{counts.non_risolti}/{n}"}


def estimate(claim: Claim, design: Design, counts: Counts, alpha=ALPHA, simultaneous: int = 1) -> dict:
    """Limite superiore per una affermazione su uno strato, in tre parti separate (conteggi descrittivi, limite valido
    per il disegno, conclusione condizionata al revisore). simultaneous: numero di affermazioni della famiglia, per
    Bonferroni (estimate_simultaneous)."""
    method = check_compatible(claim, design)
    if counts.n != design.n:
        raise Refused(f"{counts.n} esiti per {design.n} estrazioni: ogni estrazione ha un esito, anche 'mancante'")
    family = _alpha(alpha)
    a = bonferroni(family, simultaneous)
    m = _int("simultaneous", simultaneous, 1)
    k, N, n = counts.non_approvati, design.N, design.n
    unit = UNIT_NAMES[design.unita]
    where = f"popolazione {design.popolazione}, strato {design.strato}"
    level = f"{float(1 - a):g}"
    if method == "ipergeometrico":
        count = hypergeometric_upper(N, n, k, a)
        value = count / N
        if n == N:
            method, confidence = "censimento", 1.0
            statement = f"censimento di {N} {unit} ({where}): {k} unità non approvate, quota esatta {k}/{N}"
        else:
            confidence = float(1 - a)
            statement = (f"con probabilità almeno {level} sull'estrazione dichiarata ({n} {unit} su {N}, senza "
                         f"reinserimento con probabilità uguali), al più {count} unità su {N} ({value:.4%}; {where}) "
                         "non sarebbero approvate dalla revisione")
        subject = f"la quota di unità ({unit}) errate"
    else:
        count, value, confidence = None, clopper_pearson_upper(n, k, a), float(1 - a)
        statement = (f"con probabilità almeno {level} sulle {n} estrazioni con reinserimento e probabilità proporzionale "
                     f"ai target-token, la quota dei target-token in QA che la revisione non approverebbe ({where}) è al "
                     f"più {value:.4%}")
        subject = "la quota dei target-token in QA errate"
    return {
        "affermazione": asdict(claim), "disegno": asdict(design),
        "conteggi": describe(counts),
        "limite": {"tipo": "valido per il disegno dichiarato", "metodo": method, "verso": "superiore unilaterale",
                   "alpha": str(a), "alpha_famiglia": str(family), "affermazioni_simultanee": m,
                   "confidenza": confidence, "k": k, "N": N, "n": n, "D_U": count, "valore": value,
                   "valore_esatto": None if count is None else f"{count}/{N}",
                   "k_comprende": "errati, incerti, non giudicabili, mancanti e approvazioni non indipendenti (caso "
                                  "peggiore)",
                   "enunciato": statement},
        "conclusione": {"tipo": "condizionata all'attendibilità del revisore",
                        "condizione": "il revisore non approva unità errate per la proprietà: sensibilità non misurata "
                                      "(S4 §4)",
                        "enunciato": f"se la condizione vale, {subject} per «{claim.proprieta}» ({where}) è al più "
                                     f"{value:.4%}, con la stessa confidenza; altrimenti il limite vale solo per gli "
                                     "esiti della revisione",
                        "correzione_per_sensibilita": "non applicata: corrected_upper non è adottato (S4 §4)",
                        "esiti_mancanti": counts.mancante, "completa": counts.mancante == 0},
        "certificazione": "nessuna: il calcolo non certifica, non promuove a gold e non applica soglie (D4, D5)",
    }


def estimate_simultaneous(items: Iterable[tuple[Claim, Design, Counts]], alpha=ALPHA) -> list[dict]:
    """m affermazioni simultanee con livello di famiglia alpha, ciascuna a alpha / m (Bonferroni). I risultati restano
    separati, uno per affermazione e strato: nessuna unione automatica."""
    items = list(items)
    if not items:
        raise ValueError("nessuna affermazione")
    keys = [(c.proprieta, c.quantita, c.popolazione, c.strato) for c, _, _ in items]
    if len(set(keys)) != len(keys):
        raise Refused("affermazione ripetuta nella stessa famiglia")
    return [estimate(c, d, k, alpha, len(items)) for c, d, k in items]
