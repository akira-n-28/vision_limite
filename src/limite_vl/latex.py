"""LaTeX delle formule di UniMER: dal formato a token separati da spazi (stile im2latex) a LaTeX naturale,
e confronto fra trascrizioni.

Nel formato im2latex ogni token è separato da uno spazio (`\\frac { 1 } { 2 }`, `0 . 5`); uno spazio di
controllo `\\ ` diventa il token `\\` seguito dal separatore. `despace` toglie i separatori e rimette uno
spazio solo dove serve: dopo una parola di controllo seguita da una lettera (`\\alpha b`), e nello spazio
di controllo. `latex_tokens` divide in token sia il formato spaziato sia quello naturale, così due
trascrizioni si confrontano a meno degli spazi.
"""
from __future__ import annotations

import re

_CONTROL_WORD_END = re.compile(r"\\[A-Za-z]+$")
_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\S")


def despace(spaced: str) -> str:
    out = ""
    for token in spaced.split():
        if token == "\\":  # spazio di controllo
            out += "\\ "
            continue
        if out and token[0].isalpha() and _CONTROL_WORD_END.search(out):
            out += " "
        out += token
    return out


def latex_tokens(text: str) -> list[str]:
    """Token LaTeX: parole di controllo, simboli di controllo (incluso `\\ `), singoli caratteri non spazio."""
    return _TOKEN.findall(text)


def same_latex(a: str, b: str) -> bool:
    return latex_tokens(a) == latex_tokens(b)
