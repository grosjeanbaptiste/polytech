r"""Ce que les corriges du recueil d'algebre ont en commun : l'ecriture.

Le recueil -- `Mathematique/SYLLABUS/Syllabus_Algebre.pdf` -- ecrit l'unite
imaginaire `i`. `formatage.formule` l'ecrit `j` ; les macros du recueil
passent donc par `ecriture`, qui suit le syllabus.
"""

from __future__ import annotations

import re
from itertools import takewhile
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application.formatage import nombre
from polytech.maths.application.macros_exs_01 import NOMBRES

if TYPE_CHECKING:
    from collections.abc import Iterable

DECIMALES = 3
DIZAINES = ("Vingt", "Trente", "Quarante", "Cinquante")
MAXIMUM = 59
"""Le plus grand numero d'exercice nommable ; le recueil s'arrete a 45."""


def ecriture(expression: sp.Basic) -> str:
    """Le LaTeX d'une expression exacte, dans les notations du syllabus."""
    rendu: str = sp.latex(
        expression, ln_notation=True, inv_trig_style="full", mat_delim="("
    )
    return rendu


def liste(expressions: Iterable[sp.Basic]) -> str:
    r"""Les expressions a la suite, separees par `,\;`."""
    return r",\; ".join(ecriture(e) for e in expressions)


def en_lettres(numero: int) -> str:
    """`DixHuit` pour 18 : un nom de macro n'a pas de chiffre."""
    if numero <= len(NOMBRES):
        return NOMBRES[numero - 1]
    dizaine, unite = divmod(numero, 10)
    return DIZAINES[dizaine - 2] + (NOMBRES[unite - 1] if unite else "")


def nom(prefixe: str, numero: int, suffixe: str = "") -> str:
    """`AlgUnDixHuitC` pour l'exercice 18 c) : un nom de macro n'a pas de chiffre."""
    if not 1 <= numero <= MAXIMUM:
        msg = f"numero d'exercice hors de la table : {numero}"
        raise ValueError(msg)
    return f"{prefixe}{en_lettres(numero)}{suffixe[:1].upper()}{suffixe[1:]}"


def nom_de_la_cle(prefixe: str, cle: str, suffixe: str = "") -> str:
    """`36W1` devient `...TrenteSixWUn` : le numero, puis le reste sans chiffre."""
    chiffres = "".join(takewhile(str.isdigit, cle))
    if not chiffres:
        msg = f"cle sans numero d'exercice : {cle!r}"
        raise ValueError(msg)
    reste = re.sub(r"\d+", lambda m: en_lettres(int(m.group())), cle[len(chiffres) :])
    return nom(prefixe, int(chiffres), reste + suffixe)


def points(complexes: Iterable[sp.Expr]) -> str:
    """`(x,y) (x,y) ...` : les affixes, pour une figure du plan de Gauss."""
    valeurs = [complex(sp.N(z)) for z in complexes]
    return " ".join(
        f"({nombre(v.real, DECIMALES)},{nombre(v.imag, DECIMALES)})" for v in valeurs
    )


def operations(journal: Iterable[str]) -> str:
    r"""Le journal d'un echelonnement, `l2 -> l2 - 2 l1`, en LaTeX."""
    rendues = []
    for ligne in journal:
        rendu = ligne.replace("<->", r"\leftrightarrow").replace("->", r"\to")
        rendu = re.sub(r"(\d+)/(\d+) ", r"\\tfrac{\1}{\2}\\, ", rendu)
        rendues.append(re.sub(r"l(\d+)", r"l_{\1}", rendu))
    return r";\quad ".join(rendues)
