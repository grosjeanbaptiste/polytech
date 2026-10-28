"""Comparer deux resultats : le critere vit ici, pas dans chaque test.

Deux expressions sont **egales** quand leur ecart s'annule : la simplification
symbolique de sympy n'y suffit pas (elle laisse des `|w|` et des
`sqrt(x^2)`), l'ecart est donc evalue en plusieurs points, chaque symbole
libre recevant des valeurs compatibles avec ses hypotheses (un symbole
positif ne prend que des valeurs positives).

Deux nombres sont **proches** a une tolerance donnee, ou **eloignes** d'au
moins un ecart ; un dephasage se lit en degres, comme le cours l'ecrit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.algebre.complexes import argument
from polytech.noyau import core

if TYPE_CHECKING:
    from collections.abc import Iterable

PRECISION = 30
"""Chiffres significatifs de l'evaluation : l'ecart d'arrondi est negligeable."""
TOLERANCE_DE_L_ECART = 1e-12
_VALEURS = (sp.Integer(1), sp.Rational(7, 3), sp.Rational(5, 2))
_ECHANTILLONS = 3


def _valeur(symbole: sp.Symbol, rang: int) -> sp.Expr:
    base = _VALEURS[rang % len(_VALEURS)]
    if symbole.is_negative:
        return -base
    if symbole.is_positive or rang % 2 == 0:
        return base
    return -base


def _ecart_nul_en_echantillons(ecart: sp.Expr) -> bool:
    libres = sorted(ecart.free_symbols, key=str)
    for rang in range(_ECHANTILLONS):
        point = {s: _valeur(s, rang + i) for i, s in enumerate(libres)}
        valeur = complex(sp.N(ecart.subs(point), PRECISION))
        if abs(valeur) > TOLERANCE_DE_L_ECART:
            return False
    return True


def _ecart_nul(ecart: sp.Expr) -> bool:
    """Exact quand sympy conclut (zero, ou rationnel non nul), sinon numerique."""
    simplifie = sp.simplify(ecart)
    if simplifie == 0:
        return True
    if simplifie.is_Rational:
        return False
    return _ecart_nul_en_echantillons(ecart)


def egales(obtenue: object, attendue: object) -> bool:
    """L'ecart des deux expressions (ou matrices, terme a terme) est nul.

    Deux temps : l'ecart est d'abord simplifie (zero : egales ; rationnel
    non nul : distinctes, exactement, quel que soit son ordre de grandeur) ;
    sinon il est evalue **numeriquement** en 3 jeux de points avec une
    tolerance de 1e-12. Limite connue de ce dernier recours : deux
    expressions non simplifiables qui different de moins de 1e-12 passent.

    Leve `TypeError` si l'ecart ne s'evalue pas en nombre (fonction non
    definie, par exemple) : conclure serait inventer.
    """
    premiere, seconde = sp.sympify(obtenue), sp.sympify(attendue)
    if isinstance(premiere, sp.MatrixBase) or isinstance(seconde, sp.MatrixBase):
        return premiere.shape == seconde.shape and all(
            _ecart_nul(sp.sympify(a - b))
            for a, b in zip(premiere, seconde, strict=True)
        )
    return _ecart_nul(premiere - seconde)


def proches(valeur: complex, reference: complex, tolerance: float) -> bool:
    """`|valeur - reference| <= tolerance`."""
    return core.proches(valeur, reference, tolerance)


def eloignees(valeur: complex, reference: complex, d_au_moins: float) -> bool:
    """`|valeur - reference| > d_au_moins` : la valeur citee est bien exclue."""
    return core.eloignees(valeur, reference, d_au_moins)


def proches_relativement(valeur: complex, reference: complex, tolerance: float) -> bool:
    """`|valeur - reference| <= tolerance |reference|` ; en zero, l'egalite exacte."""
    return core.proches_relativement(valeur, reference, tolerance)


def suites_proches(
    valeurs: Iterable[float],
    references: Iterable[float],
    tolerance: float,
    *,
    relativement: bool = False,
) -> bool:
    """Deux suites de meme longueur, proches terme a terme (absolu par defaut)."""
    valeurs, references = list(valeurs), list(references)
    if len(valeurs) != len(references):
        msg = f"longueurs differentes : {len(valeurs)} et {len(references)}"
        raise ValueError(msg)
    proche = proches_relativement if relativement else proches
    return all(
        proche(v, r, tolerance) for v, r in zip(valeurs, references, strict=True)
    )


def suite_nulle(valeurs: Iterable[float], tolerance: float) -> bool:
    """Chaque terme de la suite est nul a `tolerance` pres."""
    return all(proches(v, 0.0, tolerance) for v in valeurs)


def est_positive(expression: object) -> bool:
    """L'expression, simplifiee, est strictement positive (indetermine : non)."""
    return sp.simplify(sp.sympify(expression)).is_positive is True


def est_negative(expression: object) -> bool:
    """L'expression, simplifiee, est strictement negative (indetermine : non)."""
    return sp.simplify(sp.sympify(expression)).is_negative is True


def egales_arrondies(valeur: float, attendue: float, decimales: int) -> bool:
    """Les deux nombres arrondis a `decimales` chiffres apres la virgule coincident."""
    if decimales < 0:
        msg = f"decimales negatives : {decimales}"
        raise ValueError(msg)
    return round(valeur, decimales) == round(attendue, decimales)


def rapport_au_dessus(valeur: float, reference: float, facteur: float) -> bool:
    """`valeur / reference > facteur` ; reference nulle : `ZeroDivisionError`."""
    return valeur / reference > facteur


def exponentielle(exposant: float) -> float:
    """`e^exposant`, en flottant."""
    return math.exp(exposant)


def en_degres(radians: float) -> float:
    """Les radians lus en degres."""
    return core.en_degres(radians)


def _complexe(z: object) -> complex:
    return complex(sp.N(sp.sympify(z), PRECISION))


def argument_en_degres(z: object) -> float:
    """L'argument de `z` dans `]-180, 180]` ; leve en zero (voir `argument`)."""
    return en_degres(argument(_complexe(z)))


def dephasage_en_degres(z: object, reference: object) -> float:
    """De combien `z` avance sur `reference` : l'argument de `z / reference`."""
    return argument_en_degres(sp.sympify(z) / sp.sympify(reference))
