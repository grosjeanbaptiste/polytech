"""Le regime sinusoidal d'une branche : impedance, reactance, admittance.

Le syllabus d'electricite generale (16.6 a 16.12). En substituts
complexes, l'equation de la branche fondamentale devient algebrique :
`U = E + Z I` (16.7.13), avec l'impedance complexe
`Z = R + j omega L + 1 / (j omega C) = R + j X` (16.7.5). Son module est
l'impedance, son argument `psi` ; `X = omega L - 1 / (omega C)` (16.7.7)
est la reactance. Sans force electromotrice, le dephasage `phi` de la
tension sur le courant vaut `psi` (16.7.12).

Selon que `omega L` depasse ou non `1 / (omega C)`, la branche se
comporte comme une bobine d'inductance apparente (16.9.1) ou un
condensateur de capacite apparente (16.9.2) ; a l'egalite, elle resonne.

L'admittance complexe est l'inverse de l'impedance, `G + j B` (16.12.1) :
la conductance `R / Z^2` (16.12.4) et la susceptance `-X / Z^2` (16.12.5).
"""

from __future__ import annotations

from enum import Enum

import sympy as sp

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


class Nature(Enum):
    """Ce que le signe de la reactance dit de la branche."""

    INDUCTIVE = "inductive"
    CAPACITIVE = "capacitive"
    RESONANTE = "resonante"


def reactance(inductance: Grandeur, capacite: Grandeur, pulsation: Grandeur) -> sp.Expr:
    """`X = omega L - 1 / (omega C)` (16.7.7) ; sans condensateur, `C` infinie."""
    return pulsation * inductance - 1 / (pulsation * sp.sympify(capacite))


def impedance_de_la_branche(
    resistance: Grandeur, inductance: Grandeur, capacite: Grandeur, pulsation: Grandeur
) -> sp.Expr:
    """`Z = R + j X` (16.7.5), en ohms."""
    return resistance + sp.I * reactance(inductance, capacite, pulsation)


def module_et_argument(impedance: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(Z, psi)` de (16.7.6) et (16.7.8)."""
    if impedance == 0:
        msg = "impedance nulle : son argument n'est pas defini"
        raise ValueError(msg)
    return sp.simplify(sp.Abs(impedance)), sp.simplify(sp.arg(impedance))


def courant_de_la_branche(
    tension: sp.Expr, impedance: sp.Expr, *, fem: Grandeur = 0
) -> sp.Expr:
    """`I = (U - E) / Z` (16.7.13)."""
    return sp.simplify((tension - fem) / impedance)


def nature_de_la_branche(valeur: Grandeur) -> Nature:
    """Inductive si la reactance est positive, capacitive si elle est negative."""
    if valeur == 0:
        return Nature.RESONANTE
    return Nature.INDUCTIVE if valeur > 0 else Nature.CAPACITIVE


def inductance_apparente(
    inductance: Grandeur, capacite: Grandeur, pulsation: Grandeur
) -> sp.Expr:
    """`L_app = (1 - 1 / (omega^2 L C)) L` (16.9.1)."""
    valeur = reactance(inductance, capacite, pulsation)
    if nature_de_la_branche(valeur) is not Nature.INDUCTIVE:
        msg = f"branche non inductive : reactance {valeur} ohm"
        raise ValueError(msg)
    return valeur / pulsation


def capacite_apparente(
    inductance: Grandeur, capacite: Grandeur, pulsation: Grandeur
) -> sp.Expr:
    """`C_app = C / (1 - omega^2 L C)` (16.9.2)."""
    valeur = reactance(inductance, capacite, pulsation)
    if nature_de_la_branche(valeur) is not Nature.CAPACITIVE:
        msg = f"branche non capacitive : reactance {valeur} ohm"
        raise ValueError(msg)
    return -1 / (pulsation * valeur)


def admittance(impedance: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(G, B) = (R / Z^2, -X / Z^2)` : (16.12.4) et (16.12.5)."""
    inverse = sp.simplify(1 / impedance)
    return sp.simplify(sp.re(inverse)), sp.simplify(sp.im(inverse))
