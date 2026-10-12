"""Recueil d'algebre, chapitre I : formes algebrique, trigonometrique, exponentielle.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1, 3, 4, 5,
6, 8 et 11 (sections I.1.4 et I.2.5). Chaque fonction rend ce que le corrige
compose ; les tests le confrontent a la page de solutions du syllabus.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.algebre.complexes_exacts import forme_polaire, par_moivre

IMAG: Final = sp.I
R3: Final = sp.sqrt(3)
Polaire = tuple[sp.Expr, sp.Expr]

ENONCES_1: Final = {
    "a": (1 + 2 * IMAG) - (3 - 4 * IMAG) + (5 - 6 * IMAG),
    "b": sp.Mul(7 - 8 * IMAG, 9 + IMAG, evaluate=False),
    "c": sp.Pow(4 + 3 * IMAG, 2, evaluate=False),
    "d": sp.Mul(
        -2 + 3 * IMAG, sp.Pow(-3 + 4 * IMAG, -1, evaluate=False), evaluate=False
    ),
    "e": sp.Pow(IMAG, 5, evaluate=False),
}
ENONCES_3: Final = {
    "a": 1 + R3 * IMAG,
    "b": 2 * IMAG,
    "c": sp.Integer(-3),
    "d": 1 / (1 + IMAG),
    "e": -5 + 4 * IMAG,
}
Z_UN: Final = R3 - IMAG
Z_DEUX: Final = -3 * IMAG


def algebrique(z: sp.Expr) -> sp.Expr:
    """`a + bi`, parties reelle et imaginaire simplifiees."""
    reelle, imaginaire = sp.sympify(z).doit().as_real_imag()
    return sp.simplify(reelle) + sp.simplify(imaginaire) * IMAG


def exercice_1() -> dict[str, sp.Expr]:
    """Les cinq nombres de l'exercice 1, sous forme algebrique."""
    return {lettre: algebrique(z) for lettre, z in ENONCES_1.items()}


def exercice_3() -> dict[str, Polaire]:
    """Module et argument principal des cinq nombres de l'exercice 3."""
    return {lettre: forme_polaire(z) for lettre, z in ENONCES_3.items()}


def exercice_4() -> Polaire:
    """`(1 + i)^3 / (1 + i racine(3))^4` : module et argument principal."""
    return forme_polaire((1 + IMAG) ** 3 / (1 + IMAG * R3) ** 4)


def exercice_5() -> sp.Expr:
    """`(racine(3) + i)^4`, par la formule de Moivre."""
    return algebrique(par_moivre(R3 + IMAG, 4))


def exercice_6() -> sp.Expr:
    """La somme des deux quotients de l'exercice 6, chacun par Moivre."""
    premier = par_moivre(-1 + IMAG * R3, 15) / par_moivre(1 - IMAG, 20)
    second = par_moivre(-1 - IMAG * R3, 15) / par_moivre(1 + IMAG, 20)
    return algebrique(premier + second)


def exercice_8() -> dict[str, Polaire]:
    """Module et argument de `z1 z2` et de `z1 / z2`."""
    return {
        "produit": forme_polaire(Z_UN * Z_DEUX),
        "quotient": forme_polaire(Z_UN / Z_DEUX),
    }


def exercice_11() -> dict[str, sp.Expr]:
    """a) et b) sous forme algebrique, c) sous forme exponentielle.

    Pour c), le produit multiplie les modules et ajoute les arguments :
    on ne developpe rien.
    """
    module, angle = forme_polaire(R3 - IMAG)
    return {
        "a": algebrique(3 * IMAG + 2 * sp.exp(IMAG * sp.pi)),
        "b": algebrique(sp.exp(IMAG * sp.pi / 3) - (3 + 2 * IMAG)),
        "c": 3 * module * sp.exp(IMAG * (sp.pi / 5 + angle)),
    }
