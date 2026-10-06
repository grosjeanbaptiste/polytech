r"""Macros des exercices 1 de mathematique : integration, matrices, determinants.

Le prefixe est `MathUn`. Les macros sont des *formules*, composees par
`sympy` : le document les inclut en mode mathematique. Un nom de macro
n'a pas de chiffre : les numeros de la feuille s'ecrivent en lettres,
`\MathUnPrimitiveDeuxA` pour la premiere primitive de l'exercice 2.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application import algebre_exs_01 as al
from polytech.maths.application.courbes_exs_01 import courbes_du_12
from polytech.maths.application.definies_exs_01 import INTEGRALES
from polytech.maths.application.formatage import coordonnees, formule
from polytech.maths.application.primitives_exs_01 import (
    PRIMITIVES,
    formes_linearisees,
    primitive_du_12_sur_r,
)
from polytech.maths.domain.symboles import VARIABLE as X

NOMBRES: Final = (
    "Un",
    "Deux",
    "Trois",
    "Quatre",
    "Cinq",
    "Six",
    "Sept",
    "Huit",
    "Neuf",
    "Dix",
    "Onze",
    "Douze",
    "Treize",
    "Quatorze",
    "Quinze",
    "Seize",
    "DixSept",
    "DixHuit",
    "DixNeuf",
)
"""Les numeros de 1 a 19, pour les noms de macros."""
T: Final = sp.Symbol("t", real=True)
EN_T: Final = frozenset({"2a", "2b", "2c", "2d", "5"})
"""Les primitives que la feuille ecrit en `t` ; les integrales definies aussi."""


def nom(numero: str) -> str:
    """`2a` devient `DeuxA`, `14` devient `Quatorze`."""
    chiffres = "".join(c for c in numero if c.isdigit())
    return NOMBRES[int(chiffres) - 1] + numero[len(chiffres) :].upper()


def _primitives() -> dict[str, str]:
    macros = {}
    for numero, resoudre in PRIMITIVES.items():
        resolution = resoudre()
        variable = T if numero in EN_T else X
        for lettre, etape in zip("ABC", resolution.etapes, strict=False):
            macros[f"MathUnEtape{nom(numero)}{lettre}"] = formule(
                etape.subs(X, variable)
            )
        primitive = resolution.primitive.subs(X, variable)
        macros[f"MathUnPrimitive{nom(numero)}"] = formule(primitive)
    cosinus, hyperbolique = formes_linearisees()
    macros["MathUnLineaireCos"] = formule(cosinus.subs(X, T))
    macros["MathUnLineaireCosh"] = formule(hyperbolique.subs(X, T))
    macros["MathUnDouzePartout"] = formule(primitive_du_12_sur_r())
    return macros


def _definies() -> dict[str, str]:
    macros = {}
    for numero, etablir in INTEGRALES.items():
        integrale = etablir()
        if integrale.primitive is not None:
            primitive = integrale.primitive.subs(X, T)
            macros[f"MathUnPrimitiveDefinie{NOMBRES[numero - 1]}"] = formule(primitive)
    return macros


def _vecteur(composantes: tuple[sp.Expr, ...]) -> str:
    """Un vecteur ecrit en ligne, entre parentheses : `(1, -1, 0)`."""
    return "(" + ",\\ ".join(formule(c) for c in composantes) + ")"


def _algebre() -> dict[str, str]:
    premier, deuxieme, troisieme = al.solutions_des_systemes()
    a, b, c = al.SECONDS_MEMBRES
    determinant = sp.expand(al.determinant())
    macros = {
        "MathUnDeterminant": formule(determinant),
        "MathUnComatrice": formule(al.comatrice()),
        "MathUnInverseNumerateur": formule(sp.simplify(al.inverse() * -determinant)),
        "MathUnSystemeUn": _vecteur(premier),
        "MathUnSystemeDeux": _vecteur(deuxieme),
        "MathUnDetSystemeUn": formule(al.determinants_des_systemes()[0]),
        "MathUnDetSystemeTrois": formule(al.determinants_des_systemes()[2]),
        "MathUnDetParametriqueUn": formule(sp.factor(al.determinant_parametrique(1))),
        "MathUnDetParametriqueDeux": formule(al.determinant_parametrique(2)),
        "MathUnNoyauAB": _vecteur(al.solutions_non_nulles(1, {b: a})[0]),
        "MathUnNoyauBC": _vecteur(al.solutions_non_nulles(1, {c: b})[0]),
        "MathUnNoyauAC": _vecteur(al.solutions_non_nulles(1, {c: a})[0]),
    }
    for lettre, terme in zip("ABC", al.developpement(), strict=True):
        macros[f"MathUnTerme{lettre}"] = formule(terme)
    for lettre, composante in zip("XYZ", troisieme, strict=True):
        macros[f"MathUnSystemeTrois{lettre}"] = formule(sp.factor(composante))
    return macros


def _courbes() -> dict[str, str]:
    branches, (abscisses, partout) = courbes_du_12()
    macros = {"MathUnCourbePartout": coordonnees(abscisses, partout, 3)}
    for lettre, (x, y) in zip("ABC", branches, strict=True):
        macros[f"MathUnBranche{lettre}"] = coordonnees(x, y, 3)
    return macros


def macros_des_exercices_un() -> dict[str, str]:
    """Toutes les formules que le corrige des exercices 1 compose."""
    return {**_primitives(), **_definies(), **_algebre(), **_courbes()}
