"""La resolution par substitution, etape par etape, comme la planche 4 l'ecrit.

« La premiere equation s'ecrit aussi `y = 1 - 2x`. On remplace maintenant
`y` dans la deuxieme equation : `3x + 7(1 - 2x) = -2`, d'ou `11x = 9`,
`x = 9/11`. On en deduit `y`. N'oubliez pas de verifier que votre
solution fonctionne ! »

Chaque etape porte sa justification : c'est un raisonnement, non un
resultat. La verification finale en fait partie.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.raisonnement.domain.etapes import Etape, Raisonnement

if TYPE_CHECKING:
    from polytech.maths.domain.algebre.systemes import SystemeLineaire

SOURCE: Final = (
    "Mathematique/SLIDES/Slides_01_Algebre_lineaire_et_integrales.pdf, planche 4"
)
INCONNUES: Final = sp.symbols("x y")
ORDRE_DEUX: Final = (2, 2)


def _isolee(systeme: SystemeLineaire) -> int:
    """L'inconnue a isoler de la premiere equation : celle de coefficient 1."""
    coefficients = list(systeme.matrice[0, :])
    simples = [j for j, a in enumerate(coefficients) if abs(a) == 1]
    return simples[0] if simples else 1


def _etapes(systeme: SystemeLineaire) -> tuple[list[Etape], dict[sp.Symbol, sp.Expr]]:
    j = _isolee(systeme)
    isolee, autre = INCONNUES[j], INCONNUES[1 - j]
    (a, b), (c, d) = systeme.matrice.tolist()
    e, f = systeme.second_membre
    expression = sp.solve(a * INCONNUES[0] + b * INCONNUES[1] - e, isolee)[0]
    gauche = c * INCONNUES[0] + d * INCONNUES[1]
    ecrite = str(gauche).replace(str(isolee), f"({expression})")
    reduite = sp.expand(gauche.subs(isolee, expression) - f)
    pente = reduite.coeff(autre)
    signe = 1 if pente > 0 else -1
    valeur = sp.solve(reduite, autre)[0]
    retour = expression.subs(autre, valeur)
    etapes = [
        Etape(f"{isolee} = {expression}", "c'est la premiere equation", SOURCE),
        Etape(
            f"{signe * pente}*{autre} = {signe * (pente * autre - reduite)}",
            f"on remplace {isolee} dans la deuxieme equation : {ecrite} = {f}",
            SOURCE,
        ),
        Etape(f"{autre} = {valeur}", "on divise par le coefficient", SOURCE),
        Etape(
            f"{isolee} = {retour}", f"on reporte dans {isolee} = {expression}", SOURCE
        ),
    ]
    return etapes, {isolee: retour, autre: valeur}


def par_substitution(systeme: SystemeLineaire) -> Raisonnement:
    """Les etapes de la planche 4, pour un systeme de deux equations."""
    if systeme.matrice.shape != ORDRE_DEUX:
        forme = f"{systeme.matrice.rows} x {systeme.matrice.cols}"
        msg = f"systeme 2 x 2 attendu, {forme} recu"
        raise ValueError(msg)
    if systeme.matrice.det() == 0:
        msg = "determinant nul : l'inconnue restante disparait de la deuxieme equation"
        raise ValueError(msg)
    etapes, valeurs = _etapes(systeme)
    point = sp.Matrix([valeurs[INCONNUES[0]], valeurs[INCONNUES[1]]])
    verification = Etape(
        "la solution verifie les deux equations",
        "reportee dans le systeme, elle rend les deux seconds membres",
        SOURCE,
    )
    conclusion = f"x = {point[0]}, y = {point[1]}"
    return Raisonnement(
        "Que vaut la solution du systeme ?", (*etapes, verification), conclusion
    )
