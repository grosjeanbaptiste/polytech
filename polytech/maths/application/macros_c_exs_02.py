r"""Macros des exercices 2 de mathematique, seance III : systemes differentiels."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import courbes_exs_02 as co
from polytech.maths.application import systemes_exs_02 as sy
from polytech.maths.application.formatage import coordonnees, formule
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T


def par_mode(composante: sp.Expr) -> sp.Expr:
    """La composante, rangee par exponentielle : c'est ainsi que la feuille l'ecrit."""
    return sp.collect(sp.expand(composante), [sp.exp(-T), sp.exp(3 * T)])


def _trajectoires() -> dict[str, str]:
    return {
        f"MathDeuxPhase{lettre}": coordonnees(premiere, seconde, 3)
        for lettre, (premiere, seconde) in zip(
            "ABCDEFGHIJKL", co.trajectoires_de_phase(), strict=True
        )
    }


def macros_de_la_seance_trois() -> dict[str, str]:
    """Les formules de la seance III."""
    (lente, vecteur_lent), (rapide, vecteur_rapide) = sy.modes_du_1()
    wronskien, _ = sy.wronskiens_du_1()
    par_laplace, _ = sy.solutions_du_2()
    troisieme = sy.solution_du_3()
    image_x, image_y = sy.images_du_4()
    x_quatre, y_quatre = sy.solution_du_4()
    x_cinq, y_cinq = sy.solution_du_5()
    permanent_x, permanent_y = sy.permanent_du_5()
    caracteristique = P * sp.eye(2) - sy.DEUXIEME.matrice
    commun = sp.factor(caracteristique.det())
    return {
        **_trajectoires(),
        "MathDeuxValeurLente": formule(lente),
        "MathDeuxVecteurLent": formule(vecteur_lent),
        "MathDeuxValeurRapide": formule(rapide),
        "MathDeuxVecteurRapide": formule(vecteur_rapide),
        "MathDeuxWronskien": formule(wronskien),
        "MathDeuxNature": sy.nature_du_1().value,
        "MathDeuxNatureDeux": sy.natures()[1].value,
        "MathDeuxNatureTrois": sy.natures()[2].value,
        "MathDeuxRetour": formule(sy.retour_a_l_origine_du_2()),
        "MathDeuxResolvanteNumerateur": formule(
            (sy.resolvante_du_2() * commun).applyfunc(sp.cancel)
        ),
        "MathDeuxResolvanteDenominateur": formule(commun),
        "MathDeuxSystemeDeuxA": formule(par_mode(par_laplace[0])),
        "MathDeuxSystemeDeuxB": formule(par_mode(par_laplace[1])),
        "MathDeuxSystemeTroisA": formule(troisieme[0]),
        "MathDeuxSystemeTroisB": formule(troisieme[1]),
        "MathDeuxSystemeQuatreImageX": formule(sp.factor(image_x)),
        "MathDeuxSystemeQuatreImageY": formule(sp.apart(image_y, P)),
        "MathDeuxSystemeQuatreX": formule(sp.expand(x_quatre)),
        "MathDeuxSystemeQuatreY": formule(sp.expand(y_quatre)),
        "MathDeuxSystemeCinqX": formule(sp.expand(x_cinq)),
        "MathDeuxSystemeCinqY": formule(sp.expand(y_cinq)),
        "MathDeuxPermanentX": formule(permanent_x),
        "MathDeuxPermanentY": formule(permanent_y),
    }
