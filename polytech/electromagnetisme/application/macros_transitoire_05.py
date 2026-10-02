"""Macros de l'exercice 5 du regime transitoire : le systeme 5x5 resolu.

Entierement litteral, comme les exercices 5 de statique et 11 des
forces : le jeu publie des expressions LaTeX et non des valeurs.

Les cinq equations sont publiees **telles que la bibliotheque les
resout**, et non recopiees dans le document. C'est ce qui garantit que
le systeme compose est celui qui a ete resolu -- un systeme recopie
pourrait diverger du systeme calcule sans que rien ne le signale.

Les transformees sont publiees en **numerateur et denominateur
separes**, chacun collecte selon les puissances de `p`. Deux raisons :
l'expression entiere depasse six cents caracteres de LaTeX et deborderait
la page, et surtout la forme collectee est celle qui **se lit** -- les
deux theoremes de valeur s'y ramenent a un rapport de coefficients
extremes.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    ecart_des_degres,
    valeur_finale,
    valeur_initiale,
)
from polytech.electromagnetisme.domain.transitoire.systeme import (
    COURANT_1,
    COURANT_2,
    ETAT_INITIAL_C1,
    degre_caracteristique,
    equations,
    solution,
    terme_source_de_la_premiere_maille,
)

NOMS_DES_EQUATIONS = ("Une", "Deux", "Trois", "Quatre", "Cinq")
"""Les cinq equations, nommees en lettres : une macro LaTeX n'a pas de chiffre."""


def _collecte(expression: sp.Expr) -> str:
    """Developpe puis regroupe selon les puissances de `p`.

    C'est la forme qui se lit : les theoremes de valeur se ramenent alors
    a un rapport de coefficients extremes, visible a l'oeil.
    """
    return expression_latex(sp.collect(sp.expand(expression), VARIABLE))


def _fraction(transformee: sp.Expr) -> tuple[str, str]:
    """Le numerateur et le denominateur, collectes chacun."""
    numerateur, denominateur = sp.fraction(sp.cancel(sp.together(transformee)))
    return _collecte(numerateur), _collecte(denominateur)


def _macros_du_systeme() -> dict[str, str]:
    """Les cinq equations, telles que la bibliotheque les resout."""
    return {
        f"TransitoireCinqEquation{nom}": expression_latex(equation)
        for nom, equation in zip(NOMS_DES_EQUATIONS, equations(), strict=True)
    }


def _macros_des_transformees() -> dict[str, str]:
    """Les deux courants demandes, en numerateur et denominateur."""
    numerateur_un, denominateur_un = _fraction(solution()[COURANT_1])
    numerateur_deux, denominateur_deux = _fraction(solution()[COURANT_2])
    return {
        "TransitoireCinqNumerateurUn": numerateur_un,
        "TransitoireCinqDenominateurUn": denominateur_un,
        "TransitoireCinqNumerateurDeux": numerateur_deux,
        "TransitoireCinqDenominateurDeux": denominateur_deux,
    }


def _macros_des_limites() -> dict[str, str]:
    """Les quatre limites, et les deux ecarts de degres qui les expliquent."""
    courant_un, courant_deux = solution()[COURANT_1], solution()[COURANT_2]
    return {
        "TransitoireCinqUnInitial": expression_latex(valeur_initiale(courant_un)),
        "TransitoireCinqUnFinal": expression_latex(valeur_finale(courant_un)),
        "TransitoireCinqDeuxInitial": expression_latex(valeur_initiale(courant_deux)),
        "TransitoireCinqDeuxFinal": expression_latex(valeur_finale(courant_deux)),
        "TransitoireCinqEcartUn": str(ecart_des_degres(courant_un)),
        "TransitoireCinqEcartDeux": str(ecart_des_degres(courant_deux)),
    }


def macros_transitoire_cinq() -> dict[str, str]:
    """Le systeme, les deux transformees, les limites et l'ordre."""
    return {
        **_macros_du_systeme(),
        **_macros_des_transformees(),
        **_macros_des_limites(),
        "TransitoireCinqOrdre": str(degre_caracteristique()),
        "TransitoireCinqEtatInitial": expression_latex(ETAT_INITIAL_C1),
        "TransitoireCinqTermeSource": expression_latex(
            terme_source_de_la_premiere_maille(),
        ),
    }
