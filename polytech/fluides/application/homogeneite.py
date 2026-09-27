"""Verifie l'homogeneite des champs avant d'en publier quoi que ce soit.

`dimensions.py` annonce verifier « la consequence sur chaque resultat
publie ». C'etait une intention : seuls des tests l'appelaient, si bien
qu'une formule inhomogene aurait pu entrer dans le corrige sans que rien ne
bronche. Ce module est le cas d'usage qui manquait -- il s'interpose entre
le calcul et la publication.

Les coefficients vivaient eux aussi dans un fichier de test. Ils sont un
fait sur l'enonce, pas sur sa verification : l'enonce donne des champs a
coefficients numeriques nus, et ``u = xt`` n'est une vitesse que si le 1
sous-entendu porte des secondes a la puissance -2. Ils sont donc ici, avec
les champs qu'ils qualifient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.dimensions import (
    TAUX,
    TEMPS,
    VITESSE,
    ExpressionNonHomogene,
    dimension_de,
    dimension_des_contributions,
)

if TYPE_CHECKING:
    import sympy as sp

    from polytech.fluides.domain.cinematique.champ import ChampDeVitesse

COEFFICIENT_B_U: Final = TEMPS**-2
"""u = xt : pour que ce soit une vitesse, le coefficient est en s^-2."""
COEFFICIENT_B_V: Final = TEMPS**-1
"""v = x - y : une longueur, donc un coefficient en s^-1."""
COEFFICIENT_C: Final = TEMPS**-1
"""Les deux composantes du champ C sont en x ou y : un seul coefficient."""


def _exiger_une_vitesse(expression: sp.Expr, coefficient: sp.Expr, nom: str) -> None:
    """Leve si ``expression`` n'est pas une vitesse sous ce coefficient.

    Homogene ne suffit pas. `u = x` sous un coefficient en s^-2 est
    parfaitement homogene et vaut une acceleration : seule la comparaison a
    la dimension attendue le refuse.
    """
    obtenue = dimension_de(expression, coefficient)
    if obtenue != VITESSE:
        msg = f"{nom} n'est pas une vitesse mais {obtenue} : {expression}"
        raise ExpressionNonHomogene(msg)


def verifier_l_homogeneite(
    champ: ChampDeVitesse,
    coefficient_u: sp.Expr,
    coefficient_v: sp.Expr,
) -> None:
    """Exige que ce champ et sa divergence soient dimensionnellement sains.

    La divergence est dimensionnee *contribution par contribution* : pour le
    champ B, du/dx porte le coefficient de u et dv/dy celui de v, si bien
    qu'un coefficient unique sur la somme donnerait un resultat absurde.
    """
    _exiger_une_vitesse(champ.u, coefficient_u, "u")
    _exiger_une_vitesse(champ.v, coefficient_v, "v")
    derivees = champ.derivees_partielles()
    taux = dimension_des_contributions(
        [(derivees.du_dx, coefficient_u), (derivees.dv_dy, coefficient_v)],
    )
    if taux != TAUX:
        msg = f"la divergence n'est pas un taux mais {taux}"
        raise ExpressionNonHomogene(msg)


def verifier_les_champs_publies() -> None:
    """Verifie les deux champs de l'enonce ; leve au premier defaut."""
    verifier_l_homogeneite(enonce.CHAMP_B, COEFFICIENT_B_U, COEFFICIENT_B_V)
    verifier_l_homogeneite(enonce.CHAMP_C, COEFFICIENT_C, COEFFICIENT_C)
