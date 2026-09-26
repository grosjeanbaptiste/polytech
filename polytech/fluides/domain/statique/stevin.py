"""Cours II.2 -- Loi de Stevin.

Dans un fluide au repos, de masse volumique uniforme, la pression varie avec
la seule difference d'altitude : ``Delta p = rho g h``. La forme du recipient
n'intervient pas.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.grandeurs import PESANTEUR, Ecart, Hauteur, MasseVolumique
from polytech.fluides.domain.statique.symboles import ALTITUDE, PROFONDEUR, RHO, G


def variation_de_pression(
    masse_volumique: MasseVolumique,
    hauteur: Hauteur,
    pesanteur: float = PESANTEUR,
) -> Ecart:
    """Poids d'une colonne de fluide de hauteur ``h`` (cours II.2).

    Le resultat est toujours positif : c'est l'appelant qui decide, par le
    sens de parcours, s'il l'ajoute ou le retranche.
    """
    if pesanteur <= 0:
        msg = f"pesanteur non strictement positive : {pesanteur}"
        raise ValueError(msg)
    return Ecart(pascals=masse_volumique.kg_par_m3 * pesanteur * hauteur.metres)


def loi_de_stevin() -> sp.Eq:
    """Enonce du cours II.2 : ``p + rho g z = constante``."""
    return sp.Eq(sp.Symbol("p") + RHO * G * ALTITUDE, sp.Symbol("C"))


def relation_entre_deux_points() -> sp.Eq:
    """Forme pratique : ``p_bas = p_haut + rho g h``, utilisee pour la marche."""
    return sp.Eq(
        sp.Symbol("p_bas"),
        sp.Symbol("p_haut") + RHO * G * PROFONDEUR,
    )


def expression_variation_de_pression() -> sp.Eq:
    """Forme litterale du poids d'une colonne : ``Delta p = rho g h``."""
    return sp.Eq(
        sp.Symbol(r"\Delta p"),
        RHO * G * PROFONDEUR,
        evaluate=False,
    )
