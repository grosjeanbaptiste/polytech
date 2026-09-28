"""Chaine d'amplification : combien d'etages, et quels gains.

Un amplificateur operationnel a produit gain-bande constant : plus on lui
demande de gain, moins il garde de bande. Viser un gain global sur une
bande donnee impose donc un nombre minimal d'etages, chacun restant sous le
gain que la bande autorise.
"""

from __future__ import annotations

import math


def gain_maximal_par_etage(
    produit_gain_bande: float,
    bande_passante: float,
) -> float:
    """Gain qu'un etage peut tenir sur cette bande, sans la retrecir."""
    if produit_gain_bande <= 0 or bande_passante <= 0:
        msg = (
            "produit gain-bande et bande passante doivent etre positifs : "
            f"{produit_gain_bande}, {bande_passante}"
        )
        raise ValueError(msg)
    return produit_gain_bande / bande_passante


def nombre_d_etages(
    gain_vise: float,
    produit_gain_bande: float,
    bande_passante: float,
) -> int:
    """Nombre minimal d'etages identiques pour atteindre ce gain.

    Seule la valeur absolue du gain compte : un signe negatif s'obtient en
    choisissant un montage inverseur, sans cout en bande passante.
    """
    par_etage = gain_maximal_par_etage(produit_gain_bande, bande_passante)
    if par_etage <= 1:
        msg = f"un etage n'amplifie pas a cette bande : gain {par_etage}"
        raise ValueError(msg)
    return math.ceil(math.log(abs(gain_vise), par_etage))


def gain_non_inverseur(resistance_masse: float, contre_reaction: float) -> float:
    """Gain d'un montage non-inverseur : `1 + R_f / R_g`.

    Son impedance d'entree est celle de l'amplificateur lui-meme, tres
    elevee : c'est pourquoi le corrige le place en premier.
    """
    if resistance_masse <= 0 or contre_reaction <= 0:
        msg = f"resistances non positives : {resistance_masse}, {contre_reaction}"
        raise ValueError(msg)
    return 1 + contre_reaction / resistance_masse


def gain_inverseur(resistance_entree: float, contre_reaction: float) -> float:
    """Gain d'un montage inverseur : `-R_f / R_e`, donc negatif."""
    if resistance_entree <= 0 or contre_reaction <= 0:
        msg = f"resistances non positives : {resistance_entree}, {contre_reaction}"
        raise ValueError(msg)
    return -contre_reaction / resistance_entree
