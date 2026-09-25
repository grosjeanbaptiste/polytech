"""Macros des donnees de l'enonce.

L'enonce n'est transcrit qu'une fois, dans :mod:`fluides.application.enonce` ;
le document le cite par ces macros. Corriger une donnee a un seul endroit
suffit donc a corriger tout le corrige.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.application.formatage import expression_latex, nombre
from polytech.fluides.domain.grandeurs import PESANTEUR
from polytech.fluides.domain.statique.parements import ParementVertical

_MOITIE = sp.Rational(1, 2)


def _macros_fluides() -> dict[str, str]:
    return {
        "MasseVolumiqueEau": nombre(enonce.EAU.kg_par_m3, 0),
        "MasseVolumiqueHuile": nombre(enonce.HUILE.kg_par_m3, 0),
        "MasseVolumiqueMercure": nombre(enonce.MERCURE.kg_par_m3, 0),
        "DensiteHuile": nombre(enonce.HUILE.densite_relative(enonce.EAU), 2),
        "DensiteMercure": nombre(enonce.MERCURE.densite_relative(enonce.EAU), 1),
        "Pesanteur": nombre(PESANTEUR, 2),
    }


def _macros_hauteurs() -> dict[str, str]:
    noms = ("HauteurUn", "HauteurDeux", "HauteurTrois")
    hauteurs = (t.hauteur.metres for t in enonce.MANOMETRE.troncons)
    return {nom: f"{h:.2f}" for nom, h in zip(noms, hauteurs, strict=True)}


def macros_donnees() -> dict[str, str]:
    """Rend les donnees de l'enonce, telles que le domaine les porte."""
    mur = ParementVertical(fluide=enonce.EAU, largeur=1.0)
    return {
        **_macros_fluides(),
        **_macros_hauteurs(),
        "ChampBU": expression_latex(enonce.CHAMP_B.u),
        "ChampBV": expression_latex(enonce.CHAMP_B.v),
        "ChampCU": expression_latex(enonce.CHAMP_C.u),
        "ChampCV": expression_latex(enonce.CHAMP_C.v),
        "RapportHauteurPrismatique": expression_latex(
            mur.hauteur_pour_fraction_de_volume(_MOITIE),
        ),
        "PenteLignesCourantC": expression_latex(
            enonce.CHAMP_C.pente_lignes_de_courant(),
        ),
    }
