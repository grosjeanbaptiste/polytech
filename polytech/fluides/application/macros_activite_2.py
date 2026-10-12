"""Les macros que le corrige de l'activite d'integration 2 consomme.

Aucun nombre du corrige n'est tape a la main : chacun sort d'ici, calcule
par `activite_2` sur les donnees de l'enonce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.fluides.application import activite_2 as a2
from polytech.fluides.application.formatage import expression_latex, nombre
from polytech.fluides.domain.fondamentales.film_tombant import (
    debit_du_film,
    vitesse_du_film,
)
from polytech.fluides.domain.grandeurs import PESANTEUR

if TYPE_CHECKING:
    from collections.abc import Iterable

_POURCENT = 100.0
_FRACTIONS_DU_NIVEAU = (0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 0.95, 0.99)
"""Les niveaux, en fraction de `z_max`, ou la courbe de remplissage est calculee."""
_KILO = 1000.0


def _film() -> dict[str, str]:
    profil = vitesse_du_film(a2.X, a2.H, a2.RHO, a2.MU, a2.G)
    return {
        "FilmProfil": expression_latex(sp.factor(profil)),
        "FilmProfilIntegre": expression_latex(a2.profil_par_integration()),
        "FilmProfilNormalise": expression_latex(a2.profil_normalise()),
        "FilmVitesseSurface": expression_latex(sp.simplify(profil.subs(a2.X, a2.H))),
        "FilmVitesseSurfaceNormalisee": expression_latex(
            a2.profil_normalise().subs(a2.ABSCISSE_REDUITE, 1)
        ),
        "FilmRapportMoyenne": expression_latex(a2.rapport_moyenne_sur_surface()),
        "FilmDebit": expression_latex(debit_du_film(a2.H, a2.RHO, a2.MU, a2.G)),
        "FilmCisaillementParoi": expression_latex(a2.cisaillement_a_la_paroi()),
        "FilmCourbe": _points(
            (k / 10, float(a2.profil_normalise().subs(a2.ABSCISSE_REDUITE, k / 10)))
            for k in range(11)
        ),
    }


def _reservoir() -> dict[str, str]:
    debit, diametre = sp.symbols("mdot D_o", positive=True)
    hauteur = sp.simplify(a2.hauteur_maximale(debit, sp.pi * diametre**2 / 4))
    gain = hauteur.subs(diametre, diametre / 2) / hauteur
    return {
        "ReservoirHauteurMaximale": expression_latex(hauteur),
        "ReservoirGainOrificeMoitie": nombre(float(gain), 0),
        "ReservoirDureeQuatreVingtDix": nombre(float(a2.duree_reduite(0.90)), 2),
        "ReservoirDureeQuatreVingtDixNeuf": nombre(float(a2.duree_reduite(0.99)), 1),
        "ReservoirCourbe": _points(
            (float(a2.duree_reduite(f)), f) for f in _FRACTIONS_DU_NIVEAU
        ),
    }


def _points(couples: Iterable[tuple[float, float]]) -> str:
    """Des coordonnees que `pgfplots` trace telles quelles."""
    return " ".join(f"({nombre(x, 4)},{nombre(y, 4)})" for x, y in couples)


def _bateau() -> dict[str, str]:
    bateau = a2.bateau_pompe()
    cinetique = bateau.ejection**2 / (2 * PESANTEUR)
    massique = PESANTEUR * (a2.HAUTEUR_DE_LA_BUSE + a2.PERTES) + bateau.ejection**2 / 2
    return {
        "BateauVitesseEjection": nombre(bateau.ejection, 1),
        "BateauVitesseAspiration": nombre(bateau.aspiration, 2),
        "BateauHauteurCinetique": nombre(cinetique, 1),
        "BateauHauteurPompe": nombre(bateau.hauteur, 1),
        "BateauPartCinetique": nombre(_POURCENT * cinetique / bateau.hauteur, 0),
        "BateauPuissanceHydraulique": nombre(bateau.hydraulique / _KILO, 1),
        "BateauPuissanceArbre": nombre(bateau.arbre / _KILO, 1),
        "BateauPuissancePerdue": nombre((bateau.arbre - bateau.hydraulique) / _KILO, 1),
        "BateauEnergieMassique": nombre(massique, 0),
        "BateauPuissanceParEnergie": nombre(
            a2.EAU_DE_MER * a2.DEBIT * massique / _KILO, 1
        ),
        "BateauDebitMassique": nombre(a2.EAU_DE_MER * a2.DEBIT, 0),
        "BateauCinetiqueAspiration": nombre(bateau.aspiration**2 / (2 * PESANTEUR), 2),
    }


def macros() -> dict[str, str]:
    """Toutes les macros de l'activite, exercice par exercice."""
    return {**_film(), **_reservoir(), **_bateau()}
