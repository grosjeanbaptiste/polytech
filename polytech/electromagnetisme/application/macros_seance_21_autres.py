"""Macros de la seance 2.1, exercices 3, 5 et 17.

Trois exercices courts, reunis ici parce qu'aucun ne remplit un module.

L'exercice 3 -- la nappe plane de spires -- publie le resultat du
manuscrit, la limite de la couronne mince et la moyenne de `1/r` que le
logarithme mesure. Prefixe `DeuxUnTrois`.

L'exercice 5 -- la spire tournante -- publie la f.e.m. par les **deux**
routes. Prefixe `DeuxUnCinq`.

L'exercice 17 -- le couple entre deux bobines -- est le seul numerique.
Il publie le couple juste **et** celui qu'on obtient en oubliant de
convertir la pente en radians : c'est le piege, et le mettre en regard
vaut mieux que de l'annoncer. Prefixe `DeuxUnDixSept`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    forme_logarithmique,
    nombre,
)
from polytech.electromagnetisme.application.macros_seance_21_spire import (
    macros_de_la_spire,
)
from polytech.electromagnetisme.domain.magnetostatique.couple_des_bobines import (
    couple_a_courants_constants,
    pente_par_radian,
)
from polytech.electromagnetisme.domain.magnetostatique.nappe_plane import (
    induction_d_une_spire_au_centre,
    induction_de_la_nappe_au_centre,
    moyenne_de_l_inverse_du_rayon,
)
from polytech.electromagnetisme.domain.magnetostatique.spire_tournante import (
    fem_par_le_flux,
    fem_par_les_cotes_actifs,
    flux_a_travers_la_spire,
)
from polytech.electromagnetisme.domain.symboles import PULSATION

PERMEABILITE = sp.Symbol("mu_0", positive=True)
COURANT = sp.Symbol("I", positive=True)
INTERIEUR = sp.Symbol("R_1", positive=True)
EXTERIEUR = sp.Symbol("R_2", positive=True)

# Donnees de l'exercice 17, lues dans la transcription.
PROPRE_1_EN_MICROHENRYS: Final = 60.0
PROPRE_2_EN_MICROHENRYS: Final = 240.0
PENTE_EN_MICROHENRYS_PAR_DEGRE: Final = -90.0
PREMIER_COURANT: Final = 1.0
SECOND_COURANT_EN_MILLIAMPERES: Final = 5.0


def _macros_de_la_nappe() -> dict[str, str]:
    """Le resultat du manuscrit, sa limite, et ce que le logarithme mesure."""
    nappe = induction_de_la_nappe_au_centre(COURANT, INTERIEUR, EXTERIEUR, PERMEABILITE)
    rayon = sp.Symbol("R", positive=True)
    return {
        "DeuxUnTroisSpire": expression_latex(
            induction_d_une_spire_au_centre(COURANT, rayon, PERMEABILITE),
        ),
        "DeuxUnTroisNappe": forme_logarithmique(nappe, EXTERIEUR / INTERIEUR),
        "DeuxUnTroisCouronneMince": expression_latex(
            sp.limit(nappe, EXTERIEUR, INTERIEUR),
        ),
        "DeuxUnTroisMoyenne": forme_logarithmique(
            moyenne_de_l_inverse_du_rayon(INTERIEUR, EXTERIEUR),
            EXTERIEUR / INTERIEUR,
        ),
    }


def _macros_de_la_spire_tournante() -> dict[str, str]:
    """Le flux, la f.e.m. par deux routes, et la vitesse d'un cote actif."""
    arguments = (
        sp.Symbol("B", positive=True),
        sp.Symbol("ell", positive=True),
        sp.Symbol("L", positive=True),
    )
    return {
        "DeuxUnCinqFlux": expression_latex(flux_a_travers_la_spire(*arguments)),
        "DeuxUnCinqFem": expression_latex(fem_par_le_flux(*arguments)),
        "DeuxUnCinqFemParLesCotes": expression_latex(
            fem_par_les_cotes_actifs(*arguments),
        ),
        "DeuxUnCinqVitesseDuCote": expression_latex(PULSATION * arguments[1] / 2),
    }


def _macros_du_couple() -> dict[str, str]:
    """Les donnees, la pente convertie, le couple, et le couple fautif."""
    courants = (PREMIER_COURANT, SECOND_COURANT_EN_MILLIAMPERES * 1e-3)
    par_radian = pente_par_radian(PENTE_EN_MICROHENRYS_PAR_DEGRE)
    juste = couple_a_courants_constants(*courants, par_radian)
    fautif = couple_a_courants_constants(*courants, PENTE_EN_MICROHENRYS_PAR_DEGRE)
    return {
        "DeuxUnDixSeptPropreUn": nombre(PROPRE_1_EN_MICROHENRYS, 0),
        "DeuxUnDixSeptPropreDeux": nombre(PROPRE_2_EN_MICROHENRYS, 0),
        "DeuxUnDixSeptPenteParDegre": nombre(PENTE_EN_MICROHENRYS_PAR_DEGRE, 0),
        "DeuxUnDixSeptPremierCourant": nombre(PREMIER_COURANT, 0),
        "DeuxUnDixSeptSecondCourant": nombre(SECOND_COURANT_EN_MILLIAMPERES, 0),
        "DeuxUnDixSeptPenteParRadian": nombre(par_radian, 0),
        "DeuxUnDixSeptFacteur": nombre(par_radian / PENTE_EN_MICROHENRYS_PAR_DEGRE, 1),
        "DeuxUnDixSeptCouple": nombre(juste, 1),
        "DeuxUnDixSeptCoupleFautif": nombre(fautif, 2),
    }


def macros_des_trois_autres() -> dict[str, str]:
    """La nappe, la spire tournante, et le couple entre deux bobines."""
    return {
        **_macros_de_la_nappe(),
        **_macros_de_la_spire_tournante(),
        **_macros_du_couple(),
    }


def macros_de_la_seance_deux_un() -> dict[str, str]:
    """Les quatre exercices de la seance 2.1, reunis pour son document."""
    return {**macros_de_la_spire(), **macros_des_trois_autres()}
