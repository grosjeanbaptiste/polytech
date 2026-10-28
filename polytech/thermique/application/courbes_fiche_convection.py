"""Les courbes de convection et d'echangeurs de la fiche, calculees par la bibliotheque."""

from __future__ import annotations

from typing import Final

from polytech.thermique.application.courbes_fiche_conduction import Points
from polytech.thermique.domain.convection.couches_limites import angle_de_decollement
from polytech.thermique.domain.convection.forcee import nusselt_plaque
from polytech.thermique.domain.convection.naturelle import (
    grashof,
    nusselt_lame_de_fluide,
    rayleigh,
)
from polytech.thermique.domain.convection.plaque_plane import nusselt_local_laminaire
from polytech.thermique.domain.echangeurs_a_courants_paralleles import Sens
from polytech.thermique.domain.profils_d_echangeur import profils_cocourants
from polytech.thermique.domain.rendement_d_echangeur import rendement

REYNOLDS_PLAQUE: Final = 1e5
PRANDTL: Final = 0.71
ENTREES: Final = (90.0, 20.0)
UNITES: Final = 1.0
"""`N = K S / C_1` de l'exemple du DTLM, les deux debits de capacite egaux."""
LAME: Final = (1 / 283.0, 20.0, 1.236e-5)
"""`(beta, ecart, nu)` de la lame d'air du syllabus (tome 2, 1.7)."""
CONDUCTIVITE_DE_L_AIR: Final = 0.023


def coefficient_local_de_la_plaque() -> Points:
    """`h_x / h_x(0,02 L)` le long de la plaque : `Nu_x ~ Re_x^(1/2)`, `Re_x ~ x`."""
    depart = nusselt_local_laminaire(REYNOLDS_PLAQUE * 0.02, PRANDTL) / 0.02
    return [
        (x, nusselt_local_laminaire(REYNOLDS_PLAQUE * x, PRANDTL) / x / depart)
        for x in (0.02 + 0.02 * i for i in range(50))
    ]


def rapport_moyen_sur_local() -> float:
    """`h_moyen / h_local(L)` : le 2 de la fiche, des deux correlations."""
    return nusselt_plaque(REYNOLDS_PLAQUE, PRANDTL) / nusselt_local_laminaire(
        REYNOLDS_PLAQUE, PRANDTL
    )


def angle_de_separation() -> float:
    """Le decollement du cylindre, en degres, pour `Re = 1e5`."""
    return angle_de_decollement(1e5)


def _epaisseur_au_rayleigh(valeur: float) -> float:
    beta, ecart, nu = LAME
    unitaire = rayleigh(grashof(beta, ecart, 1.0, nu), PRANDTL)
    return float((valeur / unitaire) ** (1 / 3))


def resistance_de_la_lame() -> Points:
    """`b / (lambda Nu)` reduite, `b` en mm : les points ou la correlation existe.

    Entre `Ra = 1400` et `Ra = 6000` le formulaire ne correle pas : on relie
    les deux bornes, qui sont des points de la correlation.
    """
    beta, ecart, nu = LAME
    bornes = [_epaisseur_au_rayleigh(1400.0), _epaisseur_au_rayleigh(6000.0)]
    epaisseurs = [b / 1000 for b in range(1, 41) if not bornes[0] < b / 1000 < bornes[1]]
    epaisseurs = sorted({*epaisseurs, *bornes})
    brut = [
        (b, b / (CONDUCTIVITE_DE_L_AIR * nusselt_lame_de_fluide(
            rayleigh(grashof(beta, ecart, b, nu), PRANDTL)
        )))
        for b in epaisseurs
    ]
    maximum = max(r for _, r in brut)
    return [(b * 1000, r / maximum) for b, r in brut]


def profil_chaud() -> Points:
    """La temperature du fluide chaud, de l'entree a la sortie."""
    return [(x / 20, _profils(x / 20)[0]) for x in range(21)]


def profil_froid() -> Points:
    """La temperature du fluide froid."""
    return [(x / 20, _profils(x / 20)[1]) for x in range(21)]


def _profils(position: float) -> tuple[float, float]:
    return profils_cocourants(*ENTREES, unites=UNITES, rapport=1.0, position=position)


def ecart_en(position: float) -> tuple[float, float]:
    """`(t_froid, t_chaud)` a une position : les deux bouts d'une fleche d'ecart."""
    chaud, froid = _profils(position)
    return froid, chaud


def efficacite_a_paroi_isotherme() -> Points:
    """`1 - e^(-NUT)` : `R = 0`, quel que soit le sens."""
    return _efficacite(Sens.CONTRECOURANT, 0.0)


def efficacite_a_contre_courant() -> Points:
    """`NUT / (1 + NUT)` : `R = 1`, contre-courant."""
    return _efficacite(Sens.CONTRECOURANT, 1.0)


def efficacite_a_courants_paralleles() -> Points:
    """`(1 - e^(-2 NUT)) / 2` : `R = 1`, courants paralleles."""
    return _efficacite(Sens.COCOURANT, 1.0)


def _efficacite(sens: Sens, rapport: float) -> Points:
    return [(n / 20, rendement(sens, n / 20, rapport)) for n in range(1, 101)]
