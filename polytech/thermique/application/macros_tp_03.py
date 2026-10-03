"""Macros du TP 3 de thermique : le capteur solaire a air.

La vitre a la temperature de deux sources, les coefficients a
convergence et a la premiere iteration de l'officiel, les trois
temperatures, et le bilan global -- ou l'on voit, en plus de ce que
l'officiel demande, ou part le reste du soleil et le rendement qui en
resulte.

Le prefixe est `Capteur`.
"""

from __future__ import annotations

from polytech.thermique.application import enonce_tp_03 as e
from polytech.thermique.application.formatage import nombre
from polytech.thermique.domain.convection.canal_plan import reynolds_du_canal
from polytech.thermique.domain.convection.naturelle import grashof
from polytech.thermique.domain.rayonnement.corps_noir import (
    fraction_emise_sous,
    longueur_d_onde_du_maximum,
)


def _macros_de_la_vitre() -> dict[str, str]:
    return {
        "CapteurMaximumSoleil": nombre(longueur_d_onde_du_maximum(6000.0) * 1e6, 2),
        "CapteurMaximumCapteur": nombre(longueur_d_onde_du_maximum(300.0) * 1e6, 1),
        "CapteurFraction": nombre(fraction_emise_sous(3.5e-6, 6000.0), 4),
        "CapteurTransmission": nombre(e.VITRE.transmission(6000.0), 4),
        "CapteurAbsorptionSoleil": nombre(e.VITRE.absorption(6000.0), 4),
        "CapteurEmissiviteVitre": nombre(e.VITRE.absorption(300.0), 3),
        "CapteurAbsorbe": nombre(e.absorbe(), 1),
    }


def _macros_des_coefficients(etat: e.EtatDuCapteur) -> dict[str, str]:
    avant, arriere, lame = e.coefficients(etat.vitre, etat.absorbeur, etat.isolant)
    avant_0, arriere_0, lame_0 = e.coefficients(300.0, 350.0, 300.0)
    par_vitre, par_isolant = e.conductances_exterieures()
    gr = grashof(
        1 / e.AIR, etat.absorbeur - etat.vitre, e.LAME, e.VISCOSITE / e.MASSE_VOLUMIQUE
    )
    return {
        "CapteurReynolds": nombre(
            reynolds_du_canal(e.DEBIT, e.LARGEUR, e.CANAL, e.VISCOSITE), 0
        ),
        "CapteurConvection": nombre(e.convection_du_canal(), 2),
        "CapteurGrashof": nombre(gr, 0),
        "CapteurLame": nombre(lame, 2),
        "CapteurRayAvant": nombre(avant, 2),
        "CapteurRayArriere": nombre(arriere, 2),
        "CapteurLameInitiale": nombre(lame_0, 2),
        "CapteurRayAvantInitial": nombre(avant_0, 2),
        "CapteurRayArriereInitial": nombre(arriere_0, 2),
        "CapteurParVitre": nombre(par_vitre, 2),
        "CapteurParIsolant": nombre(par_isolant, 3),
    }


def macros_du_tp_trois() -> dict[str, str]:
    """La vitre, les coefficients, les temperatures, les flux."""
    etat = e.resoudre_le_capteur()
    return {
        **_macros_de_la_vitre(),
        **_macros_des_coefficients(etat),
        "CapteurVitre": nombre(etat.vitre - e.KELVIN, 2),
        "CapteurAbsorbeur": nombre(etat.absorbeur - e.KELVIN, 2),
        "CapteurIsolant": nombre(etat.isolant - e.KELVIN, 2),
        "CapteurFlux": nombre(etat.flux_vers_l_air, 1),
        "CapteurPerteAvant": nombre(etat.perte_avant, 1),
        "CapteurPerteArriere": nombre(etat.perte_arriere, 1),
        "CapteurRendement": nombre(etat.flux_vers_l_air / e.SOLEIL * 100, 1),
        "CapteurEchauffement": nombre(etat.echauffement, 2),
    }
