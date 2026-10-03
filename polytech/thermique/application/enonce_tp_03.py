"""Le capteur solaire a air du TP 3 de thermique.

Source : `ThermiqueFondements/TP/TP_03_Capteur_solaire_Enonce.pdf`.

Trois noeuds : la vitre, l'absorbeur, la face interne de l'isolant. Le
soleil, transmis par la vitre, est absorbe par l'absorbeur ; il en
repart vers l'air du canal (convection forcee, sur ses deux faces), vers
la vitre (rayonnement et convection naturelle dans la lame d'air) et vers
l'isolant (rayonnement a travers le canal). La vitre et l'isolant perdent
vers l'exterieur par leur conductance globale.

Le modele est celui de l'officiel : les reflexions multiples entre vitre
et absorbeur et l'absorption solaire de la vitre sont negligees. Les
coefficients dependent des temperatures ; les trois bilans sont resolus
ensemble, sans approximations successives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from scipy.optimize import fsolve

from polytech.thermique.domain.convection.canal_plan import coefficient_du_canal
from polytech.thermique.domain.convection.naturelle import (
    grashof,
    nusselt_lame_de_fluide,
)
from polytech.thermique.domain.rayonnement.vitre_selective import (
    VitreSelective,
    coefficient_entre_plans_gris,
)

KELVIN: Final = 273.15
AIR, AMBIANT, SOLEIL, TEMPERATURE_SOLAIRE = 70 + KELVIN, 20 + KELVIN, 700.0, 6000.0
DEBIT, CHALEUR_MASSIQUE = 0.056, 1008.0
VISCOSITE, MASSE_VOLUMIQUE, CONDUCTIVITE_AIR, PRANDTL = 2.036e-5, 1.027, 0.0295, 0.696
LAME, CANAL, LARGEUR = 0.02, 0.01, 1.0
VITRE: Final = VitreSelective(
    transmission_courte=0.917, reflexion=0.083, coupure=3.5e-6
)
ABSORPTION_SOLAIRE, EMISSIVITE_AVANT, EMISSIVITE_ARRIERE, EMISSIVITE_ISOLANT = (
    0.95,
    0.1,
    0.95,
    0.95,
)
EXTERIEUR: Final = 23.0
"""Vent et rayonnement vers l'exterieur, reunis par l'enonce."""
RESISTANCE_VITRE, RESISTANCE_ISOLANT = 0.004 / 1.0, 0.05 / 0.045


def conductances_exterieures() -> tuple[float, float]:
    """Vers l'exterieur : a travers la vitre, et a travers l'isolant."""
    return (
        1 / (1 / EXTERIEUR + RESISTANCE_VITRE),
        1 / (1 / EXTERIEUR + RESISTANCE_ISOLANT),
    )


def coefficients(
    vitre: float, absorbeur: float, isolant: float
) -> tuple[float, float, float]:
    """`K_ray` avant, `K_ray` arriere et `K_cn` de la lame, a ces temperatures."""
    avant = coefficient_entre_plans_gris(
        EMISSIVITE_AVANT, VITRE.absorption(vitre), absorbeur, vitre
    )
    arriere = coefficient_entre_plans_gris(
        EMISSIVITE_ARRIERE, EMISSIVITE_ISOLANT, absorbeur, isolant
    )
    gr = grashof(1 / AIR, absorbeur - vitre, LAME, VISCOSITE / MASSE_VOLUMIQUE)
    lame = nusselt_lame_de_fluide(gr * PRANDTL) * CONDUCTIVITE_AIR / LAME
    return avant, arriere, lame


@dataclass(frozen=True, slots=True)
class EtatDuCapteur:
    """Les trois temperatures, et les flux qui en decoulent, par m2."""

    vitre: float
    absorbeur: float
    isolant: float
    absorbe: float
    flux_vers_l_air: float
    perte_avant: float
    perte_arriere: float

    @property
    def echauffement(self) -> float:
        """`dT/dx = phi / (q c)`, par metre de longueur de capteur."""
        return self.flux_vers_l_air * LARGEUR / (DEBIT * CHALEUR_MASSIQUE)


def convection_du_canal() -> float:
    """Le coefficient de convection forcee, le meme sur les deux faces."""
    return coefficient_du_canal(DEBIT, LARGEUR, CANAL, VISCOSITE, CONDUCTIVITE_AIR)


def absorbe() -> float:
    """`alpha_a tau_v(6000 K) phi_sol` : ce que l'absorbeur recoit du soleil."""
    return ABSORPTION_SOLAIRE * VITRE.transmission(TEMPERATURE_SOLAIRE) * SOLEIL


def _bilans(inconnues: list[float]) -> list[float]:
    vitre, absorbeur, isolant = inconnues
    avant, arriere, lame = coefficients(vitre, absorbeur, isolant)
    par_vitre, par_isolant = conductances_exterieures()
    canal = convection_du_canal()
    return [
        (avant + lame) * (absorbeur - vitre) - par_vitre * (vitre - AMBIANT),
        absorbe()
        - (avant + lame) * (absorbeur - vitre)
        - canal * (absorbeur - AIR)
        - arriere * (absorbeur - isolant),
        canal * (AIR - isolant)
        + arriere * (absorbeur - isolant)
        - par_isolant * (isolant - AMBIANT),
    ]


def resoudre_le_capteur() -> EtatDuCapteur:
    """Les trois bilans, resolus ensemble ; leve si la resolution echoue."""
    solution, _, statut, message = fsolve(
        _bilans, [300.0, 350.0, 300.0], xtol=1e-12, full_output=True
    )
    if statut != 1:
        msg = f"le capteur n'a pas converge : {message}"
        raise ValueError(msg)
    vitre, absorbeur, isolant = (float(t) for t in solution)
    par_vitre, par_isolant = conductances_exterieures()
    canal = convection_du_canal()
    return EtatDuCapteur(
        vitre=vitre,
        absorbeur=absorbeur,
        isolant=isolant,
        absorbe=absorbe(),
        flux_vers_l_air=canal * (absorbeur - AIR) + canal * (isolant - AIR),
        perte_avant=par_vitre * (vitre - AMBIANT),
        perte_arriere=par_isolant * (isolant - AMBIANT),
    )
