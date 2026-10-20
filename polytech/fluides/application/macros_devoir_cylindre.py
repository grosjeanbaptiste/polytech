"""Les macros des deux rapports du devoir cote : l'ecoulement autour d'un cylindre.

Aucun nombre des rapports n'est tape a la main. Les cas viennent de
`devoir_cylindre_cas`, les fluides et les regimes de
`devoir_cylindre_fluides`, chacun avec sa source.
"""

from __future__ import annotations

import math

from scipy.integrate import quad

from polytech.fluides.application import devoir_cylindre_fluides as dcf
from polytech.fluides.application.devoir_cylindre_cas import CAS, CasDEtude, cas
from polytech.fluides.application.formatage import nombre
from polytech.fluides.domain.couche_limite import epaisseur_de_couche_limite

_POURCENT = 100.0
_SEUIL_DE_MACH = 0.3
"""Le critere d'incompressibilite du cours (`regimes.MACH_INCOMPRESSIBLE`)."""


def _scientifique(valeur: float, chiffres: int = 2) -> str:
    """`chiffres` significatifs, en notation que siunitx lit : `6.6e3`."""
    mantisse, exposant = f"{valeur:.{chiffres - 1}e}".split("e")
    return f"{mantisse}e{int(exposant)}"


def _couche_limite(etude: CasDEtude) -> dict[str, str]:
    """`delta / D` au sommet, `x = pi R / 2` depuis l'arret, par Blasius.

    `x` est l'arc parcouru depuis le point d'arret amont. Passe
    `Re_x = 5e5`, Blasius ne decrit plus rien, et
    `epaisseur_de_couche_limite` leve : un cas turbulent au sommet ferait
    echouer la generation plutot que de publier une valeur fausse.
    """
    p = etude.point
    sommet = math.pi * p.diametre / 4
    nu = p.fluide.viscosite_cinematique
    delta = _POURCENT * epaisseur_de_couche_limite(p.vitesse, sommet, nu) / p.diametre
    return {f"{etude.cle}CoucheLimitePourcent": nombre(delta, 3)}


def _cas(etude: CasDEtude) -> dict[str, str]:
    p = etude.point
    macros = {
        f"{etude.cle}Reynolds": _scientifique(p.reynolds),
        f"{etude.cle}Regime": p.regime.nom,
        f"{etude.cle}VitesseMax": nombre(p.vitesse_maximale, 1),
        f"{etude.cle}EcartPression": nombre(p.ecart_de_pression_maximal, 1),
    }
    if p.mach is not None:
        macros[f"{etude.cle}Mach"] = nombre(p.mach, 4)
    return {**macros, **_couche_limite(etude)}


def celerite(fluide: dcf.Fluide) -> float:
    """La celerite du son publiee ; leve pour un fluide traite incompressible."""
    if fluide.celerite is None:
        msg = f"{fluide.nom} : aucune celerite, le fluide est traite incompressible"
        raise ValueError(msg)
    return fluide.celerite


def _fluides() -> dict[str, str]:
    froid, chaud = dcf.air_a(-10.0), dcf.air_a(25.0)
    ecart = abs(dcf.air_a(20.0).viscosite / dcf.AIR.viscosite - 1)
    return {
        "AirMasseVolumique": nombre(dcf.AIR.masse_volumique, 3),
        "AirViscosite": _scientifique(dcf.AIR.viscosite, 3),
        "AirViscositeCinematique": _scientifique(dcf.AIR.viscosite_cinematique, 3),
        "EcartSourcesAirPourcent": nombre(_POURCENT * ecart, 2),
        "AirFroidMasseVolumique": nombre(froid.masse_volumique, 3),
        "AirFroidViscosite": _scientifique(froid.viscosite, 3),
        "AirChaudViscosite": _scientifique(chaud.viscosite, 3),
        "EauViscosite": _scientifique(dcf.EAU.viscosite, 3),
        "EauViscositeCinematique": _scientifique(dcf.EAU.viscosite_cinematique, 3),
        "RapportViscositesCinematiques": nombre(
            dcf.AIR.viscosite_cinematique / dcf.EAU.viscosite_cinematique, 1
        ),
        "CeleriteAir": nombre(celerite(dcf.AIR), 0),
        "VitesseSeuilMach": nombre(_SEUIL_DE_MACH * celerite(dcf.AIR), 0),
        "ReynoldsTransition": _scientifique(dcf.REYNOLDS_DE_TRANSITION),
        "ReynoldsCrise": _scientifique(dcf.REYNOLDS_DE_CRISE),
    }


def trainee_par_quadrature() -> float:
    """`-int (1 - 4 sin^2) cos d theta` par quadrature : le second chemin."""
    valeur, _erreur = quad(
        lambda t: -(1 - 4 * math.sin(t) ** 2) * math.cos(t), 0, 2 * math.pi
    )
    return float(valeur)


def ecart_de_c_p_entre_les_cas(points: int = 181) -> float:
    """Le plus grand ecart de `C_p` entre les cas, sur la paroi."""
    angles = [math.pi * k / (points - 1) for k in range(points)]
    valeurs = [[c.point.coefficient_de_pression(t) for c in CAS] for t in angles]
    return max(max(v) - min(v) for v in valeurs)


def macros() -> dict[str, str]:
    """Toutes les macros du devoir cote."""
    four = cas("ChemineeFour").point
    return {
        **_fluides(),
        **{cle: v for etude in CAS for cle, v in _cas(etude).items()},
        "ChemineeFourPressionArret": nombre(four.pression_relative(math.pi), 1),
        "ChemineeFourDepressionSommet": nombre(four.pression_relative(math.pi / 2), 1),
        "RecoupementTrainee": _scientifique(abs(trainee_par_quadrature())),
        "CpEcartEntreLesCas": _scientifique(ecart_de_c_p_entre_les_cas()),
    }
