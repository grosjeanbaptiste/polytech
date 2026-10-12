"""L'examen de mecanique des fluides de janvier 2025, refait par la bibliotheque.

La fiche de revision s'appuie sur ces questions : ce qu'elle en dit doit
tenir. L'examen (`MecaniqueFluides/EXAMENS/Exam_2025_Janvier.pdf`) n'a pas
de corrige au depot : chaque reponse est donc recoupee par un second
chemin, et arrondie a deux chiffres significatifs comme la consigne le
demande.

La derniere question -- la puissance de la pompe -- s'arrete au
coefficient `lambda`. La relation qui en tire la perte de charge
appartient au chapitre V du cours, « Ecoulements en conduite », dont le
depot ne porte aucun fascicule : la finir serait employer une formule
qu'aucun document d'ici n'etablit.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.fluides.domain.aerodynamique import Ecoulement, SurfacePortante
from polytech.fluides.domain.fondamentales.bernoulli import MASSE_VOLUMIQUE, charge
from polytech.fluides.domain.grandeurs import PESANTEUR, MasseVolumique
from polytech.fluides.domain.regime_de_reynolds import (
    RegimeDeReynolds,
    nombre_de_reynolds,
    regime_en_conduite,
)
from polytech.fluides.domain.trainee import vitesse_de_chute

# --- Question 3 : la maquette d'avion au dixieme ------------------------------

VISCOSITE_DE_L_AIR = 1.8e-5 / 0.9
"""`nu = mu / rho`, avec les valeurs de l'enonce, en m^2/s."""


def test_q3_la_maquette_n_est_pas_en_similitude_de_reynolds() -> None:
    """`Re` de l'avion vaut 83 fois celui de la maquette : 250 x 10 / 30."""
    avion = nombre_de_reynolds(250.0, 1.0, VISCOSITE_DE_L_AIR)
    maquette = nombre_de_reynolds(30.0, 0.1, VISCOSITE_DE_L_AIR)
    assert math.isclose(avion / maquette, 250 * 10 / 30)


def test_q3_la_similitude_demanderait_dix_fois_la_vitesse_de_l_avion() -> None:
    """Meme air, echelle 1/10 : `v_m = 10 v`, soit 2500 m/s -- hors d'atteinte."""
    avion = nombre_de_reynolds(250.0, 1.0, VISCOSITE_DE_L_AIR)
    maquette = nombre_de_reynolds(2500.0, 0.1, VISCOSITE_DE_L_AIR)
    assert math.isclose(avion, maquette)


# --- Question 5.1 : le parapente ----------------------------------------------

VOILE = SurfacePortante(Ecoulement(MasseVolumique(1.225), 12.0), 30.0)
PORTANCE = VOILE.force(1.4)


def test_q51_la_portance_vaut_trois_mille_sept_cents_newtons() -> None:
    """`F_L = C_L S rho v^2 / 2 = 1,4 x 30 x 1,225 x 144 / 2`."""
    assert math.isclose(PORTANCE, 1.4 * 30 * 1.225 * 144 / 2)
    assert f"{PORTANCE:.2g}" == "3.7e+03"


def test_q51_la_masse_maximale_vaut_trois_cent_quatre_vingts_kilogrammes() -> None:
    """La portance equilibre le poids : `m = F_L / g`."""
    assert f"{PORTANCE / PESANTEUR:.2g}" == "3.8e+02"


def test_q51_a_cette_masse_l_equilibre_se_fait_a_douze_metres_par_seconde() -> None:
    """Le second chemin : la vitesse a laquelle l'effort porte le poids."""
    vitesse = vitesse_de_chute(
        masse=PORTANCE / PESANTEUR, masse_volumique=1.225, coefficient=1.4, surface=30.0
    )
    assert math.isclose(vitesse, 12.0)


# --- Question 5.2 : le toboggan aquatique -------------------------------------

DEBIT = 10.6e-3
DIAMETRE = 0.15
VITESSE = DEBIT / (math.pi * DIAMETRE**2 / 4)
ATMOSPHERE = 1.0e5
EAU = 1000.0


def test_q52_la_vitesse_en_conduite_vaut_soixante_centimetres_par_seconde() -> None:
    """`v = Q / S`, et le debit que cette vitesse porte est bien celui donne."""
    assert f"{VITESSE:.2g}" == "0.6"
    assert math.isclose(VITESSE * math.pi * DIAMETRE**2 / 4, DEBIT)


def test_q52_la_pression_en_a_vaut_un_bar_et_un_dixieme() -> None:
    """Bernoulli de la surface libre (`z = 1,5 m`, au repos) au point A (`z = 0`)."""
    inconnue = sp.Symbol("p_A", positive=True)
    egalite = sp.Eq(charge(ATMOSPHERE, 1.5, 0), charge(inconnue, 0, VITESSE))
    (pression,) = sp.solve(egalite.subs(MASSE_VOLUMIQUE, EAU), inconnue)
    attendue = ATMOSPHERE + EAU * PESANTEUR * 1.5 - EAU * VITESSE**2 / 2
    assert math.isclose(float(pression), attendue)
    assert f"{float(pression) / 1e5:.2g}" == "1.1"


def test_q52_l_ecoulement_est_turbulent_et_lambda_vaut_dix_huit_milliemes() -> None:
    """`Re = rho v d / mu = 9,0 x 10^4`, puis `lambda = 0,316 Re^(-1/4)`."""
    reynolds = nombre_de_reynolds(VITESSE, DIAMETRE, 1e-3 / EAU)
    assert f"{reynolds:.2g}" == "9e+04"
    assert regime_en_conduite(reynolds) is RegimeDeReynolds.TURBULENT
    assert f"{0.316 * reynolds**-0.25:.2g}" == "0.018"
