"""Similitude, Bernoulli, portance et Mach : la fiche de fluides, refaite.

Les valeurs chiffrees de janvier 2025 que la fiche cite (83, 2500 m/s,
1,1 bar, 3,7e3 N, 3,8e2 kg, 4,4 %, 100 m/s) et les formules de ses
encadres sont recalculees par `polytech.fluides`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.fluides.domain.aerodynamique import Ecoulement, SurfacePortante
from polytech.fluides.domain.compressibilite import variation_relative_a_l_arret
from polytech.fluides.domain.fondamentales.bernoulli import MASSE_VOLUMIQUE, charge
from polytech.fluides.domain.fondamentales.bernoulli_generalise import (
    hauteur_manometrique,
    puissance_hydraulique,
)
from polytech.fluides.domain.fondamentales.continuite import vitesse_dans_la_section
from polytech.fluides.domain.grandeurs import PESANTEUR, MasseVolumique
from polytech.fluides.domain.regime_de_reynolds import nombre_de_reynolds
from polytech.fluides.domain.regimes import (
    GAMMA_DE_L_AIR,
    MACH_INCOMPRESSIBLE,
    VARIATION_ADMISE,
    est_incompressible,
    nombre_de_mach,
)
from polytech.fluides.domain.similitude import (
    nombre_de_froude,
    rapport_des_viscosites_en_similitude_complete,
)
from polytech.fluides.domain.statique.gaz import AIR, CONSTANTE_DES_GAZ
from polytech.fluides.domain.trainee import vitesse_de_chute
from polytech.maths.domain.comparaison import (
    egales,
    egales_arrondies,
    proches,
    proches_relativement,
)

NU = 1.8e-5 / 0.9


def test_la_similitude_complete_impose_le_produit_des_rapports_egal_a_un() -> None:
    echelle, rapport_g, longueur, vitesse, nu_1 = 0.1, 1.0, 3.0, 7.0, 1.5e-5
    nu_2 = nu_1 * rapport_des_viscosites_en_similitude_complete(
        echelle=echelle, rapport_des_pesanteurs=rapport_g
    )
    vitesse_2 = vitesse * math.sqrt(echelle * rapport_g)
    assert proches(
        nombre_de_froude(vitesse_2, echelle * longueur, PESANTEUR * rapport_g),
        nombre_de_froude(vitesse, longueur),
        1e-12,
    )
    assert proches_relativement(
        nombre_de_reynolds(vitesse_2, echelle * longueur, nu_2),
        nombre_de_reynolds(vitesse, longueur, nu_1),
        1e-12,
    )
    assert proches((nu_1 / nu_2) ** 2 * echelle**3 * rapport_g, 1.0, 1e-12)


def test_janvier_2025_3_la_maquette_au_dixieme_ne_respecte_ni_re_ni_mach() -> None:
    avion = nombre_de_reynolds(250.0, 1.0, NU)
    maquette = nombre_de_reynolds(30.0, 0.1, NU)
    assert egales_arrondies(avion / maquette, 83.0, 0)
    assert proches_relativement(nombre_de_reynolds(2500.0, 0.1, NU), avion, 1e-12)
    celerite = 340.0
    assert nombre_de_mach(250.0, celerite) != nombre_de_mach(30.0, celerite)
    assert nombre_de_mach(250.0, celerite) == nombre_de_mach(250.0, celerite)


def test_janvier_2025_5_2_bernoulli_de_la_surface_au_point_a_donne_1_1_bar() -> None:
    vitesse = vitesse_dans_la_section(10.6e-3, math.pi * 0.15**2 / 4)
    assert egales_arrondies(vitesse, 0.60, 2)
    pression = sp.Symbol("p_A", positive=True)
    egalite = sp.Eq(charge(1.0e5, 1.5, 0), charge(pression, 0, vitesse))
    (valeur,) = sp.solve(egalite.subs(MASSE_VOLUMIQUE, 1000.0), pression)
    assert egales_arrondies(float(valeur) / 1e5, 1.1, 1)


def test_bernoulli_generalise_est_la_charge_conservee_avec_pompe_et_pertes() -> None:
    p1, v1, z1, p2, v2, z2, perte = 2.0e5, 1.0, 0.5, 1.5e5, 4.0, 3.0, 2.0
    rho_g = 1000.0 * PESANTEUR
    pompe = hauteur_manometrique(
        denivele=z2 - z1, vitesse_finale=v2, vitesse_initiale=v1,
        hauteur_de_pression=(p2 - p1) / rho_g, pertes=perte,
    )  # fmt: skip
    gauche = p1 / rho_g + v1**2 / (2 * PESANTEUR) + z1 + pompe
    droite = p2 / rho_g + v2**2 / (2 * PESANTEUR) + z2 + perte
    assert proches(gauche, droite, 1e-9)
    assert proches_relativement(
        puissance_hydraulique(1000.0, 0.01, pompe), rho_g * 0.01 * pompe, 1e-12
    )


def test_janvier_2025_5_1_la_portance_du_parapente_et_sa_masse_maximale() -> None:
    voile = SurfacePortante(Ecoulement(MasseVolumique(1.225), 12.0), 30.0)
    portance = voile.force(1.4)
    assert proches_relativement(portance, 1.4 * 30 * 0.5 * 1.225 * 12.0**2, 1e-12)
    assert proches_relativement(portance, 3.7e3, 0.01)
    masse = portance / PESANTEUR
    assert proches_relativement(masse, 3.8e2, 0.01)
    equilibre = vitesse_de_chute(
        masse=masse, masse_volumique=1.225, coefficient=1.4, surface=30.0
    )
    assert proches(equilibre, 12.0, 1e-9)
    plus_vite = SurfacePortante(Ecoulement(MasseVolumique(1.225), 24.0), 30.0)
    assert proches_relativement(plus_vite.force(1.4), 4 * portance, 1e-12)


def test_a_mach_0_3_la_masse_volumique_varie_de_4_4_pour_cent_de_l_arret() -> None:
    """`(rho_0 - rho) / rho_0` : 4,36 %. Rapportee a `rho`, elle serait 4,56 %."""
    variation = variation_relative_a_l_arret(0.3)
    assert egales_arrondies(100 * variation, 4.4, 1)
    assert variation < VARIATION_ADMISE
    assert est_incompressible(0.29)
    assert not est_incompressible(MACH_INCOMPRESSIBLE)


def test_les_deux_bornes_ne_sont_equivalentes_qu_a_peu_pres() -> None:
    """Les 5 % tombent a `Ma = 0,32` : `Ma < 0,3` est la borne prudente."""
    assert variation_relative_a_l_arret(0.3) < VARIATION_ADMISE
    assert variation_relative_a_l_arret(0.33) > VARIATION_ADMISE


def test_la_variation_de_masse_volumique_croit_comme_le_carre_du_mach() -> None:
    rapport = variation_relative_a_l_arret(0.2) / variation_relative_a_l_arret(0.1)
    assert proches_relativement(rapport, 4.0, 0.02)


def test_mach_0_3_dans_l_air_fait_environ_100_metres_par_seconde() -> None:
    celerite = math.sqrt(
        GAMMA_DE_L_AIR * CONSTANTE_DES_GAZ / AIR.masse_molaire * 293.15
    )
    vitesse = MACH_INCOMPRESSIBLE * celerite
    assert proches(vitesse, 100.0, 5.0)
    assert egales(nombre_de_mach(vitesse, celerite), MACH_INCOMPRESSIBLE)
