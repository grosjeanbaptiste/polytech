"""Planches de mecanique des fluides, chapitre IX : portance et trainee.

Source : `MecaniqueFluides/SLIDES/Slides_08_Aerodynamique.pdf`, planches 19
a 37, lues sur les pages rendues.

Les oracles du cours : `D_p = Delta p x A` (planche 26), le coefficient
`0,34` de l'exemple 11-1 (planche 29), le tableau de la planche 30,
`C_D = 24 / Re` (planche 31), et les donnees des deux exercices.
"""

from __future__ import annotations

import math

import pytest

from polytech.fluides.domain.aerodynamique import (
    Ecoulement,
    SurfacePortante,
    trainee_induite,
)
from polytech.fluides.domain.efforts_aerodynamiques import Facette, resultante
from polytech.fluides.domain.grandeurs import PESANTEUR, MasseVolumique
from polytech.fluides.domain.trainee import (
    COEFFICIENTS_DE_TRAINEE_AUTOMOBILE,
    coefficient_de_stokes,
    hauteur_de_chute_libre,
    variation_de_la_trainee,
    vitesse_de_chute,
)

DROIT = math.pi / 2


def test_la_trainee_de_pression_d_une_plaque_face_au_vent() -> None:
    """Planche 26 : `D_p = Delta p x A`."""
    amont = Facette(pression=101_500.0, frottement=0.0, angle=math.pi, aire=2.0)
    aval = Facette(pression=101_000.0, frottement=0.0, angle=0.0, aire=2.0)
    trainee, portance = resultante((amont, aval))
    assert trainee == pytest.approx(500.0 * 2.0)
    assert portance == pytest.approx(0.0, abs=1e-9)


def test_la_trainee_de_friction_d_une_plaque_dans_le_lit_du_vent() -> None:
    """Planche 24 : seul le frottement freine une plaque alignee.

    Vers l'aval des deux cotes : positif dessus, negatif dessous, puisque
    `tau_w` se compte dans le sens de la normale tournee vers la droite.
    """
    dessus = Facette(pression=101_000.0, frottement=3.0, angle=DROIT, aire=2.0)
    dessous = Facette(pression=101_000.0, frottement=-3.0, angle=-DROIT, aire=2.0)
    trainee, portance = resultante((dessus, dessous))
    assert trainee == pytest.approx(2 * 3.0 * 2.0)
    assert portance == pytest.approx(0.0, abs=1e-9)


def test_la_portance_vient_de_l_ecart_de_pression_entre_les_deux_faces() -> None:
    """Planche 20 : souffler sur le dessus d'une feuille la souleve."""
    dessus = Facette(pression=100_900.0, frottement=0.0, angle=DROIT, aire=2.0)
    dessous = Facette(pression=101_000.0, frottement=0.0, angle=-DROIT, aire=2.0)
    assert resultante((dessus, dessous))[1] == pytest.approx(100.0 * 2.0)


def test_une_pression_uniforme_ne_pousse_pas_un_corps_ferme() -> None:
    cotes = 12
    facettes = tuple(
        Facette(101_325.0, 0.0, 2 * math.pi * k / cotes, 0.5) for k in range(cotes)
    )
    trainee, portance = resultante(facettes)
    assert trainee == pytest.approx(0.0, abs=1e-8)
    assert portance == pytest.approx(0.0, abs=1e-8)


def test_le_frottement_est_tangent_a_la_paroi() -> None:
    """La planche 22 ecrit `+ tau_w cos(theta)` dans `L` ; c'est `-`.

    La figure dessine `tau_w` vers l'aval sur une paroi de normale
    `(cos theta, sin theta)` : sa direction est `(sin theta, -cos theta)`.
    Avec le signe de la planche, le frottement aurait une composante
    normale a la paroi.
    """
    angle = math.radians(70.0)
    facette = Facette(pression=0.0, frottement=5.0, angle=angle, aire=1.0)
    selon_x, selon_y = facette.force()
    assert facette.composante_normale((selon_x, selon_y)) == pytest.approx(0.0)
    assert selon_x > 0 > selon_y
    de_la_planche = facette.force_avec_le_signe_de_la_planche_22()
    normale = facette.composante_normale(de_la_planche)
    assert normale == pytest.approx(5.0 * math.sin(2 * angle))
    assert normale == pytest.approx(3.2139, abs=5e-5)


def test_une_facette_sans_aire_est_refusee() -> None:
    with pytest.raises(ValueError, match="aire nulle ou negative"):
        Facette(pression=1.0, frottement=0.0, angle=0.0, aire=0.0)
    with pytest.raises(ValueError, match="aucune facette"):
        resultante(())


LIVRE_FORCE = 4.4482216152605
LIVRE = 0.45359237
PIED = 0.3048
MILLE_PAR_HEURE = 1609.344 / 3600
"""Definitions exactes des unites de l'exemple 11-1."""


def test_le_coefficient_de_trainee_de_la_voiture_de_l_exemple_11_1() -> None:
    """Planche 29 : 68 lbf a 60 mi/h sur 22,26 ft^2, `C_D = 0,34`."""
    air = MasseVolumique(0.07489 * LIVRE / PIED**3)
    voiture = SurfacePortante(
        Ecoulement(air, 60 * MILLE_PAR_HEURE), surface_de_reference=22.26 * PIED**2
    )
    assert voiture.coefficient(68 * LIVRE_FORCE) == pytest.approx(0.34, abs=0.005)


def test_le_tableau_des_voitures_de_la_planche_30() -> None:
    assert COEFFICIENTS_DE_TRAINEE_AUTOMOBILE == {
        "Peugeot 5008": 0.28,
        "Porsche Cayenne": 0.36,
        "Audi A3": 0.31,
        "Tesla Model S": 0.24,
    }


def test_le_cycliste_releve_traine_presque_deux_fois_plus() -> None:
    """Planche 26 : `C_D = 0,5` couche, `0,9` releve, a surface et vitesse egales."""
    cycliste = SurfacePortante(Ecoulement(MasseVolumique(1.2), 10.0), 0.4)
    assert cycliste.force(0.9) / cycliste.force(0.5) == pytest.approx(1.8)


def test_la_loi_de_stokes_de_la_planche_31() -> None:
    """`C_D = 24 / Re`, la droite de gauche de la courbe de la sphere."""
    assert coefficient_de_stokes(0.1) == pytest.approx(240.0)
    assert coefficient_de_stokes(1.0) == pytest.approx(24.0)
    with pytest.raises(ValueError, match="hors du regime de Stokes"):
        coefficient_de_stokes(100.0)
    with pytest.raises(ValueError, match="Reynolds nul ou negatif"):
        coefficient_de_stokes(0.0)


def test_l_exercice_1_ralentir_de_120_a_90_km_h() -> None:
    """La consommation suit la force, et la force le carre de la vitesse : -43,75 %."""
    assert variation_de_la_trainee(120.0, 90.0) == pytest.approx(-0.4375)
    voiture = SurfacePortante(Ecoulement(MasseVolumique(1.2), 120 / 3.6), 2.0)
    lente = SurfacePortante(Ecoulement(MasseVolumique(1.2), 90 / 3.6), 2.0)
    assert lente.force(0.3) / voiture.force(0.3) - 1 == pytest.approx(-0.4375)
    with pytest.raises(ValueError, match="vitesse nulle ou negative"):
        variation_de_la_trainee(0.0, 90.0)


AIR = 1.2
"""Masse volumique de l'air, en kg/m^3 : l'exercice 2 ne la donne pas."""


def test_l_exercice_2_la_vitesse_de_chute_du_parachutiste() -> None:
    """120 kg, parachute de 6 m, `C_D = 1,3` sur `A = pi D^2 / 4`."""
    surface = math.pi * 6.0**2 / 4
    vitesse = vitesse_de_chute(
        masse=120.0, masse_volumique=AIR, coefficient=1.3, surface=surface
    )
    assert vitesse == pytest.approx(7.31, abs=0.01)
    parachute = SurfacePortante(Ecoulement(MasseVolumique(AIR), vitesse), surface)
    assert parachute.force(1.3) == pytest.approx(120.0 * PESANTEUR)


def test_l_exercice_2_la_hauteur_de_saut_equivalente() -> None:
    """`h = V^2 / (2 g)`, soit `m / (rho C_D A)` : la pesanteur n'y est plus."""
    surface = math.pi * 6.0**2 / 4
    vitesse = vitesse_de_chute(
        masse=120.0, masse_volumique=AIR, coefficient=1.3, surface=surface
    )
    hauteur = hauteur_de_chute_libre(vitesse)
    assert hauteur == pytest.approx(2.72, abs=0.01)
    assert hauteur == pytest.approx(120.0 / (AIR * 1.3 * surface))
    with pytest.raises(ValueError, match="surface nulle ou negative"):
        vitesse_de_chute(masse=120.0, masse_volumique=AIR, coefficient=1.3, surface=0.0)


def test_la_trainee_induite_croit_comme_le_carre_de_la_portance() -> None:
    """Planche 34 : les tourbillons de bout d'aile, prix de la portance."""
    assert trainee_induite(1.0, 8.0) == pytest.approx(4 * trainee_induite(0.5, 8.0))
