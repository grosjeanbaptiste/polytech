"""Les planches de thermique : la conductibilite, deuxieme partie.

Source : `ThermiqueFondements/SLIDES/Slides_03_Conductibilite_2.pdf`, les
19 planches, lues une a une.

Les oracles : les ailettes des planches 9 a 19 -- leurs trois graphes
compris --, et le crayon de reacteur des planches 20 a 24, dont les
resultats sont imprimes.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.thermique.domain.conduction.ailette_circulaire import (
    efficacite_circulaire,
)
from polytech.thermique.domain.conduction.ailette_droite import (
    flux_au_pied,
    flux_d_une_surface_ailetee,
    flux_sans_resistance,
    parametre_d_ailette,
    profil_d_ailette,
)
from polytech.thermique.domain.conduction.generation import (
    BarreauGenerateur,
    chute_dans_une_gaine,
    chute_de_convection,
)

X = sp.Symbol("x", nonnegative=True)


def test_l_equation_de_l_ailette_droite() -> None:
    """Planches 12 et 13 : `theta'' - m^2 theta = 0`, `m = sqrt(K P / (lambda S))`."""
    parametre = parametre_d_ailette(25.0, 2.02, 200.0, 0.002)
    assert parametre == pytest.approx(math.sqrt(25.0 * 2.02 / (200.0 * 0.002)))
    m, longueur = sp.symbols("m l", positive=True)
    profil = profil_d_ailette(X, m, longueur)
    assert sp.simplify(sp.diff(profil, X, 2) - m**2 * profil) == 0
    assert profil.subs(X, 0) == 1
    assert sp.diff(profil, X).subs(X, longueur) == 0
    assert sp.simplify(profil.subs(X, longueur) - 1 / sp.cosh(m * longueur)) == 0


def test_le_profil_de_la_planche_15() -> None:
    """Le graphe pour `m l = 1,5` : il finit a `1 / cosh(1,5)`, vers 0,43."""
    profil = profil_d_ailette(X, 1.5, 1)
    assert float(profil.subs(X, 1)) == pytest.approx(0.425, abs=0.005)
    assert float(profil.subs(X, 0.5)) == pytest.approx(0.55, abs=0.01)


def test_le_flux_draine_sature_quand_l_ailette_s_allonge() -> None:
    """Planche 14 : `Phi_1 = lambda S theta_1 m tanh(m l)`, et `tanh` plafonne a un."""
    conductance, parametre, ecart = 0.4, 11.0, 60.0
    flux = [
        flux_au_pied(parametre, conductance, ecart=ecart, longueur=longueur)
        for longueur in (0.05, 0.2, 0.6, 2.0)
    ]
    assert flux == sorted(flux)
    plafond = conductance * ecart * parametre
    assert flux[-1] == pytest.approx(plafond)
    assert flux[2] == pytest.approx(plafond, rel=1e-4)
    assert flux[0] < 0.6 * plafond


def test_l_efficacite_de_l_ailette_droite() -> None:
    """Planche 15 : `Omega = Phi_1 / (K theta_1 P l) = tanh(m l) / (m l)`."""
    echange, perimetre, conductivite, section = 25.0, 2.02, 200.0, 0.002
    parametre = parametre_d_ailette(echange, perimetre, conductivite, section)
    longueur, ecart = 0.08, 60.0
    draine = flux_au_pied(
        parametre, conductivite * section, ecart=ecart, longueur=longueur
    )
    ideal = flux_sans_resistance(echange, perimetre, ecart=ecart, longueur=longueur)
    reduit = parametre * longueur
    assert draine / ideal == pytest.approx(math.tanh(reduit) / reduit)
    assert math.tanh(4.0) / 4.0 == pytest.approx(0.25, abs=0.005)
    assert math.tanh(1.0) / 1.0 == pytest.approx(0.76, abs=0.005)


def test_le_flux_total_d_une_surface_ailetee() -> None:
    """Planche 16 : `Phi_TOT = K theta_1 (S_0 + Omega S_a)`."""
    total = flux_d_une_surface_ailetee(
        25.0, 60.0, primaire=0.3, secondaire=2.0, efficacite=0.8
    )
    assert total == pytest.approx(25.0 * 60.0 * (0.3 + 0.8 * 2.0))
    nue = flux_d_une_surface_ailetee(
        25.0, 60.0, primaire=0.5, secondaire=0.0, efficacite=0.8
    )
    assert total > 3 * nue


def test_l_efficacite_de_l_ailette_circulaire() -> None:
    """Planche 18 : `Omega = f(m (r_2 - r_1), r_2 / r_1)`, sous l'ailette droite."""
    droite = math.tanh(1.0) / 1.0
    mince = efficacite_circulaire(1.0, 100.0, 101.0)
    assert mince == pytest.approx(droite, abs=0.005)
    large = efficacite_circulaire(1.0, 1.0, 2.0)
    assert large == pytest.approx(0.70, abs=0.02)
    assert efficacite_circulaire(1.0, 1.0, 4.0) < large < mince


def test_l_erreur_d_immersion_d_un_godet() -> None:
    """Planche 19 : `t_l - t_f = (t_p - t_f) / cosh(m l)` ; un godet long lit juste."""
    paroi, fluide = 80.0, 200.0
    erreurs = [
        (paroi - fluide) * float(profil_d_ailette(longueur, 40.0, longueur))
        for longueur in (0.02, 0.05, 0.15)
    ]
    assert erreurs[0] == pytest.approx(-120.0 / math.cosh(0.8))
    assert [abs(e) for e in erreurs] == sorted((abs(e) for e in erreurs), reverse=True)
    assert abs(erreurs[-1]) < 1.0


CRAYON = BarreauGenerateur(generation=800e6, rayon=0.00385, conductivite=1.745)


def test_le_profil_parabolique_du_crayon() -> None:
    """Planche 22 : `(1/r)(r t')' + q/lambda = 0`, `t_c - t_0 = q R^2 / (4 lambda)`."""
    rayon = sp.Symbol("r", positive=True)
    q, grand, conductivite = sp.symbols("q R lambda", positive=True)
    profil = BarreauGenerateur(q, grand, conductivite).profil(rayon, surface=0)
    cylindrique = sp.diff(rayon * sp.diff(profil, rayon), rayon) / rayon
    assert sp.simplify(cylindrique + q / conductivite) == 0
    assert profil.subs(rayon, grand) == 0
    lineique = q * sp.pi * grand**2
    au_centre = profil.subs(rayon, 0)
    assert sp.simplify(au_centre - lineique / (4 * sp.pi * conductivite)) == 0


def test_les_resultats_du_crayon_de_la_planche_24() -> None:
    """`q = 800e6 W/m^3` : 37,2 kW/m, 1696, 61,2 et 53,2 degres ; `t_c = 2145`."""
    lineique = 800e6 * math.pi * 0.00385**2
    assert lineique == pytest.approx(37.2e3, rel=2e-3)
    au_centre = float(CRAYON.echauffement_au_centre())
    assert au_centre == pytest.approx(1696.0, abs=3.0)
    gaine = chute_dans_une_gaine(800e6, 0.0077, 0.0089, conductivite=14.0)
    assert gaine == pytest.approx(61.2, abs=0.15)
    ebullition = chute_de_convection(800e6, 0.0077, 0.0089, echange=25000.0)
    assert ebullition == pytest.approx(53.2, abs=0.15)
    face_externe = 335.0 + ebullition
    assert face_externe == pytest.approx(388.0, abs=0.5)
    assert face_externe + gaine + au_centre == pytest.approx(2145.0, abs=3.5)


def test_la_planche_24_imprime_459_pour_449() -> None:
    """`t_1 = t_2 + 61,2 = 449 degres`. La planche ecrit 459 ; son `t_c` suppose 449."""
    assert pytest.approx(449.2) == 388.0 + 61.2
    assert pytest.approx(2145.0, abs=0.5) == 449.2 + 1696.0
    assert pytest.approx(2145.0, abs=5.0) != 459.0 + 1696.0


def test_la_dissipation_est_de_quelques_centaines_de_megawatts() -> None:
    """Planche 20 : « 400 a 800 MW/m^3 ». La planche 24 ecrit « 4.10^6 - 8.10^6 »."""
    faible = BarreauGenerateur(generation=8e6, rayon=0.00385, conductivite=1.745)
    assert float(faible.echauffement_au_centre()) == pytest.approx(17.0, abs=0.1)
    assert float(CRAYON.echauffement_au_centre()) > 1600.0
