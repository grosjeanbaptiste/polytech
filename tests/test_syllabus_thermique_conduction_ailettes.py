"""Le syllabus de thermique, tome 1 : ailettes et generation interne.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 47 a 61
(premiere partie, sections 2.3 et 2.4, exercices I.2.1 et I.2.2), lues
sur les pages rendues.

Les oracles : les equations (2.13) a (2.23), la table 2.1, le crayon de
reacteur (1750 degres d'echauffement, 2137 au centre), et les deux
exercices resolus -- 26,23 par metre, 0,88, 1056 et 4737 W/m ; 22,57 mm,
48,91 par metre, 0,78, 8,39 et 6,54 W par ailette.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.thermique.domain.conduction.ailette_circulaire import (
    efficacite_circulaire,
    rayon_equivalent_d_un_carre,
)
from polytech.thermique.domain.conduction.ailette_droite import (
    flux_au_pied,
    flux_d_une_surface_ailetee,
    flux_sans_resistance,
    parametre_d_ailette,
    parametre_d_ailette_mince,
    profil_d_ailette,
)
from polytech.thermique.domain.conduction.ailettes import (
    AiletteCirculaire,
    AiletteDroite,
)
from polytech.thermique.domain.conduction.generation import (
    BarreauGenerateur,
    chute_dans_une_gaine,
    chute_de_convection,
)

X = sp.Symbol("x", nonnegative=True)
M, LONGUEUR = sp.symbols("m l", positive=True)

TABLE_2_1 = {
    0.5: (1.128, 0.462),
    1: (1.543, 0.761),
    1.5: (2.352, 0.905),
    2: (3.762, 0.964),
    3: (10.07, 0.995),
    5: (74.21, 0.9999),
}


@pytest.mark.parametrize("argument", sorted(TABLE_2_1))
def test_la_table_2_1_des_fonctions_hyperboliques(argument: float) -> None:
    cosinus, tangente = TABLE_2_1[argument]
    assert math.cosh(argument) == pytest.approx(cosinus, rel=5e-4)
    assert math.tanh(argument) == pytest.approx(tangente, abs=1e-3)


def test_la_table_2_1_arrondit_mal_le_cosinus_de_six() -> None:
    """Elle imprime 201,5 ; `cosh(6)` vaut 201,7."""
    assert math.cosh(6) == pytest.approx(201.7, abs=0.05)
    assert abs(math.cosh(6) - 201.5) > 0.2


def test_le_profil_de_l_ailette_resout_son_equation() -> None:
    """(2.13) : `theta'' - m^2 theta = 0`, `theta(0) = theta_1`, pente nulle au bout."""
    profil = profil_d_ailette(X, M, LONGUEUR)
    assert sp.simplify(sp.diff(profil, X, 2) - M**2 * profil) == 0
    assert profil.subs(X, 0) == 1
    assert sp.diff(profil, X).subs(X, LONGUEUR) == 0
    assert sp.simplify(profil.subs(X, LONGUEUR) - 1 / sp.cosh(M * LONGUEUR)) == 0
    droite = AiletteDroite(parametre=2.0, longueur=0.5)
    assert droite.ecart_en_bout(1.0) == pytest.approx(
        float(profil.subs({X: 0.5, M: 2, LONGUEUR: 0.5}))
    )


def test_le_parametre_d_une_ailette() -> None:
    """(2.14) : `m = racine(K P / (lambda S))` ; mince, `racine(2 K / (lambda b))`."""
    epaisseur, largeur = 0.005, 1.0
    general = parametre_d_ailette(80.0, 2 * largeur, 46.5, epaisseur * largeur)
    assert general == pytest.approx(parametre_d_ailette_mince(80.0, 46.5, epaisseur))
    with pytest.raises(ValueError, match="section nulle ou negative"):
        parametre_d_ailette(80.0, 2.0, 46.5, 0.0)


def test_le_flux_au_pied_est_celui_que_l_ailette_cede() -> None:
    """(2.16) : `Phi_1 = racine(K P lambda S) theta_1 tanh(m l)`."""
    echange, perimetre, conductivite, section = 80.0, 2.0, 46.5, 0.005
    parametre = parametre_d_ailette(echange, perimetre, conductivite, section)
    au_pied = flux_au_pied(
        parametre, conductivite * section, ecart=300.0, longueur=0.025
    )
    racine = math.sqrt(echange * perimetre * conductivite * section)
    assert au_pied == pytest.approx(racine * 300 * math.tanh(parametre * 0.025))
    profil = 300 * profil_d_ailette(X, parametre, 0.025)
    cede = echange * perimetre * sp.integrate(profil, (X, 0, 0.025))
    assert float(cede) == pytest.approx(au_pied)


def test_l_efficacite_est_le_flux_rapporte_a_l_ailette_isotherme() -> None:
    """(2.17) : `Omega = Phi_1 / (K theta_1 P l) = tanh(m l) / (m l)`."""
    parametre = parametre_d_ailette(80.0, 2.0, 46.5, 0.005)
    reel = flux_au_pied(parametre, 46.5 * 0.005, ecart=300.0, longueur=0.025)
    ideal = flux_sans_resistance(80.0, 2.0, ecart=300.0, longueur=0.025)
    droite = AiletteDroite(parametre=parametre, longueur=0.025)
    assert reel / ideal == pytest.approx(droite.efficacite())


def test_l_exercice_i_2_1_le_tube_d_economiseur() -> None:
    """Deux ailettes de 5 mm x 25 mm, acier, `K = 80`, 300 degres d'ecart."""
    parametre = parametre_d_ailette_mince(80.0, 46.5, 0.005)
    assert parametre == pytest.approx(26.23, abs=0.005)
    droite = AiletteDroite(parametre=parametre, longueur=0.025)
    assert droite.argument == pytest.approx(0.66, abs=0.005)
    assert droite.efficacite() == pytest.approx(0.88, abs=0.005)
    maximal = flux_sans_resistance(80.0, 2.0, ecart=300.0, longueur=0.025)
    assert maximal == pytest.approx(1200.0)
    assert 0.88 * maximal == pytest.approx(1056.0)
    primaire = (math.pi * 0.038 - 2 * 0.005) * 80.0 * 300.0
    assert primaire == pytest.approx(2625.0, abs=0.5)
    total = flux_d_une_surface_ailetee(
        80.0,
        300.0,
        primaire=math.pi * 0.038 - 0.01,
        secondaire=2 * 2 * 0.025,
        efficacite=0.88,
    )
    assert total == pytest.approx(4737.0, abs=1.0)
    assert total / (math.pi * 0.038 * 80.0 * 300.0) == pytest.approx(1.65, abs=0.01)


def test_l_efficacite_exacte_de_l_ailette_circulaire() -> None:
    """Page 54 : les fonctions de Bessel modifiees ; `r_2 / r_1 = 1` rend la droite."""
    parametre, pied = 48.91, 0.008
    assert efficacite_circulaire(parametre, 100.0, 100.01457) == pytest.approx(
        AiletteDroite(parametre, 0.01457).efficacite(), abs=1e-4
    )
    exacte = efficacite_circulaire(parametre, pied, 0.02257)
    assert exacte < AiletteDroite(parametre, 0.01457).efficacite()
    du_formulaire = AiletteCirculaire(AiletteDroite(parametre, 0.01457), pied, 0.02257)
    assert exacte == pytest.approx(du_formulaire.efficacite(), abs=0.01)
    with pytest.raises(ValueError, match="rayon exterieur non superieur"):
        efficacite_circulaire(parametre, pied, pied)


def test_l_exercice_i_2_2_les_ailettes_carrees_d_aluminium() -> None:
    """Cote 40 mm, 0,2 mm d'epaisseur, sur un tube de 16 mm ; `K = 50`, 60 degres."""
    exterieur = rayon_equivalent_d_un_carre(0.040)
    assert exterieur * 1000 == pytest.approx(22.57, abs=0.005)
    assert exterieur / 0.008 == pytest.approx(2.82, abs=0.005)
    parametre = parametre_d_ailette_mince(50.0, 209.0, 0.0002)
    assert parametre == pytest.approx(48.91, abs=0.005)
    assert parametre * (exterieur - 0.008) == pytest.approx(0.713, abs=0.001)
    efficacite = efficacite_circulaire(parametre, 0.008, exterieur)
    assert efficacite == pytest.approx(0.78, abs=0.01)
    ideal = 2 * (0.040**2 - math.pi * 0.008**2) * 50.0 * 60.0
    assert ideal == pytest.approx(8.39, abs=0.005)
    assert 0.78 * ideal == pytest.approx(6.54, abs=0.01)


def test_l_exercice_i_2_2_l_ailette_d_acier_equivalente() -> None:
    """`lambda b` constant : 0,9 mm d'acier, et treize fois la masse."""
    acier = 0.2 * 209 / 46.5
    assert acier == pytest.approx(0.9, abs=0.005)
    alu = parametre_d_ailette_mince(50.0, 209.0, 0.0002)
    assert parametre_d_ailette_mince(50.0, 46.5, acier / 1000) == pytest.approx(alu)
    assert pytest.approx(13.0, abs=0.05) == 0.9 * 7850 / (0.2 * 2720)


RAYON, R, Q, LAMBDA, T0 = sp.symbols("r R q lambda t_0", positive=True)


def test_le_profil_parabolique_d_un_barreau_generateur() -> None:
    """(2.20) et (2.21) : `t - t_0 = q (R^2 - r^2) / (4 lambda)`.

    Le syllabus ecrit l'equation « ... - q / lambda = 0 » ; (2.20), dont
    elle sort, et la ligne suivante portent le signe plus.
    """
    barreau = BarreauGenerateur(generation=Q, rayon=R, conductivite=LAMBDA)
    profil = barreau.profil(RAYON, surface=T0)
    cylindrique = sp.diff(profil, RAYON, 2) + sp.diff(profil, RAYON) / RAYON
    assert sp.simplify(cylindrique + Q / LAMBDA) == 0
    assert sp.simplify(cylindrique - Q / LAMBDA) != 0
    assert profil.subs(RAYON, R) == T0
    assert sp.diff(profil, RAYON).subs(RAYON, 0) == 0
    assert barreau.echauffement_au_centre() == Q * R**2 / (4 * LAMBDA)


def test_le_barreau_au_contact_d_un_fluide() -> None:
    """(2.22) et (2.23) : `t - t_f = R q / (2 K) + ...`, et `phi_R = R q / 2`."""
    echange, fluide = sp.symbols("K t_f", positive=True)
    barreau = BarreauGenerateur(generation=Q, rayon=R, conductivite=LAMBDA)
    assert barreau.densite_a_la_surface() == R * Q / 2
    bilan = sp.pi * R**2 * Q - 2 * sp.pi * R * barreau.densite_a_la_surface()
    assert sp.simplify(bilan) == 0
    profil = barreau.profil_dans_un_fluide(RAYON, echange, fluide)
    surface = profil.subs(RAYON, R)
    assert (
        sp.simplify(echange * (surface - fluide) - barreau.densite_a_la_surface()) == 0
    )
    sortant = -LAMBDA * sp.diff(profil, RAYON).subs(RAYON, R)
    assert sp.simplify(sortant - barreau.densite_a_la_surface()) == 0


def test_le_crayon_de_reacteur_nucleaire() -> None:
    """Pastilles de 7,7 mm, 826 MW/m^3, `lambda_c = 1,745` : 1750 degres."""
    crayon = BarreauGenerateur(generation=826e6, rayon=0.00385, conductivite=1.745)
    echauffement = float(crayon.echauffement_au_centre())
    assert echauffement == pytest.approx(1750.0, abs=5.0)
    assert 387.0 + 1750.0 == 2137.0
    assert 387.0 + echauffement < 2760.0


def test_les_deux_autres_chutes_du_crayon() -> None:
    """`d_1^2 q ln(r_2 / r_1) / (8 lambda_g)`, puis `d_1^2 q / (4 K d_2)`."""
    generation, pastille, gaine = 826e6, 0.0077, 0.0091
    lineique = math.pi * pastille**2 * generation / 4
    dans_la_gaine = chute_dans_une_gaine(generation, pastille, gaine, conductivite=15.0)
    resistance = math.log(gaine / pastille) / (2 * math.pi * 15.0)
    assert dans_la_gaine == pytest.approx(lineique * resistance)
    vers_l_eau = chute_de_convection(generation, pastille, gaine, echange=40_000.0)
    assert vers_l_eau == pytest.approx(lineique / (40_000.0 * math.pi * gaine))
    with pytest.raises(ValueError, match="gaine plus etroite"):
        chute_dans_une_gaine(generation, gaine, pastille, conductivite=15.0)
