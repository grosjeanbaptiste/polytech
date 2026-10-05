"""Le syllabus de thermique, tome 1 : Fourier, la plaque et le tube.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 29 a 47
(premiere partie, chapitre 1 et sections 2.1 et 2.2), lues sur les pages
rendues.

Les oracles : les equations (1.1) a (1.4), (2.2) a (2.12), et l'exemple
numerique 2.2.1 de la tuyauterie calorifugee -- `lambda_c = 0,0535` a
90 degres, 27,59 W par metre.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.maths.domain.analyse_vectorielle import gradient, laplacien
from polytech.thermique.domain.conduction.fourier import (
    ESPACE,
    INSTANT,
    conductivite_lineaire,
    conductivite_moyenne,
    densite_de_flux,
    diffusivite,
    flux_a_travers,
    residu_de_fourier_kirchhoff,
)
from polytech.thermique.domain.conduction.parois import (
    Couche,
    Tube,
    conductivite_interpolee,
    flux_de_plaque,
    flux_entre_deux_fluides,
    flux_lineique_de_tubes,
    profil_de_plaque,
    profil_de_tube,
    temperatures_d_interface,
)
from polytech.thermique.domain.convection.naturelle import (
    resistance_de_paroi_cylindrique,
    resistance_de_paroi_plane,
)

X, Y, Z = ESPACE
LAMBDA = sp.Symbol("lambda", positive=True)


def test_la_loi_de_fourier_est_un_gradient() -> None:
    """(1.1) : `phi = -lambda grad t`, et sa projection sur une normale."""
    champ = 3 * X**2 + X * Y - 2 * Z
    flux = densite_de_flux(champ, LAMBDA)
    attendu = tuple(-LAMBDA * derivee for derivee in gradient(champ, ESPACE))
    assert flux == attendu
    normale = (sp.Rational(3, 5), sp.Rational(4, 5), 0)
    projete = flux_a_travers(flux, normale)
    assert (
        sp.simplify(
            projete + LAMBDA * (sp.Rational(3, 5) * (6 * X + Y) + sp.Rational(4, 5) * X)
        )
        == 0
    )


def test_la_densite_de_flux_s_incline_comme_le_cosinus() -> None:
    """Page 30 : `phi_alpha = phi_n cos(alpha)`."""
    flux = densite_de_flux(-10 * X, sp.Integer(2))
    angle = sp.pi / 3
    normale = (sp.cos(angle), sp.sin(angle), 0)
    assert flux_a_travers(flux, normale) == 20 * sp.cos(angle)


def test_la_diffusivite_thermique() -> None:
    """(1.2) : `a = lambda / (rho c)`. Acier : 46,5 / (7850 x 460), vers 1,3e-5."""
    assert diffusivite(46.5, 7850.0, 460.0) == pytest.approx(1.288e-5, rel=1e-3)
    with pytest.raises(ValueError, match="capacite volumique nulle ou negative"):
        diffusivite(46.5, 0.0, 460.0)


def test_l_equation_de_fourier_kirchhoff() -> None:
    """(1.3) : `dt/dtau = a laplacien(t) + q / (rho c)`.

    Un mode qui s'eteint, puis le profil parabolique d'une generation.
    """
    a, source = sp.symbols("a s", positive=True)
    mode = sp.exp(-a * INSTANT) * sp.sin(X)
    assert residu_de_fourier_kirchhoff(mode, a) == 0
    genere = -source * X**2 / (2 * a)
    assert residu_de_fourier_kirchhoff(genere, a, source=source) == 0
    assert residu_de_fourier_kirchhoff(genere, a) != 0
    assert laplacien(X**2 - Y**2, ESPACE) == 0


def test_la_conductivite_varie_lineairement_avec_la_temperature() -> None:
    """Page 33 : `lambda = lambda_0 (1 + b t)` ; page 45, la moyenne suffit."""
    assert conductivite_lineaire(0.03, 0.005, 100.0) == pytest.approx(0.045)
    t1, t2, ordonnee, pente, epaisseur = sp.symbols("t_1 t_2 a b e", positive=True)
    moyenne = conductivite_moyenne(ordonnee, pente, t1, t2)
    assert moyenne == ordonnee + pente * (t1 + t2) / 2
    temperature = sp.Symbol("t")
    integrale = sp.integrate(ordonnee + pente * temperature, (temperature, t2, t1))
    assert sp.simplify(integrale / epaisseur - moyenne * (t1 - t2) / epaisseur) == 0


def test_le_profil_et_le_flux_d_une_plaque() -> None:
    """(2.2) et (2.3) : profil lineaire, `phi = (lambda / e)(t_1 - t_2)`."""
    profil = profil_de_plaque(X, sp.Rational(1, 5), 80, 20)
    assert profil.subs(X, 0) == 80
    assert profil.subs(X, sp.Rational(1, 5)) == 20
    assert sp.diff(profil, X, 2) == 0
    flux = flux_de_plaque(1.5, 0.2, 80.0, 20.0)
    assert flux == pytest.approx(450.0)
    assert flux == pytest.approx(float(-1.5 * sp.diff(profil, X)))
    assert 60.0 / resistance_de_paroi_plane(0.2, 1.5, 1.0) == pytest.approx(flux)


COUCHES = (Couche(0.1, 1.0), Couche(0.05, 0.05), Couche(0.02, 0.5))


def test_la_paroi_composite_et_ses_temperatures_intermediaires() -> None:
    """(2.5) et (2.6) : trois resistances en serie.

    La troisieme ligne de (2.5) imprime « e_1 / lambda_3 » ; c'est `e_3`.
    """
    resistances = [couche.resistance() for couche in COUCHES]
    assert resistances == pytest.approx([0.1, 1.0, 0.04])
    interfaces, flux = temperatures_d_interface(COUCHES, 134.0, 20.0)
    assert flux == pytest.approx(100.0)
    assert interfaces == pytest.approx([124.0, 24.0])
    derniere_chute = interfaces[1] - 20.0
    assert derniere_chute == pytest.approx(flux * 0.02 / 0.5)
    assert derniere_chute != pytest.approx(flux * 0.1 / 0.5)


def test_la_plaque_entre_deux_fluides() -> None:
    """(2.7) : `phi = (t_i - t_e) / (1 / K_i + e / lambda + 1 / K_e)`."""
    flux = flux_entre_deux_fluides(
        COUCHES[:1], interieur=(10.0, 60.0), exterieur=(5.0, 20.0)
    )
    assert flux == pytest.approx(40.0 / (0.1 + 0.1 + 0.2))
    with pytest.raises(ValueError, match="coefficient de transmission nul ou negatif"):
        flux_entre_deux_fluides(
            COUCHES[:1], interieur=(0.0, 60.0), exterieur=(5.0, 20.0)
        )


def test_le_profil_logarithmique_d_un_tube() -> None:
    """(2.8) : `(t - t_1) / (t_2 - t_1) = ln(r / r_1) / ln(r_2 / r_1)`."""
    rayon = sp.Symbol("r", positive=True)
    profil = profil_de_tube(rayon, (1, 3), (150, 30))
    assert profil.subs(rayon, 1) == 150
    assert profil.subs(rayon, 3) == 30
    cylindrique = sp.diff(rayon * sp.diff(profil, rayon), rayon)
    assert sp.simplify(cylindrique) == 0


def test_le_flux_lineique_d_un_tube() -> None:
    """(2.10) : `2 pi lambda (t_1 - t_2) / ln(r_2 / r_1)`, conserve quand `r` varie."""
    tube = Tube(rayon_interieur=0.0255, rayon_exterieur=0.029, conductivite=46.5)
    flux = flux_lineique_de_tubes((tube,), 150.0, 149.0)
    assert flux == pytest.approx(2 * math.pi * 46.5 / math.log(29 / 25.5))
    assert 1.0 / resistance_de_paroi_cylindrique(
        0.0255, 0.029, 46.5, 1.0
    ) == pytest.approx(flux)
    with pytest.raises(ValueError, match="rayon exterieur non superieur"):
        Tube(rayon_interieur=0.03, rayon_exterieur=0.029, conductivite=46.5)


def test_la_condition_a_la_face_externe_d_un_tube() -> None:
    """Page 44 : en `r = r_2`, le flux sortant vaut `K_e (t - t_e)`.

    Le syllabus y ecrit « K_e (t_e - t) = -lambda dt/dr », le signe de la
    face interne ; sa propre chute partielle, `t_2 - t_e = Phi / (2 pi r_2
    K_e)`, donne l'autre.
    """
    rayon = sp.Symbol("r", positive=True)
    profil = profil_de_tube(rayon, (1, 3), (150, 30))
    sortant = -LAMBDA * sp.diff(profil, rayon).subs(rayon, 3)
    assert sortant.is_positive
    lineique = 2 * sp.pi * 3 * sortant
    exterieur = sp.Symbol("K_e", positive=True)
    ambiante = 30 - lineique / (2 * sp.pi * 3 * exterieur)
    assert sp.simplify(sortant - exterieur * (30 - ambiante)) == 0
    assert sp.simplify(sortant - exterieur * (ambiante - 30)) != 0


TABLE = ((0.0, 0.0326), (60.0, 0.0465), (100.0, 0.0558))
"""Page 46 : la conductivite apparente de la coquille de fibres de verre."""


def test_l_exemple_2_2_1_la_tuyauterie_calorifugee() -> None:
    """Acier 51/58 mm, coquille 58/250 mm, de 150 a 30 degres : 27,59 W/m."""
    isolant = conductivite_interpolee(TABLE, (150.0 + 30.0) / 2)
    assert isolant == pytest.approx(0.0535, abs=5e-5)
    acier = Tube(0.0255, 0.029, 46.5)
    coquille = Tube(0.029, 0.125, isolant)
    assert acier.resistance() * 2 * math.pi == pytest.approx(0.0028, abs=5e-5)
    assert coquille.resistance() * 2 * math.pi == pytest.approx(27.327, abs=0.03)
    flux = flux_lineique_de_tubes((acier, coquille), 150.0, 30.0)
    assert flux == pytest.approx(27.59, abs=0.02)
    chute = flux * acier.resistance()
    assert chute == pytest.approx(120e-4, abs=5e-4)
    with pytest.raises(ValueError, match="hors de la table"):
        conductivite_interpolee(TABLE, 150.0)


def test_la_densite_de_flux_a_la_face_externe_du_calorifuge() -> None:
    """`phi_3 = Phi / (2 pi r_3)`. Le syllabus imprime 35,36 ; c'est 35,13."""
    densite = 27.59 / (math.pi * 0.25)
    assert densite == pytest.approx(35.13, abs=0.005)
    assert abs(densite - 35.36) > 0.2


def test_le_tube_entre_deux_fluides() -> None:
    """(2.12) : `1 / (K_i r_1)` et `1 / (K_e r_(n+1))` s'ajoutent aux logarithmes."""
    tube = Tube(0.0255, 0.029, 46.5)
    flux = flux_lineique_de_tubes(
        (tube,), 150.0, 30.0, interieur=2000.0, exterieur=10.0
    )
    somme = 1 / (2000 * 0.0255) + math.log(29 / 25.5) / 46.5 + 1 / (10 * 0.029)
    assert flux == pytest.approx(2 * math.pi * 120 / somme)
