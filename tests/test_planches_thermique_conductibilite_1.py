"""Les planches de thermique : la conductibilite, premiere partie.

Source : `ThermiqueFondements/SLIDES/Slides_02_Conductibilite_1.pdf`, les
25 planches, lues une a une.

Les oracles : le capteur solaire des planches 3 a 6, l'echelle des
conductivites de la planche 12, les conditions aux limites de la
planche 15, les problemes stationnaires des planches 17 a 23, et
l'exercice du tuyau calorifuge, que les planches posent sans le resoudre.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.thermique.domain.capteur_solaire import CapteurPlan
from polytech.thermique.domain.conduction.fourier import (
    ESPACE,
    INSTANT,
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
    profil_de_tube,
    temperatures_d_interface,
)
from polytech.thermique.domain.conduction.tuyau_calorifuge import (
    CONDUCTIVITES_TYPIQUES,
    Etat,
    TuyauCalorifuge,
    temperature_le_long_du_tuyau,
)

X, Y, Z = ESPACE


def test_le_capteur_solaire_par_superposition() -> None:
    """Planche 5 : `phi_ut = phi_a R_p/(R_c + R_p) + (t_amb - t_eau)/(R_c + R_p)`."""
    capteur = CapteurPlan(0.02, 0.25, 0.85)
    absorbe, eau, ambiance = 600.0, 50.0, 10.0
    par_le_flux = absorbe * 0.25 / 0.27
    par_l_ecart = (ambiance - eau) / 0.27
    utile = capteur.flux_utile(absorbe, eau - ambiance)
    assert utile == pytest.approx(par_le_flux + par_l_ecart)
    assert capteur.flux_utile(absorbe, 0.0) == pytest.approx(par_le_flux)
    assert capteur.flux_utile(0.0, eau - ambiance) == pytest.approx(par_l_ecart)


def test_le_rendement_du_capteur_est_une_droite() -> None:
    """Planche 6 : `eta = F_c [tau alpha - K_p (t_e - t_amb) / phi_sol]`."""
    capteur = CapteurPlan(0.02, 0.25, 0.85)
    facteur = capteur.facteur()
    assert facteur == pytest.approx(0.25 / 0.27)
    surface = 2.0
    rendements = [capteur.rendement(reduit, surface) for reduit in (0.0, 0.05, 0.1)]
    assert rendements[0] == pytest.approx(facteur * 0.85)
    pente = (rendements[1] - rendements[0]) / 0.05
    assert pente == pytest.approx(-facteur / (0.25 * surface))
    assert rendements[2] - rendements[1] == pytest.approx(rendements[1] - rendements[0])


def test_la_loi_de_fourier_et_sa_projection() -> None:
    """Planches 10 et 11 : `phi = -lambda grad t`, `phi_p = phi_n cos(alpha)`."""
    flux = densite_de_flux(-40 * X, sp.Integer(2))
    assert flux == (80, 0, 0)
    incline = (sp.cos(sp.pi / 3), sp.sin(sp.pi / 3), 0)
    assert flux_a_travers(flux, incline) == 40


def test_l_echelle_des_conductivites() -> None:
    """Planche 12 : de 1e-2 a 4e2 W/(m K) ; solides, puis liquides, puis gaz."""
    valeurs = [conductivite for _, conductivite in CONDUCTIVITES_TYPIQUES.values()]
    assert min(valeurs) == 0.025
    assert max(valeurs) == 380.0
    assert all(1e-2 <= valeur <= 4e2 for valeur in valeurs)
    assert CONDUCTIVITES_TYPIQUES["cuivre"] == (Etat.SOLIDE, 380.0)
    assert CONDUCTIVITES_TYPIQUES["eau"] == (Etat.LIQUIDE, 0.6)
    assert CONDUCTIVITES_TYPIQUES["hydrogene"] == (Etat.GAZ, 0.17)

    def plus_faible(etat: Etat) -> float:
        return min(v for e, v in CONDUCTIVITES_TYPIQUES.values() if e is etat)

    def plus_forte(etat: Etat) -> float:
        return max(v for e, v in CONDUCTIVITES_TYPIQUES.values() if e is etat)

    assert plus_faible(Etat.SOLIDE) >= plus_forte(Etat.LIQUIDE)
    assert plus_forte(Etat.LIQUIDE) >= plus_faible(Etat.LIQUIDE) > plus_faible(Etat.GAZ)


def test_l_equation_de_fourier_kirchhoff_et_ses_cas_particuliers() -> None:
    """Planches 14 et 16 : `dt/dtau = a laplacien(t)`, puis `laplacien(t) = 0`."""
    a = sp.Symbol("a", positive=True)
    assert diffusivite(380.0, 8900.0, 385.0) == pytest.approx(380.0 / (8900.0 * 385.0))
    mode = sp.exp(-2 * a * INSTANT) * sp.sin(X) * sp.cos(Y)
    assert residu_de_fourier_kirchhoff(mode, a) == 0
    harmonique = X**2 - Y**2 + 3 * Z
    assert residu_de_fourier_kirchhoff(harmonique, a) == 0
    assert residu_de_fourier_kirchhoff(X**2, a) != 0


def test_les_conditions_aux_limites_de_la_planche_15() -> None:
    """Neumann, Fourier, et le contact parfait entre deux solides."""
    epaisseur, conductivite = sp.Rational(1, 5), sp.Integer(2)
    profil = 80 - 300 * X
    sortant = -conductivite * sp.diff(profil, X)
    assert sortant == 600
    surface = profil.subs(X, epaisseur)
    echange = sp.Integer(30)
    fluide = surface - sortant / echange
    assert sortant == echange * (surface - fluide)
    interfaces, flux = temperatures_d_interface(
        (Couche(0.2, 2.0), Couche(0.1, 0.5)), 80.0, -100.0
    )
    assert flux == pytest.approx(600.0)
    assert interfaces == pytest.approx([20.0])
    pentes = (-(80.0 - 20.0) / 0.2, -(20.0 + 100.0) / 0.1)
    assert -2.0 * pentes[0] == pytest.approx(-0.5 * pentes[1])


def test_la_plaque_et_ses_deux_fluides() -> None:
    """Planches 17 et 18 : `phi = (t_i - t_e) / (1/K_i + e/lambda + 1/K_e)`."""
    assert flux_de_plaque(1.5, 0.3, 50.0, 10.0) == pytest.approx(200.0)
    flux = flux_entre_deux_fluides(
        (Couche(0.3, 1.5),), interieur=(10.0, 50.0), exterieur=(25.0, 10.0)
    )
    assert flux == pytest.approx(40.0 / (0.1 + 0.2 + 0.04))
    assert 50.0 - flux / 10.0 > 10.0 + flux / 25.0


def test_le_tube_et_son_profil_logarithmique() -> None:
    """Planches 20 a 22 : `d/dr (r dt/dr) = 0`, `Phi = 2 pi lambda dt / ln(r_2/r_1)`."""
    rayon = sp.Symbol("r", positive=True)
    profil = profil_de_tube(rayon, (2, 5), (90, 30))
    assert sp.simplify(sp.diff(rayon * sp.diff(profil, rayon), rayon)) == 0
    lineique = -3 * sp.diff(profil, rayon) * 2 * sp.pi * rayon
    assert sp.simplify(lineique - 2 * sp.pi * 3 * 60 / sp.log(sp.Rational(5, 2))) == 0
    tube = Tube(2.0, 5.0, 3.0)
    assert flux_lineique_de_tubes((tube,), 90.0, 30.0) == pytest.approx(float(lineique))


def test_la_conductivite_variable_se_prend_a_la_temperature_moyenne() -> None:
    """Planche 23 : `phi = (t_1 - t_2)/e (a + b (t_1 + t_2)/2)`, et son graphe."""
    assert conductivite_moyenne(0.0326, 2.32e-4, 100.0, 20.0) == pytest.approx(
        0.0326 + 2.32e-4 * 60.0
    )
    table = ((0.0, 0.0326), (60.0, 0.0465), (100.0, 0.0558))
    pentes = ((0.0465 - 0.0326) / 60.0, (0.0558 - 0.0465) / 40.0)
    assert pentes[0] == pytest.approx(pentes[1], rel=0.01)
    assert conductivite_interpolee(table, 60.0) == 0.0465


TABLE = ((0.0, 0.0326), (60.0, 0.0465), (100.0, 0.0558))
TUYAU = TuyauCalorifuge(
    acier=Tube(0.0255, 0.029, 46.5),
    rayon_de_l_isolant=0.125,
    table=TABLE,
    echanges=(3500.0, 8.0),
)


def test_l_exercice_du_tuyau_calorifuge_se_referme_sur_lui_meme() -> None:
    """Planche 25 : eau a 150 degres, ambiance a 20 ; `lambda_is` a sa moyenne.

    Les planches ne donnent pas la reponse. Le flux trouve est donc
    verifie par ses propres chutes de temperature, et par l'isolant, pris
    a la moyenne des deux faces que ce flux lui donne.
    """
    flux, isolant = TUYAU.flux_lineique(150.0, 20.0)
    acier = Tube(0.0255, 0.029, 46.5)
    coquille = Tube(0.029, 0.125, isolant)
    face_chaude = (
        150.0 - flux / (2 * math.pi * 0.0255 * 3500.0) - flux * acier.resistance()
    )
    face_froide = 20.0 + flux / (2 * math.pi * 0.125 * 8.0)
    assert face_chaude - face_froide == pytest.approx(flux * coquille.resistance())
    moyenne = (face_chaude + face_froide) / 2
    assert isolant == pytest.approx(conductivite_interpolee(TABLE, moyenne), rel=1e-6)
    direct = flux_lineique_de_tubes(
        (acier, coquille), 150.0, 20.0, interieur=3500.0, exterieur=8.0
    )
    assert flux == pytest.approx(direct)
    assert 25.0 < flux < 30.0
    assert face_chaude == pytest.approx(150.0, abs=0.1)


def test_l_eau_se_refroidit_exponentiellement_le_long_du_tuyau() -> None:
    """Planche 25 : `q c dt/dx = -Phi`, et le flux suit l'ecart a l'ambiance."""
    flux, _ = TUYAU.flux_lineique(150.0, 20.0)
    capacite = 0.1 * 4300.0
    resistance = (150.0 - 20.0) / flux
    pas = 1e-3
    proche = temperature_le_long_du_tuyau(
        pas, (150.0, 20.0), capacite=capacite, resistance=resistance
    )
    assert (150.0 - proche) / pas == pytest.approx(flux / capacite, rel=1e-4)
    longueur = capacite * resistance
    loin = temperature_le_long_du_tuyau(
        longueur, (150.0, 20.0), capacite=capacite, resistance=resistance
    )
    assert loin - 20.0 == pytest.approx(130.0 / math.e)
    with pytest.raises(ValueError, match="capacite nulle ou negative"):
        temperature_le_long_du_tuyau(1.0, (150.0, 20.0), capacite=0.0, resistance=1.0)
