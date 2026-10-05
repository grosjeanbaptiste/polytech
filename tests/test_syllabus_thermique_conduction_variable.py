"""Le syllabus de thermique, tome 1 : la conduction variable dans le temps.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 71 a 84
(premiere partie, chapitre 4, exercice I.4.3), lues sur les pages
rendues.

Les oracles : (4.1) a (4.9), les trois sollicitations du solide
semi-infini, la temperature de contact, la trempe etagee et l'integrale
de Duhamel, puis l'exercice I.4.3 -- 49 900, un rapport de 0,003 et un
facteur de 0,084 a un millimetre.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.thermique.domain.conduction.plaque_variable import (
    PlaqueEnEchelon,
    produit_des_plaques,
)
from polytech.thermique.domain.conduction.semi_infini import (
    amortissement_en_profondeur,
    attenuation_a_la_surface,
    sous_un_echelon_de_flux,
    sous_un_echelon_de_surface,
    sous_un_fluide,
    temperature_de_contact,
)
from polytech.thermique.domain.conduction.superposition import (
    integrale_de_duhamel,
    trempe_etagee,
)
from polytech.thermique.domain.inertie import SolideIsotherme
from polytech.thermique.domain.transitoire import CorpsTransitoire

X, TAU = sp.symbols("x tau", positive=True)
A, LAMBDA = sp.symbols("a lambda", positive=True)


def _residu_de_diffusion(champ: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.diff(champ, TAU) - A * sp.diff(champ, X, 2))


PLAQUE = PlaqueEnEchelon(biot=2.0)


def test_les_valeurs_propres_de_la_plaque() -> None:
    """Page 73 : `cot(delta_n) = delta_n lambda / (K l)`, une par demi-tour."""
    racines = PLAQUE.valeurs_propres(4)
    for rang, racine in enumerate(racines):
        assert rang * math.pi < racine < rang * math.pi + math.pi / 2
        assert 1 / math.tan(racine) == pytest.approx(racine / 2.0)
    with pytest.raises(ValueError, match="nombre de Biot nul ou negatif"):
        PlaqueEnEchelon(biot=0.0)


def test_chaque_terme_de_la_serie_resout_l_equation_et_ses_conditions() -> None:
    """(4.3) a (4.5) : diffusion, symetrie au centre, condition de Fourier.

    (4.7) imprime l'exposant « -delta_n (a tau / l^2) » ; c'est
    `delta_n^2`, comme dans le mode fondamental de la page suivante.
    """
    racine, demi, echange = (
        PLAQUE.valeurs_propres(1)[0],
        sp.Rational(1, 2),
        sp.Integer(4),
    )
    conductivite = echange * demi / 2
    terme = sp.exp(-(racine**2) * A * TAU / demi**2) * sp.cos(racine * X / demi)
    assert _residu_de_diffusion(terme) == 0
    assert sp.diff(terme, X).subs(X, 0) == 0
    au_bord = -conductivite * sp.diff(terme, X) - echange * terme
    assert float(au_bord.subs({X: demi, TAU: 0})) == pytest.approx(0.0, abs=1e-9)
    faux = sp.exp(-racine * A * TAU / demi**2) * sp.cos(racine * X / demi)
    assert _residu_de_diffusion(faux) != 0


def test_la_serie_part_de_l_ecart_initial() -> None:
    """(4.6) : `theta = theta_0` partout en `tau = 0`."""
    for abscisse in (0.0, 0.4, 0.8):
        assert PLAQUE.ecart_reduit(0.0, abscisse, termes=400) == pytest.approx(
            1.0, abs=5e-3
        )
    assert PLAQUE.ecart_reduit(10.0, 0.0) == pytest.approx(0.0, abs=1e-3)


def test_le_mode_fondamental_reste_seul() -> None:
    """Page 74 : `theta / theta_0 ~ A_1 exp(-delta_1^2 Fo) cos(delta_1 x / l)`."""
    (premiere,) = PLAQUE.valeurs_propres(1)
    amplitude = PLAQUE.amplitude(premiere)
    attendue = (
        2 * math.sin(premiere) / (premiere + math.sin(premiere) * math.cos(premiere))
    )
    assert amplitude == pytest.approx(attendue)
    fondamental = amplitude * math.exp(-(premiere**2) * 1.0) * math.cos(premiere * 0.3)
    assert PLAQUE.ecart_reduit(1.0, 0.3) == pytest.approx(fondamental, rel=1e-4)


def test_les_nombres_de_fourier_et_de_biot() -> None:
    """Page 74 : `Fo = tau / tau_c`, `tau_c = l^2 / a` ; `Bi = K l / lambda`."""
    corps = CorpsTransitoire(100.0, 0.02, 1.0, 2.0e6)
    assert corps.biot() == pytest.approx(2.0)
    assert corps.fourier(800.0) == pytest.approx(1.0)
    assert corps.fourier(800.0) == pytest.approx(
        800.0 / (0.02**2 / corps.diffusivite())
    )


def test_le_parallelepipede_est_un_produit_de_trois_plaques() -> None:
    """Page 74 : `theta / theta_0 = f_1(2A) f_1(2B) f_1(2C)`."""
    plaques = (PlaqueEnEchelon(2.0), PlaqueEnEchelon(1.0), PlaqueEnEchelon(4.0))
    points = ((0.5, 0.0), (0.2, 0.5), (1.0, 1.0))
    attendu = math.prod(
        p.ecart_reduit(*point) for p, point in zip(plaques, points, strict=True)
    )
    assert produit_des_plaques(plaques, points) == pytest.approx(attendu)
    assert (
        0
        < attendu
        < min(p.ecart_reduit(*point) for p, point in zip(plaques, points, strict=True))
    )


T0, TF = sp.symbols("t_0 t_f", real=True)


def test_le_solide_semi_infini_sous_un_echelon_de_surface() -> None:
    """(4.9) : `t = t_0 + (t_f - t_0) ferc(x / (2 racine(a tau)))`."""
    champ = sous_un_echelon_de_surface(X, TAU, A, initiale=T0, surface=TF)
    assert _residu_de_diffusion(champ) == 0
    assert sp.limit(champ, X, 0) == TF
    assert sp.limit(champ, X, sp.oo) == T0
    penetrant = -LAMBDA * sp.diff(champ, X).subs(X, 0)
    assert sp.simplify(penetrant - LAMBDA * (TF - T0) / sp.sqrt(sp.pi * A * TAU)) == 0


def test_le_solide_semi_infini_au_contact_d_un_fluide() -> None:
    """Page 76, cas 2 : la condition de Fourier en `x = 0`."""
    echange = sp.Symbol("K", positive=True)
    champ = sous_un_fluide(X, TAU, A, echange / LAMBDA, initiale=T0, fluide=TF)
    assert _residu_de_diffusion(champ) == 0
    surface = sp.limit(champ, X, 0)
    entrant = -LAMBDA * sp.limit(sp.diff(champ, X), X, 0)
    assert sp.simplify(entrant - echange * (TF - surface)) == 0
    valeurs = {X: 0.01, TAU: 50.0, A: 1e-5, T0: 20.0, TF: 100.0, LAMBDA: 1.0}
    echelon = sous_un_echelon_de_surface(X, TAU, A, initiale=T0, surface=TF)
    tres_bon = champ.subs(echange, 1e7).subs(valeurs)
    assert float(tres_bon) == pytest.approx(float(echelon.subs(valeurs)), rel=1e-3)


def test_le_solide_semi_infini_sous_un_flux_impose() -> None:
    """Page 76, cas 3 : `-lambda dt/dx = phi_0` a la surface."""
    flux = sp.Symbol("phi_0", positive=True)
    champ = sous_un_echelon_de_flux(X, TAU, A, flux / LAMBDA, initiale=T0)
    assert _residu_de_diffusion(champ) == 0
    entrant = -LAMBDA * sp.limit(sp.diff(champ, X), X, 0)
    assert sp.simplify(entrant - flux) == 0
    surface = sp.limit(champ, X, 0)
    assert sp.simplify(surface - T0 - 2 * flux / LAMBDA * sp.sqrt(A * TAU / sp.pi)) == 0


def test_la_temperature_de_contact_de_deux_solides() -> None:
    """Page 77 : `(t_2 - t_m) / (t_m - t_1) = racine(e_1 / e_2)`, `e = l r c`."""
    cuivre, bois = 380 * 8900 * 385, 0.15 * 600 * 2000
    contact = temperature_de_contact((37.0, 1.0e6), (20.0, cuivre))
    assert (20.0 - contact) / (contact - 37.0) == pytest.approx(
        math.sqrt(1.0e6 / cuivre)
    )
    assert contact - 20.0 < 37.0 - contact
    sur_le_bois = temperature_de_contact((37.0, 1.0e6), (20.0, bois))
    assert sur_le_bois > contact
    assert temperature_de_contact((37.0, 1.0e6), (20.0, 1.0e6)) == pytest.approx(28.5)


def _echelon_isotherme(constante: float):  # noqa: ANN202 -- une fonction du temps
    return lambda duree: 1 - math.exp(-duree / constante) if duree > 0 else 0.0


def test_la_trempe_etagee_superpose_deux_echelons() -> None:
    """4.4.2 : `t = t_0 + (t_f1 - t_0) F_1(tau) + (t_f2 - t_f1) F_1(tau - tau_1)`."""
    reponse = _echelon_isotherme(10.0)
    avant = trempe_etagee(
        reponse, 5.0, initiale=600.0, bains=((0.0, 300.0), (20.0, 20.0))
    )
    assert avant == pytest.approx(600 - 300 * (1 - math.exp(-0.5)))
    longtemps = trempe_etagee(
        reponse, 500.0, initiale=600.0, bains=((0.0, 300.0), (20.0, 20.0))
    )
    assert longtemps == pytest.approx(20.0, abs=1e-6)
    a_la_bascule = trempe_etagee(
        reponse, 20.0, initiale=600.0, bains=((0.0, 300.0), (20.0, 20.0))
    )
    assert a_la_bascule == pytest.approx(600 - 300 * (1 - math.exp(-2.0)))


def test_l_integrale_de_duhamel_retrouve_la_rampe_du_chapitre_3() -> None:
    """4.4.3 : `t = t_0 + integrale de (dt_f/dtau*) F_1(tau - tau*)`.

    Pour un fluide qui monte en rampe, c'est la reponse de la figure 3.3.
    """
    constante, pente = 10.0, 2.0
    solide = SolideIsotherme(resistance=1.0, capacite=constante)
    par_duhamel = integrale_de_duhamel(
        _echelon_isotherme(constante), lambda _: pente, 30.0, initiale=15.0
    )
    rampe = solide.premier_ordre().reponse_a_la_rampe()
    attendue = 15.0 + pente * float(rampe.subs(sp.Symbol("t", nonnegative=True), 30.0))
    assert par_duhamel == pytest.approx(attendue, rel=1e-5)
    assert par_duhamel == pytest.approx(
        15 + 2 * 30 - 2 * 10 * (1 - math.exp(-3)), rel=1e-5
    )


def test_l_exercice_i_4_3_la_paroi_d_un_cylindre_de_moteur() -> None:
    """Fonte, `K = 500`, periode 0,03 s : la paroi ne suit pas les gaz.

    Le rapport vaut 0,00316, que le syllabus arrondit a 0,003 avant de le
    multiplier par 1600 : il annonce 4,8 degres, le produit en fait 5,1.
    """
    diffusivite, conductivite, echange, periode = 1.7e-5, 45.0, 500.0, 60 / 2000
    groupe = math.pi * conductivite**2 / (diffusivite * periode * echange**2)
    assert groupe == pytest.approx(49900, rel=2e-3)
    rapport = attenuation_a_la_surface(conductivite / echange, diffusivite, periode)
    assert rapport == pytest.approx((1 + 2 * math.sqrt(groupe) + 2 * groupe) ** -0.5)
    assert rapport == pytest.approx(0.003, abs=2e-4)
    assert rapport * 1600 == pytest.approx(5.05, abs=0.01)
    assert pytest.approx(4.8) == 0.003 * 1600


def test_l_oscillation_ne_penetre_qu_une_couche_mince() -> None:
    """`exp(-x racine(pi / (a tau_0)))` : 0,084 a un millimetre."""
    facteur = amortissement_en_profondeur(0.001, 1.7e-5, 0.03)
    assert facteur == pytest.approx(0.084, abs=0.001)
    profondeur = np.array([0.0, 0.001, 0.002])
    facteurs = [amortissement_en_profondeur(x, 1.7e-5, 0.03) for x in profondeur]
    assert facteurs[2] == pytest.approx(facteurs[1] ** 2)
    with pytest.raises(ValueError, match="periode nulle ou negative"):
        amortissement_en_profondeur(0.001, 1.7e-5, 0.0)


def test_un_bain_a_venir_n_agit_pas_encore() -> None:
    """La reponse d'un echelon n'existe qu'apres lui, quelle que soit sa formule."""
    sans_garde = lambda duree: 1 - math.exp(-duree / 10.0)  # noqa: E731 -- une formule
    bains = ((0.0, 300.0), (20.0, 20.0))
    avant = trempe_etagee(sans_garde, 5.0, initiale=600.0, bains=bains)
    assert avant == pytest.approx(600 - 300 * (1 - math.exp(-0.5)))


def test_l_integrale_de_duhamel_sous_un_fluide_qui_accelere() -> None:
    """`t_f = tau^2` : `t = tau^2 - 2 tau_c tau + 2 tau_c^2 (1 - exp(-tau / tau_c))`."""
    constante, duree = 10.0, 25.0
    par_duhamel = integrale_de_duhamel(
        _echelon_isotherme(constante), lambda instant: 2 * instant, duree, initiale=0.0
    )
    exacte = duree**2 - 2 * constante * duree
    exacte += 2 * constante**2 * (1 - math.exp(-duree / constante))
    assert par_duhamel == pytest.approx(exacte, rel=1e-6)


def test_le_saut_initial_s_ajoute_a_l_integrale() -> None:
    """Le premier terme, `(t_f(0) - t_0) F_1(tau)`."""
    reponse = _echelon_isotherme(10.0)
    seul = integrale_de_duhamel(reponse, lambda _: 0.0, 10.0, initiale=20.0, saut=80.0)
    assert seul == pytest.approx(20 + 80 * (1 - math.exp(-1)))
