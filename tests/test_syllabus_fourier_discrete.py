"""Syllabus de mathematiques appliquees, sections 10.7 et 10.9 a 10.12.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 181 a 189. Les distributions (Dirac, fonctions test, espace de
Schwartz), les transformees de la constante et du cosinus, le peigne de
Dirac, la diffraction de sources ponctuelles ; la transformee de Fourier
discrete et son exemple, la transformee a deux dimensions, le signal
bruite du 10.12.2.

Une erreur du syllabus est etablie : la fonction constante 1 n'appartient
pas a l'espace de Schwartz (page 184).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.maths.domain.fourier.deux_dimensions import (
    PULSATIONS,
    transformee_a_deux_dimensions,
)
from polytech.maths.domain.fourier.discrete import (
    echantillonner,
    raies_dominantes,
    spectre_unilateral,
    transformee_discrete,
    transformee_discrete_inverse,
)
from polytech.maths.domain.fourier.distributions import (
    FONCTION_TEST,
    appliquer_le_dirac,
    appliquer_le_peigne,
    est_a_decroissance_rapide,
    transformee_de_la_constante,
    transformee_des_impulsions,
    transformee_du_cosinus,
    transformee_du_sinus,
)
from polytech.maths.domain.fourier.transformee import PULSATION as W
from polytech.maths.domain.fourier.transformee import (
    integrale_sur_la_droite,
    transformee_inverse,
)
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
GAUSSIENNE = sp.exp(-(X**2))
EXEMPLE = (2.0, -1.618, 0.618, 0.618, -1.618, 2.0, -1.618, 0.618, 0.618, -1.618)


def _cosinus_de_frequence_deux(t: float) -> float:
    return 2 * math.cos(2 * math.pi * 2 * t)


def test_la_distribution_de_dirac_evalue_la_fonction_test() -> None:
    """10.7.2 : `<delta_a, phi> = phi(a)`, fonctionnelle lineaire."""
    assert appliquer_le_dirac(0, GAUSSIENNE) == 1
    assert appliquer_le_dirac(2, GAUSSIENNE) == sp.exp(-4)
    somme = appliquer_le_dirac(2, 3 * GAUSSIENNE + X)
    assert somme == 3 * appliquer_le_dirac(2, GAUSSIENNE) + appliquer_le_dirac(2, X)


def test_la_fonction_test_du_syllabus_est_lisse_et_a_support_compact() -> None:
    """10.7.2 : `exp(1 / (t^2 - 1))` sur `]-1, 1[`, nulle ailleurs.

    Le syllabus n'ecrit que `t >= 1` pour la branche nulle ; il faut `|t| >= 1`.
    """
    assert FONCTION_TEST.subs(X, 0) == sp.exp(-1)
    assert FONCTION_TEST.subs(X, 2) == 0
    assert FONCTION_TEST.subs(X, -2) == 0
    interieur = sp.exp(1 / (X**2 - 1))
    # la formule interieure seule (branche nulle pour `t >= 1` uniquement)
    # ne s'annule pas pour `t <= -1` : en t = -2, e^(1/3)
    assert interieur.subs(X, -2) == sp.exp(sp.Rational(1, 3))
    for ordre in range(4):
        assert sp.limit(sp.diff(interieur, X, ordre), X, 1, "-") == 0
        assert sp.limit(sp.diff(interieur, X, ordre), X, -1, "+") == 0


def test_la_gaussienne_est_dans_l_espace_de_schwartz_sans_etre_a_support_compact() -> (
    None
):
    """10.7.3 : `x^n phi^(p)` reste borne ; mais `phi` ne s'annule jamais."""
    for puissance in range(4):
        for ordre in range(3):
            assert est_a_decroissance_rapide(GAUSSIENNE, puissance, ordre)
    assert GAUSSIENNE.subs(X, 10) != 0


def test_la_constante_n_est_pas_dans_l_espace_de_schwartz() -> None:
    """Page 184 : « il est facile de verifier que [1] appartient bien a `S(R)` » -- non.

    `x . 1` n'est pas borne. La constante est une distribution temperee, un
    element de `S'(R)` : c'est a ce titre qu'elle a une transformee.
    """
    assert est_a_decroissance_rapide(sp.Integer(1), 0, 0)
    assert not est_a_decroissance_rapide(sp.Integer(1), 1, 0)


def test_la_transformee_de_la_constante_est_un_dirac() -> None:
    """10.7.4 : `F(w) = 2 pi delta(w)`, et l'inverse redonne 1."""
    image = transformee_de_la_constante(1)
    assert image == 2 * sp.pi * sp.DiracDelta(W)
    assert transformee_inverse(image) == 1
    assert transformee_inverse(transformee_de_la_constante(5)) == 5


def test_l_integrale_du_dirac_contre_l_exponentielle_porte_sur_omega() -> None:
    """Le syllabus ecrit `1 = int delta(w) e^(jwt) dt` ; l'integrale est en `dw`."""
    noyau = sp.DiracDelta(W) * sp.exp(sp.I * W * X)
    assert sp.integrate(noyau, (W, -sp.oo, sp.oo)) == 1


def test_la_transformee_du_cosinus_a_deux_pics() -> None:
    """10.7.5 : `pi (delta(w - w0) + delta(w + w0))`, a `+-w0` de l'origine."""
    w0 = sp.Symbol("omega_0", positive=True)
    image = transformee_du_cosinus(w0)
    assert image == sp.pi * (sp.DiracDelta(W - w0) + sp.DiracDelta(W + w0))
    assert sp.simplify(transformee_inverse(image) - sp.cos(w0 * X)) == 0


def test_la_transformee_du_sinus() -> None:
    """Meme calcul avec `sin = (e^(j w0 t) - e^(-j w0 t)) / 2j`."""
    w0 = sp.Symbol("omega_0", positive=True)
    image = transformee_du_sinus(w0)
    assert sp.simplify(transformee_inverse(image) - sp.sin(w0 * X)) == 0


def test_le_peigne_de_dirac_echantillonne() -> None:
    """10.7.6 : `somme delta(t - nT)` applique a `phi` rend `somme phi(nT)`."""
    periode = R(1, 2)
    attendu = sum(GAUSSIENNE.subs(X, n * periode) for n in range(-3, 4))
    assert appliquer_le_peigne(periode, 3, GAUSSIENNE) == attendu
    assert appliquer_le_peigne(periode, 0, GAUSSIENNE) == 1


def test_la_diffraction_de_sources_ponctuelles() -> None:
    """10.7.7 : une source en 0 donne 1, en `a`, `e^(-jqa)` ; en `+-a`, `2 cos(qa)`."""
    a = sp.Symbol("a", positive=True)
    assert transformee_des_impulsions((0,)) == 1
    assert transformee_des_impulsions((a,)) == sp.exp(-sp.I * W * a)
    deux = transformee_des_impulsions((a, -a))
    assert sp.simplify(deux - 2 * sp.cos(W * a)) == 0


def test_l_intensite_ne_change_pas_par_translation_mais_deux_sources_interferent() -> (
    None
):
    """`I = |A|^2` : 1 pour une source ou qu'elle soit, `4 cos^2(qa)` pour deux."""
    a = sp.Symbol("a", positive=True)

    def intensite(positions: tuple[object, ...]) -> sp.Expr:
        amplitude = transformee_des_impulsions(positions)
        return sp.simplify(sp.expand_complex(amplitude * sp.conjugate(amplitude)))

    assert intensite((0,)) == intensite((a,)) == 1
    assert sp.simplify(intensite((a, -a)) - 4 * sp.cos(W * a) ** 2) == 0


def test_les_echantillons_de_l_exemple_10_9_1() -> None:
    """`2 cos(2 pi 2 t)` tous les `t_0 = 0,2` : dix valeurs, a trois decimales."""
    echantillons = echantillonner(_cosinus_de_frequence_deux, 0.2, 10)
    assert echantillons == pytest.approx(EXEMPLE, abs=5e-4)


def test_la_transformee_discrete_de_l_exemple_10_9_1() -> None:
    """(10.20) : `X = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]`."""
    echantillons = echantillonner(_cosinus_de_frequence_deux, 0.2, 10)
    spectre = transformee_discrete(echantillons)
    assert spectre == pytest.approx([0, 0, 0, 0, 1, 0, 1, 0, 0, 0], abs=1e-12)


def test_la_transformee_discrete_inverse_retrouve_les_echantillons() -> None:
    """(10.21) : `x[n] = somme X[k] e^(j 2 pi k n / N)`, exactement."""
    echantillons = echantillonner(_cosinus_de_frequence_deux, 0.2, 10)
    retour = transformee_discrete_inverse(transformee_discrete(echantillons))
    assert retour == pytest.approx(echantillons, abs=1e-12)


def test_la_transformee_discrete_est_celle_de_la_transformee_rapide() -> None:
    """10.11 : la FFT calcule la meme somme, au facteur `1/N` de (10.20) pres."""
    alea = np.random.default_rng(7).normal(size=64)
    assert transformee_discrete(alea) == pytest.approx(np.fft.fft(alea) / 64, abs=1e-12)


def test_les_coefficients_discrets_sont_periodiques() -> None:
    """Page 187 : `w_(k+N)[n] = w_k[n]`, donc `X[k + N] = X[k]`."""
    nombre, n = 10, 3
    for k in range(nombre):
        proche = np.exp(2j * np.pi * k * n / nombre)
        loin = np.exp(2j * np.pi * (k + nombre) * n / nombre)
        assert loin == pytest.approx(proche, abs=1e-12)


def test_le_signal_continu_se_retrouve_par_sa_serie() -> None:
    """Page 187 : `X[4] e^(j 2 pi 4 f0 t) + X[6] e^(-j 2 pi 4 f0 t)` est le signal.

    La periode implicite est `N t_0 = 2`, donc `f0 = 1/2` et la frequence 2.
    """
    f0 = 1 / (10 * 0.2)
    for instant in (0.0, 0.13, 0.77):
        serie = 2 * math.cos(2 * math.pi * 4 * f0 * instant)
        assert serie == pytest.approx(_cosinus_de_frequence_deux(instant), abs=1e-12)


def test_une_frequence_qui_ne_tombe_pas_sur_la_grille_s_etale() -> None:
    """Page 188 : a la frequence 2,2, « beaucoup plus de composantes apparaissent »."""

    def decale(t: float) -> float:
        return 2 * math.cos(2 * math.pi * 2.2 * t)

    echantillons = echantillonner(decale, 0.2, 10)
    spectre = transformee_discrete(echantillons)
    assert np.sum(np.abs(spectre) > 1e-3) == 10
    retour = transformee_discrete_inverse(spectre)
    assert retour == pytest.approx(echantillons, abs=1e-12)


def test_le_spectre_du_signal_synthetique_du_10_12_2() -> None:
    """`0,7 sin(2 pi 50 t) + sin(2 pi 120 t)`, `Fs = 1000`, `L = 1000` : deux raies."""
    cadence, nombre = 1000, 1000
    instants = np.arange(nombre) / cadence
    signal = 0.7 * np.sin(2 * np.pi * 50 * instants) + np.sin(
        2 * np.pi * 120 * instants
    )
    frequences, amplitudes = spectre_unilateral(signal, cadence)
    assert frequences[50] == pytest.approx(50)
    assert amplitudes[50] == pytest.approx(0.7, abs=1e-9)
    assert amplitudes[120] == pytest.approx(1.0, abs=1e-9)
    assert sorted(raies_dominantes(signal, cadence, 2)) == pytest.approx([50, 120])


def test_les_deux_raies_emergent_du_bruit() -> None:
    """10.12.2 : sous un bruit d'ecart-type 2, le spectre montre encore 50 et 120 Hz."""
    cadence, nombre = 1000, 1000
    instants = np.arange(nombre) / cadence
    signal = 0.7 * np.sin(2 * np.pi * 50 * instants) + np.sin(
        2 * np.pi * 120 * instants
    )
    bruite = signal + 2 * np.random.default_rng(0).normal(size=nombre)
    assert np.max(np.abs(bruite)) > 2 * np.max(np.abs(signal))
    assert sorted(raies_dominantes(bruite, cadence, 2)) == pytest.approx([50, 120])


def test_un_signal_vide_n_a_pas_de_transformee() -> None:
    with pytest.raises(ValueError, match="aucun echantillon"):
        transformee_discrete(np.array([]))


def test_la_transformee_a_deux_dimensions_d_une_gaussienne() -> None:
    """10.10 : `e^(-x1^2 - x2^2)` devient `pi e^(-(w1^2 + w2^2) / 4)`."""
    x1, x2 = sp.symbols("x1 x2", real=True)
    w1, w2 = PULSATIONS
    image = transformee_a_deux_dimensions(sp.exp(-(x1**2) - x2**2), (x1, x2))
    assert sp.simplify(image - sp.pi * sp.exp(-(w1**2 + w2**2) / 4)) == 0


def test_une_fonction_separable_a_une_image_separable() -> None:
    """`f(x1) g(x2)` devient `F(w1) G(w2)`."""
    x1, x2 = sp.symbols("x1 x2", real=True)
    w1, w2 = PULSATIONS
    image = transformee_a_deux_dimensions(
        sp.exp(-(x1**2)) * x2 * sp.exp(-(x2**2)), (x1, x2)
    )
    premiere = sp.sqrt(sp.pi) * sp.exp(-(w1**2) / 4)
    seconde = -sp.I * sp.sqrt(sp.pi) * w2 * sp.exp(-(w2**2) / 4) / 2
    assert sp.simplify(image - premiere * seconde) == 0


def test_la_decroissance_doit_etre_rapide_des_deux_cotes() -> None:
    """`e^(-x)` s'eteint a droite et explose a gauche : elle n'est pas dans `S(R)`."""
    assert not est_a_decroissance_rapide(sp.exp(-X), 0, 0)


def test_l_integrale_sur_la_droite_traite_les_morceaux_et_les_impulsions() -> None:
    """Une exponentielle causale s'integre sur sa demi-droite ; un Dirac, tel quel."""
    causale = sp.Piecewise((sp.exp(-X), X >= 0), (0, True))
    assert integrale_sur_la_droite(causale, X) == 1
    assert integrale_sur_la_droite(sp.Abs(X) * sp.exp(-(X**2)), X) == 1
    assert integrale_sur_la_droite(X**2 * sp.DiracDelta(X - 2), X) == 4
