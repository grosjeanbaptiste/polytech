"""Mathematiques appliquees, 8.6 et 10.2 a 10.6 : ce que Fourier laissait au test.

Source : `Syllabus_Mathematiques_appliquees_2016_2017.pdf`.

- 8.6 (pages 149 et 150) : les trois ecritures de la serie -- complexe,
  en cosinus et sinus, en amplitude et phase -- et Parseval sous sa forme
  complexe, `somme |c_n|^2 = ||f||^2` ;
- exemple 8.3 et remarque 8.4 (pages 150 et 151) : `f = t^2` sur `[0, 1[`.
  La valeur de la serie en `t = 0` donne `somme 1/n^2 = pi^2/6`, puis
  Parseval `somme 1/n^4 = pi^4/90` ;
- 10.2.1 (page 172) : `c_n = F_T(omega_n) / T` ; (10.7), `F(0)` est
  l'integrale de `f` ;
- 10.2.2 (page 173) : la porte `Pi(t)`, de largeur 1 ;
- 10.2.3 : la convention en frequence, `omega = 2 pi nu` ;
- 10.6.3 : la modulation par un cosinus, par un sinus ;
- 10.6.5 : la convolution sur la droite, `F[f * g] = F G`, et
  `F[f g] = (1 / 2 pi) F * G`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.fourier.droite import (
    FREQUENCE,
    coefficient_depuis_la_transformee,
    convolution_sur_la_droite,
    en_frequence,
    image_du_produit,
    par_un_cosinus,
    par_un_sinus,
    porte,
    valeur_a_l_origine,
)
from polytech.maths.domain.fourier.formes import (
    forme_complexe,
    forme_en_amplitude_et_phase,
    identite_de_parseval,
    identite_en_un_point,
    parseval_complexe,
)
from polytech.maths.domain.fourier.series import SerieDeFourier
from polytech.maths.domain.fourier.transformee import (
    PULSATION as W,
)
from polytech.maths.domain.fourier.transformee import (
    integrale_sur_la_droite,
    transformee_de_fourier,
)
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
pi = sp.pi
N = sp.Symbol("n", integer=True, positive=True)
CARRE = SerieDeFourier(X**2, 0, 1)
"""L'exemple 8.3."""
POINTS = (R(1, 7), R(2, 5), R(5, 6))


def _proches(premiere: sp.Expr, seconde: sp.Expr) -> bool:
    ecart = premiere - seconde
    return all(abs(complex(ecart.subs(X, p).evalf())) < 1e-12 for p in POINTS)


def test_les_trois_ecritures_de_la_serie_sont_la_meme_fonction() -> None:
    reelle = CARRE.somme_partielle(3)
    assert _proches(forme_complexe(CARRE, 3), reelle)
    assert _proches(forme_en_amplitude_et_phase(CARRE, 3), reelle)


def test_la_forme_complexe_porte_les_rangs_negatifs() -> None:
    forme = forme_complexe(CARRE, 1)
    assert forme.coeff(sp.exp(-2 * sp.I * pi * X)) == sp.conjugate(CARRE.coefficient(1))


def test_parseval_complexe_est_la_somme_tronquee_de_la_serie_reelle() -> None:
    assert sp.simplify(parseval_complexe(CARRE, 4) - CARRE.energie(4)) == 0


def test_parseval_complexe_d_une_fonction_a_valeurs_complexes() -> None:
    # e^(2 j pi x) : un seul coefficient, c_1 = 1, et ||f||^2 = 1.
    harmonique = SerieDeFourier(sp.exp(2 * sp.I * pi * X), 0, 1)
    assert parseval_complexe(harmonique, 2) == 1


def test_la_remarque_8_4_somme_des_inverses_des_carres() -> None:
    terme, somme = identite_en_un_point(CARRE, 0, N)
    assert sp.simplify(terme - 1 / (pi**2 * N**2)) == 0
    assert somme == R(1, 6)
    assert sp.summation(terme, (N, 1, sp.oo)) == somme
    assert pi**2 * somme == pi**2 / 6


def test_parseval_donne_la_somme_des_inverses_des_puissances_quatriemes() -> None:
    terme, somme = identite_de_parseval(CARRE, N)
    assert somme == R(1, 5) - R(1, 9)
    # terme = A / n^4 + B / n^2, et somme 1/n^2 = pi^2 / 6 (remarque 8.4).
    inverse = sp.Symbol("m")
    polynome = sp.Poly(sp.expand(terme).subs(N, 1 / inverse), inverse)
    devant_n4, devant_n2 = (
        polynome.coeff_monomial(inverse**4),
        polynome.coeff_monomial(inverse**2),
    )
    assert polynome.as_expr() == devant_n4 * inverse**4 + devant_n2 * inverse**2
    assert sp.simplify((somme - devant_n2 * pi**2 / 6) / devant_n4) == pi**4 / 90
    assert sp.summation(terme, (N, 1, sp.oo)) == somme


def test_la_porte_et_sa_transformee() -> None:
    assert [porte().subs(X, x) for x in (-1, R(-1, 4), 0, R(1, 2), 1)] == [
        0,
        1,
        1,
        1,
        0,
    ]
    image = transformee_de_fourier(porte())
    assert sp.simplify(image - 2 * sp.sin(W / 2) / W) == 0
    assert valeur_a_l_origine(image) == 1
    assert integrale_sur_la_droite(porte(), X) == 1


@pytest.mark.parametrize("periode", [2, 4, 8])
def test_les_coefficients_de_la_porte_repetee(periode: int) -> None:
    image = transformee_de_fourier(porte())
    serie = SerieDeFourier(porte(), -R(periode, 2), periode)
    for rang in (1, 2, 3):
        attendu = sp.sin(pi * rang / periode) / (pi * rang)
        obtenu = coefficient_depuis_la_transformee(image, periode, rang)
        assert sp.simplify(obtenu - attendu) == 0
        assert sp.simplify(serie.coefficient(rang) - attendu) == 0


def test_une_largeur_nulle_n_est_pas_une_porte() -> None:
    with pytest.raises(ValueError, match="strictement positive"):
        porte(0)


def test_la_convolution_de_deux_gaussiennes_est_une_gaussienne() -> None:
    gaussienne = sp.exp(-(X**2))
    convolee = convolution_sur_la_droite(gaussienne, gaussienne)
    assert sp.simplify(convolee - sp.sqrt(pi / 2) * sp.exp(-(X**2) / 2)) == 0


def test_la_convolution_ajoute_les_retards() -> None:
    # Deux gaussiennes centrees en 1 et en 2 : les retards s'ajoutent.
    premiere, seconde = sp.exp(-((X - 1) ** 2)), sp.exp(-((X - 2) ** 2))
    convolee = convolution_sur_la_droite(premiere, seconde)
    assert sp.simplify(convolee - sp.sqrt(pi / 2) * sp.exp(-((X - 3) ** 2) / 2)) == 0


def test_la_transformee_d_une_convolution_est_le_produit() -> None:
    gaussienne = sp.exp(-(X**2))
    image = transformee_de_fourier(gaussienne)
    convolee = convolution_sur_la_droite(gaussienne, gaussienne)
    assert sp.simplify(transformee_de_fourier(convolee) - image**2) == 0


def test_la_transformee_d_un_produit_est_une_convolution_des_images() -> None:
    gaussienne = sp.exp(-(X**2))
    image = transformee_de_fourier(gaussienne)
    directe = transformee_de_fourier(gaussienne**2)
    assert sp.simplify(image_du_produit(image, image) - directe) == 0


def _memes_images(premiere: sp.Expr, seconde: sp.Expr) -> bool:
    ecart = premiere - seconde
    pulsations = (R(-2), R(3, 10), R(2))
    return all(abs(complex(ecart.subs(W, w).evalf())) < 1e-12 for w in pulsations)


def test_la_modulation_par_un_cosinus_et_par_un_sinus() -> None:
    f = sp.exp(-(X**2))
    image = transformee_de_fourier(f)
    directe = transformee_de_fourier(f * sp.cos(3 * X))
    assert _memes_images(par_un_cosinus(image, 3), directe)
    assert not _memes_images(par_un_sinus(image, 3), directe)
    directe = transformee_de_fourier(f * sp.sin(3 * X))
    assert _memes_images(par_un_sinus(image, 3), directe)


def test_la_convention_en_frequence_remplace_omega_par_deux_pi_nu() -> None:
    image = transformee_de_fourier(sp.exp(-pi * X**2))
    assert sp.simplify(en_frequence(image) - sp.exp(-pi * FREQUENCE**2)) == 0


def test_le_produit_de_deux_fonctions_dont_l_une_est_decalee() -> None:
    premiere, seconde = sp.exp(-((X - 1) ** 2)), sp.exp(-(X**2))
    images = [transformee_de_fourier(f) for f in (premiere, seconde)]
    directe = transformee_de_fourier(premiere * seconde)
    assert _memes_images(image_du_produit(*images), directe)
