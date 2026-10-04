"""Syllabus de mathematiques appliquees, sections 8.5.3 a 8.8 : les series de Fourier.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 148 a 154. Les fonctions periodiques, les coefficients `c_n`, `a_n`,
`b_n`, l'amplitude et la phase, la formule de Parseval, la valeur de la
serie en une discontinuite ; les exemples 8.3 a 8.6 et les sommes de
series numeriques qu'ils donnent.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.maths.domain.fourier.parite import (
    est_impaire,
    est_paire,
    prolongement_impair,
    prolongement_pair,
)
from polytech.maths.domain.fourier.periodiques import (
    est_periodique,
    integrale_sur_une_periode,
    periode_de_l_exponentielle,
)
from polytech.maths.domain.fourier.series import SerieDeFourier
from polytech.maths.domain.symboles import VARIABLE as T

# La variable des motifs est reelle et de signe quelconque : un motif se donne
# aussi bien sur `[-pi, pi]`, ou `|t|` ne se reduit pas a `t`.
R = sp.Rational
N = sp.Symbol("n", integer=True, positive=True)
PARABOLE = SerieDeFourier(T**2, 0, 1)
VALEUR_ABSOLUE = SerieDeFourier(sp.Abs(T), -sp.pi, 2 * sp.pi)
DENT_DE_SCIE = SerieDeFourier(T, -sp.pi, 2 * sp.pi)


def test_la_periode_d_une_exponentielle_imaginaire() -> None:
    """Definition 8.8 : `e^(lambda t)` avec `lambda = j w` a pour periode `2 pi / w`."""
    assert periode_de_l_exponentielle(3 * sp.I) == 2 * sp.pi / 3
    assert periode_de_l_exponentielle(-2 * sp.I) == sp.pi
    assert est_periodique(sp.exp(3 * sp.I * T), 2 * sp.pi / 3)
    assert not est_periodique(sp.exp(3 * sp.I * T), 1)


def test_une_exponentielle_reelle_n_est_pas_periodique() -> None:
    with pytest.raises(ValueError, match="pas imaginaire pur"):
        periode_de_l_exponentielle(1 + sp.I)


def test_l_integrale_sur_une_periode_ne_depend_pas_du_depart() -> None:
    """Propriete 8.5.1 : `int_x^(x+T) f(t) dt` est une constante."""
    x = sp.Symbol("x", real=True)
    fonction = sp.sin(T) ** 2 + sp.cos(3 * T)
    integrale = integrale_sur_une_periode(fonction, x, 2 * sp.pi)
    assert not integrale.has(x)
    assert integrale == sp.pi


def test_les_coefficients_complexes_de_l_exemple_8_3() -> None:
    """`f = t^2` sur `[0, 1[` : `c_n = 1/(2 pi^2 n^2) + j/(2 pi n)`, `c_0 = 1/3`."""
    attendu = 1 / (2 * sp.pi**2 * N**2) + sp.I / (2 * sp.pi * N)
    assert sp.simplify(PARABOLE.coefficient(N) - attendu) == 0
    assert PARABOLE.coefficient(0) == R(1, 3)
    assert PARABOLE.pulsation == 2 * sp.pi


def test_les_coefficients_reels_de_l_exemple_8_3() -> None:
    """`a_n = 1/(pi^2 n^2)`, `b_n = -1/(pi n)`, et `c_n = (a_n - j b_n)/2`."""
    assert sp.simplify(PARABOLE.a(N) - 1 / (sp.pi**2 * N**2)) == 0
    assert sp.simplify(PARABOLE.b(N) + 1 / (sp.pi * N)) == 0
    lien = (PARABOLE.a(N) - sp.I * PARABOLE.b(N)) / 2
    assert sp.simplify(PARABOLE.coefficient(N) - lien) == 0


def test_les_coefficients_d_une_fonction_reelle_sont_conjugues() -> None:
    """Page 149 : si `f` est reelle, `c_(-n) = conj(c_n)`."""
    for rang in (1, 2, 5):
        direct, oppose = PARABOLE.coefficient(rang), PARABOLE.coefficient(-rang)
        assert sp.simplify(oppose - sp.conjugate(direct)) == 0


def test_l_amplitude_et_la_phase() -> None:
    """`c_n = A_n e^(j phi_n) / 2`, avec `phi_n = arctan(-b_n / a_n)`."""
    for rang in (1, 3):
        amplitude, phase = PARABOLE.amplitude(rang), PARABOLE.phase(rang)
        reconstruit = amplitude * sp.exp(sp.I * phase) / 2
        assert (
            sp.simplify(sp.expand_complex(reconstruit - PARABOLE.coefficient(rang)))
            == 0
        )
        attendue = sp.atan(-PARABOLE.b(rang) / PARABOLE.a(rang))
        assert sp.simplify(phase - attendue) == 0


def test_la_somme_partielle_de_l_exemple_8_3() -> None:
    """`1/3 + somme (cos(2 pi n t)/(pi^2 n^2) - sin(2 pi n t)/(pi n))`."""
    attendue = R(1, 3) + sum(
        sp.cos(2 * sp.pi * k * T) / (sp.pi**2 * k**2)
        - sp.sin(2 * sp.pi * k * T) / (sp.pi * k)
        for k in (1, 2, 3)
    )
    assert sp.simplify(PARABOLE.somme_partielle(3) - attendue) == 0


def test_la_serie_prend_la_moyenne_des_limites_en_une_discontinuite() -> None:
    """Remarque 8.4 : en `t = 0`, `(f(0+) + f(0-))/2 = (0 + 1)/2`."""
    assert PARABOLE.valeur_en(0) == R(1, 2)
    assert PARABOLE.valeur_en(3) == R(1, 2)
    assert PARABOLE.valeur_en(R(1, 2)) == R(1, 4)
    assert PARABOLE.valeur_en(R(5, 2)) == R(1, 4)


def test_la_somme_des_inverses_des_carres() -> None:
    """Exemple 8.3 : `1/2 = 1/3 + somme 1/(pi^2 n^2)`, d'ou `somme 1/n^2 = pi^2/6`."""
    en_zero = PARABOLE.coefficient(0) + sp.Sum(PARABOLE.a(N), (N, 1, sp.oo)).doit()
    assert sp.simplify(en_zero - PARABOLE.valeur_en(0)) == 0
    assert sp.Sum(1 / N**2, (N, 1, sp.oo)).doit() == sp.pi**2 / 6


def test_la_formule_de_parseval_sur_l_exemple_8_3() -> None:
    """`||f||^2 = 1/5 = 1/9 + somme A_n^2 / 2`, d'ou `somme 1/n^4 = pi^4/90`."""
    assert PARABOLE.norme_au_carre() == R(1, 5)
    harmoniques = sp.Sum(sp.simplify(PARABOLE.amplitude(N) ** 2 / 2), (N, 1, sp.oo))
    assert sp.simplify(R(1, 9) + harmoniques.doit() - R(1, 5)) == 0
    assert sp.Sum(1 / N**4, (N, 1, sp.oo)).doit() == sp.pi**4 / 90


def test_l_energie_des_premiers_harmoniques_reste_sous_la_norme() -> None:
    """Parseval, vu comme l'inegalite de Bessel : les sommes croissent vers `1/5`."""
    energies = [PARABOLE.energie(ordre) for ordre in (1, 3, 6)]
    assert energies[0] < energies[1] < energies[2] < PARABOLE.norme_au_carre()


def test_la_parite_des_fonctions_des_exemples() -> None:
    """8.7.1 : `|t|` est paire, `t` impaire, `t^2 + t` ni l'une ni l'autre."""
    assert est_paire(sp.Abs(T))
    assert not est_impaire(sp.Abs(T))
    assert est_impaire(T)
    assert not est_paire(T)
    assert not est_paire(T**2 + T)
    assert not est_impaire(T**2 + T)


def test_les_coefficients_de_l_exemple_8_4() -> None:
    """`f = |t|` sur `[-pi, pi]` : `b_n = 0`, `a_n = 2((-1)^n - 1)/(pi n^2)`."""
    assert sp.simplify(VALEUR_ABSOLUE.a(N) - 2 * ((-1) ** N - 1) / (sp.pi * N**2)) == 0
    assert sp.simplify(VALEUR_ABSOLUE.b(N)) == 0
    assert VALEUR_ABSOLUE.coefficient(0) == sp.pi / 2
    assert VALEUR_ABSOLUE.pulsation == 1


def test_seuls_les_harmoniques_impairs_de_la_valeur_absolue_subsistent() -> None:
    """Exemple 8.4 : `pi/2 - somme 4 cos((2n + 1) t) / (pi (2n + 1)^2)`."""
    assert VALEUR_ABSOLUE.a(2) == 0
    assert VALEUR_ABSOLUE.a(4) == 0
    for rang in (1, 3, 5):
        assert VALEUR_ABSOLUE.a(rang) == -4 / (sp.pi * rang**2)


def test_les_sommes_que_donne_l_exemple_8_4() -> None:
    """`f(0) = 0` donne `somme 1/(2n+1)^2 = pi^2/8` ; Parseval, les puissances 4."""
    k = sp.Symbol("k", integer=True, nonnegative=True)
    impairs = sp.Sum(1 / (2 * k + 1) ** 2, (k, 0, sp.oo)).doit()
    assert impairs == sp.pi**2 / 8
    assert (
        sp.simplify(sp.pi / 2 - 4 / sp.pi * impairs - VALEUR_ABSOLUE.valeur_en(0)) == 0
    )
    assert VALEUR_ABSOLUE.norme_au_carre() == sp.pi**2 / 3
    quatriemes = sp.Sum(1 / (2 * k + 1) ** 4, (k, 0, sp.oo)).doit()
    assert quatriemes == sp.pi**4 / 96
    assert sp.simplify(sp.pi**2 / 4 + 8 / sp.pi**2 * quatriemes - sp.pi**2 / 3) == 0
    assert sp.pi**4 / 90 - sp.pi**4 / 96 == sp.pi**4 / 1440


def test_les_coefficients_de_l_exemple_8_5() -> None:
    """`f = t` sur `]-pi, pi]` : `a_n = 0`, `b_n = -2 (-1)^n / n`."""
    assert sp.simplify(DENT_DE_SCIE.a(N)) == 0
    assert DENT_DE_SCIE.coefficient(0) == 0
    assert sp.simplify(DENT_DE_SCIE.b(N) + 2 * (-1) ** N / N) == 0


def test_parseval_sur_l_exemple_8_5_redonne_la_somme_des_inverses_des_carres() -> None:
    """`||f||^2 = pi^2/3 = somme (2/n)^2 / 2 = 2 somme 1/n^2`."""
    assert DENT_DE_SCIE.norme_au_carre() == sp.pi**2 / 3
    harmoniques = sp.Sum(sp.simplify(DENT_DE_SCIE.amplitude(N) ** 2 / 2), (N, 1, sp.oo))
    assert sp.simplify(harmoniques.doit() - sp.pi**2 / 3) == 0


def test_en_sa_discontinuite_la_dent_de_scie_vaut_zero() -> None:
    """En `t = pi`, les limites sont `pi` et `-pi` : la serie y vaut 0."""
    assert DENT_DE_SCIE.valeur_en(sp.pi) == 0
    assert DENT_DE_SCIE.valeur_en(-sp.pi) == 0
    assert DENT_DE_SCIE.valeur_en(1) == 1


def test_deux_series_donnent_la_meme_fonction_sur_zero_pi() -> None:
    """Page 153 : la serie en sinus de `t` et celle en cosinus de `|t|` valent `t`."""
    sinus = sp.lambdify(N, DENT_DE_SCIE.b(N))
    cosinus = sp.lambdify(N, VALEUR_ABSOLUE.a(N))
    for instant in (0.5, 1.0, 2.5):
        en_sinus = sum(sinus(k) * math.sin(k * instant) for k in range(1, 4001))
        en_cosinus = math.pi / 2 + sum(
            cosinus(k) * math.cos(k * instant) for k in range(1, 4001)
        )
        assert en_sinus == pytest.approx(instant, abs=2e-3)
        assert en_cosinus == pytest.approx(instant, abs=2e-3)


def test_le_prolongement_impair_de_l_exemple_8_6() -> None:
    """`f(x) = x` sur `[0, 2]` : `T = 4`, `w = pi/2`, `b_n = 4 (-1)^(n+1) / (n pi)`."""
    serie = prolongement_impair(T, 2)
    assert serie.periode == 4
    assert serie.pulsation == sp.pi / 2
    assert sp.simplify(serie.a(N)) == 0
    assert sp.simplify(serie.b(N) - 4 * (-1) ** (N + 1) / (N * sp.pi)) == 0


def test_le_prolongement_pair_donne_une_serie_de_cosinus() -> None:
    """8.8 : `f(-t) = f(t)`. Pour `f = t` sur `[0, pi]`, c'est l'exemple 8.4."""
    serie = prolongement_pair(T, sp.pi)
    assert sp.simplify(serie.b(N)) == 0
    assert sp.simplify(serie.a(N) - VALEUR_ABSOLUE.a(N)) == 0
    assert serie.coefficient(0) == sp.pi / 2


def test_les_prolongements_ont_la_parite_annoncee() -> None:
    """Sur `[-A, A]`, le motif prolonge est pair ou impair."""
    for instant in (R(1, 2), 1, R(3, 2)):
        pair = prolongement_pair(T**2 + T, 2).motif
        impair = prolongement_impair(T**2 + T, 2).motif
        assert pair.subs(T, -instant) == pair.subs(T, instant)
        assert impair.subs(T, -instant) == -impair.subs(T, instant)


def test_une_periode_negative_ou_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="periode strictement positive"):
        SerieDeFourier(T, 0, 0)


def test_une_demi_periode_n_est_pas_une_periode() -> None:
    """`sin(pi t)` change de signe quand `t` avance de 1 : sa periode est 2."""
    assert est_periodique(sp.sin(sp.pi * T), 2)
    assert not est_periodique(sp.sin(sp.pi * T), 1)
