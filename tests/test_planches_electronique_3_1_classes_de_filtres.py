"""Les planches d'electronique : les classes de filtres (chapitre 3, partie I).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_3_part1.pdf`, planches
11 a 58, lues sur leur couche de texte et, pour les formules, sur les
pages rendues.

Les oracles : `epsilon = 0,48432` pour `A_max = 0,1` (planche 28), les
poles du Butterworth d'ordre trois (planche 101), la formule de transfert
de Chebyshev (planche 34), la bande a trois decibels de Bessel (planche
44) et la serie de Fourier du signal carre (planche 51).

Deux ecarts, lus sur la page rendue : la planche 36 ecrit `epsilon =
racine(10^(A_max) - 1)` la ou la planche 29 divise l'exposant par dix, et
« |T(j w_s)| <= A_min » la ou la planche 30 ecrit `1 - A_min`.
"""

from __future__ import annotations

import cmath
import math

import pytest
import sympy as sp

from polytech.electronique.domain.bessel import (
    bande_a_trois_decibels_de_bessel,
    poles_de_bessel,
    polynome_de_bessel_inverse,
    transfert_de_bessel,
)
from polytech.electronique.domain.butterworth import (
    module_de_butterworth,
    ondulation_depuis_l_attenuation,
    poles_de_butterworth,
)
from polytech.electronique.domain.familles import ondulation_depuis_l_affaiblissement
from polytech.electronique.domain.pole_zero import pente_asymptotique
from polytech.electronique.domain.reponse_de_chebyshev import (
    module_de_chebyshev,
    poles_de_chebyshev,
    polynome_de_chebyshev,
    transfert_de_chebyshev,
)
from polytech.electronique.domain.specifications_de_filtre import (
    module_par_les_distances,
    ordre_minimal,
    perte_d_insertion,
    retard_de_groupe,
    signal_carre_filtre,
)
from polytech.maths.domain.fourier.series import SerieDeFourier
from polytech.maths.domain.symboles import VARIABLE

EPSILON = 0.48432
"""Planche 28 : « epsilon = 0.48432 (which entails A_max = 0.1) »."""


def test_le_module_se_lit_sur_la_carte_des_poles_et_zeros() -> None:
    """Planche 20 : `|T| = |a_M| prod |jw - z_i| / prod |jw - p_i|`."""
    poles, zeros = (-1 + 2j, -1 - 2j, -3 + 0j), (0 + 4j, 0 - 4j)
    pulsation = 1.5
    s = 1j * pulsation
    direct = 2 * (s - zeros[0]) * (s - zeros[1])
    direct /= (s - poles[0]) * (s - poles[1]) * (s - poles[2])
    module = module_par_les_distances(pulsation, zeros=zeros, poles=poles, gain=2.0)
    assert module == pytest.approx(abs(direct))
    assert module_par_les_distances(4.0, zeros=zeros, poles=poles) == 0.0


def test_la_perte_d_insertion() -> None:
    """Planche 20 : `L_A = 10 log10(|T|^-2)` ; 3,01 dB a la coupure (planche 65)."""
    assert perte_d_insertion(1 / math.sqrt(2)) == pytest.approx(3.0103, abs=1e-4)
    assert perte_d_insertion(1.0) == 0.0
    with pytest.raises(ValueError, match="module nul ou negatif"):
        perte_d_insertion(0.0)


def test_les_zeros_coutent_de_la_pente() -> None:
    """Planches 23 et 24 : 100 dB par decade a cinq poles, 20 avec quatre zeros."""
    assert pente_asymptotique(0, 5) == -100.0
    assert pente_asymptotique(4, 5) == -(5 - 4) * 20.0


def test_le_parametre_epsilon() -> None:
    """Planche 29 : `epsilon = racine(10^(A/10) - 1)`, ou `racine(1/(1-A)^2 - 1)`."""
    assert ondulation_depuis_l_attenuation(0.1) == pytest.approx(EPSILON, abs=5e-6)
    en_decibels = 20 * math.log10(math.sqrt(1 + EPSILON**2))
    assert en_decibels == pytest.approx(-20 * math.log10(0.9), abs=1e-5)
    depuis_les_decibels = ondulation_depuis_l_affaiblissement(en_decibels)
    assert depuis_les_decibels == pytest.approx(EPSILON)
    with pytest.raises(ValueError, match="attenuation hors de"):
        ondulation_depuis_l_attenuation(1.0)


def test_la_planche_36_oublie_de_diviser_l_exposant_par_dix() -> None:
    """`racine(10^(A_max) - 1)` rendrait 2,69 pour 0,915 dB, et non 0,484."""
    en_decibels = -20 * math.log10(0.9)
    assert math.sqrt(10**en_decibels - 1) == pytest.approx(2.69, abs=0.01)
    assert math.sqrt(10 ** (en_decibels / 10) - 1) == pytest.approx(EPSILON, abs=1e-5)


def test_le_module_de_butterworth() -> None:
    """Planches 27 et 28 : `1 / racine(1 + eps^2 (w/w_p)^(2N))`, 0,9 en `w_p`."""
    for ordre in (1, 2, 6):
        en_bord = module_de_butterworth(1000.0, ordre, EPSILON, passante=1000.0)
        assert en_bord == pytest.approx(0.9, abs=1e-5)
    assert module_de_butterworth(0.0, 6, EPSILON, passante=1000.0) == 1.0
    plus_raide = module_de_butterworth(2000.0, 6, EPSILON, passante=1000.0)
    assert plus_raide < module_de_butterworth(2000.0, 2, EPSILON, passante=1000.0)
    assert plus_raide == pytest.approx(1 / math.sqrt(1 + EPSILON**2 * 2**12))


def test_l_ordre_se_cherche_en_l_incrementant() -> None:
    """Planche 30 : augmenter `N` jusqu'a `|T(j w_s)| <= 1 - A_min`."""

    def a_la_bande_bloquante(ordre: int) -> float:
        return module_de_butterworth(2000.0, ordre, EPSILON, passante=1000.0)

    ordre = ordre_minimal(a_la_bande_bloquante, 1 - 0.9)
    assert ordre == 5
    assert a_la_bande_bloquante(5) <= 0.1 < a_la_bande_bloquante(4)
    with pytest.raises(ValueError, match="aucun ordre jusqu'a"):
        ordre_minimal(lambda _: 0.5, 0.1, maximum=8)


def test_la_planche_36_ecrit_a_min_pour_un_moins_a_min() -> None:
    """« <= A_min » avec `A_min = 0,9` : l'ordre un suffirait, sans rien bloquer."""

    def chebyshev(ordre: int) -> float:
        return module_de_chebyshev(2.0, ordre, EPSILON)

    assert ordre_minimal(chebyshev, 0.9) == 1
    assert ordre_minimal(chebyshev, 1 - 0.9) == 3


def test_les_poles_de_butterworth() -> None:
    """Planches 31 et 101 : module `w_p / eps^(1/N)`, ecartes de `pi / N`."""
    poles = poles_de_butterworth(3)
    attendus = [cmath.exp(2j * math.pi / 3), cmath.exp(1j * math.pi)]
    attendus.append(cmath.exp(4j * math.pi / 3))
    assert poles == pytest.approx(attendus)
    assert poles[0] * poles[1] * poles[2] == pytest.approx(-1)
    etires = poles_de_butterworth(6, epsilon=EPSILON, passante=1000.0)
    rayon = 1000.0 / EPSILON ** (1 / 6)
    assert [abs(pole) for pole in etires] == pytest.approx([rayon] * 6)
    assert all(pole.real < 0 for pole in etires)
    ecart = cmath.phase(etires[1] / etires[0])
    assert ecart == pytest.approx(math.pi / 6)
    module = module_par_les_distances(700.0, poles=etires, gain=rayon**6)
    assert module == pytest.approx(
        module_de_butterworth(700.0, 6, EPSILON, passante=1e3)
    )


def test_le_polynome_de_chebyshev() -> None:
    """Planche 33 : `cos(N acos w)` dans la bande, `cosh(N acosh w)` au-dela."""
    assert polynome_de_chebyshev(3, 0.5) == pytest.approx(4 * 0.5**3 - 3 * 0.5)
    assert polynome_de_chebyshev(3, 2.0) == pytest.approx(4 * 2.0**3 - 3 * 2.0)
    assert polynome_de_chebyshev(4, -1.5) == pytest.approx(8 * 1.5**4 - 8 * 1.5**2 + 1)
    assert polynome_de_chebyshev(5, 1.0) == pytest.approx(1.0)


def test_le_module_de_chebyshev_ondule_dans_la_bande() -> None:
    """Planches 33 et 35 : 0,9 en `w = 1` a tout ordre ; pair, 0,9 des `w = 0`."""
    for ordre in (1, 2, 3, 4, 7):
        assert module_de_chebyshev(1.0, ordre, EPSILON) == pytest.approx(0.9, abs=1e-5)
    assert module_de_chebyshev(0.0, 3, EPSILON) == pytest.approx(1.0)
    assert module_de_chebyshev(0.0, 4, EPSILON) == pytest.approx(0.9, abs=1e-5)
    butterworth = module_de_butterworth(2.0, 5, EPSILON)
    assert module_de_chebyshev(2.0, 5, EPSILON) < butterworth / 5


def test_la_formule_de_transfert_de_chebyshev() -> None:
    """Planche 34 : numerateur `prod racine(eta^2 + sin^2(i pi / N))`, poles `-p_i`."""
    for ordre in (1, 2, 3, 4, 5):
        poles = poles_de_chebyshev(ordre, EPSILON)
        assert all(pole.real < 0 for pole in poles)
        for pulsation in (0.0, 0.5, 1.0, 2.0):
            transfert = transfert_de_chebyshev(1j * pulsation, ordre, EPSILON)
            attendu = module_de_chebyshev(pulsation, ordre, EPSILON)
            assert abs(transfert) == pytest.approx(attendu)
    eta = math.sinh(math.asinh(1 / EPSILON) / 4)
    for pole in poles_de_chebyshev(4, EPSILON):
        ellipse = (pole.real / eta) ** 2 + (pole.imag / math.sqrt(1 + eta**2)) ** 2
        assert ellipse == pytest.approx(1.0)


def test_la_phase_lineaire_translate_sans_deformer() -> None:
    """Planches 41 et 42 : `arg T = C w` donne le meme decalage `C` aux deux sinus."""
    retard = 2e-4
    for pulsation in (1e3, 3e3):
        phase = -retard * pulsation
        instant = 1.7e-3
        sortie = math.sin(pulsation * instant + phase)
        assert sortie == pytest.approx(math.sin(pulsation * (instant - retard)))


def test_le_polynome_de_bessel_inverse() -> None:
    """Planche 44 : `theta_3(s) = s^3 + 6 s^2 + 15 s + 15`, et `T(0) = 1`."""
    s = sp.Symbol("s")
    assert (
        sp.expand(polynome_de_bessel_inverse(3, s) - (s**3 + 6 * s**2 + 15 * s + 15))
        == 0
    )
    recurrence = (2 * 4 - 1) * polynome_de_bessel_inverse(3, s)
    recurrence += s**2 * polynome_de_bessel_inverse(2, s)
    assert sp.expand(polynome_de_bessel_inverse(4, s) - recurrence) == 0
    assert transfert_de_bessel(0j, 5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="ordre inferieur a un"):
        polynome_de_bessel_inverse(0, s)


def test_la_bande_a_trois_decibels_de_bessel() -> None:
    """Planche 44 : `w_3dB ~ racine((2N - 1) ln 2)`, « good for N >= 3 »."""
    for ordre, tolerance in ((3, 0.07), (5, 0.03), (8, 0.015)):
        approchee = bande_a_trois_decibels_de_bessel(ordre)
        assert approchee == pytest.approx(math.sqrt((2 * ordre - 1) * math.log(2)))
        module = abs(transfert_de_bessel(1j * approchee, ordre))
        assert perte_d_insertion(module) == pytest.approx(3.01, rel=4 * tolerance)
    gaussienne = 10 * math.log10(math.exp(1.2**2 / (2 * 8 - 1)))
    assert perte_d_insertion(abs(transfert_de_bessel(1.2j, 8))) == pytest.approx(
        gaussienne, rel=0.01
    )


def test_le_retard_de_groupe_de_bessel_est_plat() -> None:
    """Planches 48 et 55 : constant dans la bande pour Bessel, pas pour Butterworth."""
    bessel = poles_de_bessel(5)
    assert retard_de_groupe(bessel, 0.0) == pytest.approx(1.0)
    assert retard_de_groupe(bessel, 2.0) == pytest.approx(1.0, abs=2e-3)
    butterworth = poles_de_butterworth(5)
    au_repos = retard_de_groupe(butterworth, 0.0)
    assert retard_de_groupe(butterworth, 0.95) > 1.5 * au_repos
    pas = 1e-6
    phases = [
        cmath.phase(transfert_de_bessel(1j * (0.5 + signe * pas), 5))
        for signe in (-1, 1)
    ]
    derivee = (phases[1] - phases[0]) / (2 * pas)
    assert retard_de_groupe(bessel, 0.5) == pytest.approx(-derivee, rel=1e-5)


def test_la_serie_de_fourier_du_signal_carre() -> None:
    """Planche 51 : `(4 / pi) somme_(n impair) sin(2 pi n f_o t) / n`."""
    carre = SerieDeFourier(sp.sign(VARIABLE), sp.Rational(-1, 2), sp.Integer(1))
    for rang in (1, 3, 5, 7):
        assert sp.simplify(carre.b(rang) - 4 / (sp.pi * rang)) == 0
        assert carre.a(rang) == 0
    assert carre.b(2) == 0
    sans_filtre = signal_carre_filtre(0.25e-3, 1e3, lambda _: 1 + 0j, harmoniques=399)
    assert sans_filtre == pytest.approx(1.0, abs=5e-3)


def test_butterworth_fait_sonner_le_signal_carre_et_bessel_non() -> None:
    """Planches 53 et 54 : ordre 5, coupure 5 kHz, signal de 1 kHz.

    Le signal saute de -1 a +1 : un depassement de 0,2 au-dessus de 1 est
    dix pour cent du saut. Bessel en garde moins d'un pour cent, que la
    planche dit « not noticeable ».
    """
    coupure = 2 * math.pi * 5e3
    poles = [coupure * pole for pole in poles_de_butterworth(5)]

    def butterworth(pulsation: float) -> complex:
        produit = complex(coupure**5)
        for pole in poles:
            produit /= 1j * pulsation - pole
        return produit

    echelle = 2.4274 / coupure

    def bessel(pulsation: float) -> complex:
        return transfert_de_bessel(1j * pulsation * echelle, 5)

    assert abs(bessel(coupure)) == pytest.approx(1 / math.sqrt(2), abs=1e-4)
    instants = [k * 1e-6 for k in range(500)]
    cretes = [
        max(signal_carre_filtre(t, 1e3, filtre, harmoniques=99) for t in instants)
        for filtre in (butterworth, bessel)
    ]
    depassements = [crete - 1 for crete in cretes]
    assert depassements[0] > 0.2
    assert 0 < depassements[1] < 0.02
    assert depassements[0] > 10 * depassements[1]


def test_les_gardes_des_classes_de_filtres() -> None:
    """Un ordre nul, une bande ou une ondulation non positive ne decrivent rien."""
    with pytest.raises(ValueError, match="ordre inferieur a un"):
        module_de_butterworth(1.0, 0)
    with pytest.raises(ValueError, match="pulsation passante nulle ou negative"):
        module_de_butterworth(1.0, 2, passante=0.0)
    with pytest.raises(ValueError, match="parametre d'ondulation nul ou negatif"):
        poles_de_butterworth(2, epsilon=0.0)
    with pytest.raises(ValueError, match="ordre inferieur a un"):
        poles_de_chebyshev(0, EPSILON)
    with pytest.raises(ValueError, match="parametre d'ondulation nul ou negatif"):
        poles_de_chebyshev(2, 0.0)
    with pytest.raises(ValueError, match="ordre negatif"):
        polynome_de_chebyshev(-1, 0.5)
    with pytest.raises(ValueError, match="ordre inferieur a un"):
        transfert_de_bessel(1j, 0)
    with pytest.raises(ValueError, match="ordre inferieur a un"):
        bande_a_trois_decibels_de_bessel(0)
    with pytest.raises(ValueError, match="pole sur l'axe imaginaire"):
        module_par_les_distances(2.0, poles=(2j, -2j))


def test_le_polynome_de_chebyshev_d_ordre_impair_est_impair() -> None:
    """`T_3(w) = 4 w^3 - 3 w` change de signe avec `w`, hors de la bande aussi."""
    assert polynome_de_chebyshev(3, -2.0) == pytest.approx(4 * (-2.0) ** 3 + 6.0)
    assert polynome_de_chebyshev(3, -0.5) == pytest.approx(4 * (-0.5) ** 3 + 1.5)
    assert polynome_de_chebyshev(0, 3.0) == pytest.approx(1.0)


def test_l_ordre_ne_se_cherche_pas_au_dela_du_maximum() -> None:
    """Un gabarit que seul l'ordre neuf tient est refuse a huit, trouve a neuf."""

    def tenu_a_neuf(ordre: int) -> float:
        return 0.0 if ordre >= 9 else 1.0

    assert ordre_minimal(tenu_a_neuf, 0.1, maximum=9) == 9
    with pytest.raises(ValueError, match="aucun ordre jusqu'a 8"):
        ordre_minimal(tenu_a_neuf, 0.1, maximum=8)


def test_le_retard_de_groupe_d_un_pole_culmine_a_sa_pulsation() -> None:
    """Un seul pole `-a + j b` : `a / (a^2 + (w - b)^2)`, maximal en `w = b`."""
    pole = (complex(-2.0, 5.0),)
    assert retard_de_groupe(pole, 5.0) == pytest.approx(1 / 2.0)
    assert retard_de_groupe(pole, -5.0) == pytest.approx(2.0 / (4.0 + 100.0))
