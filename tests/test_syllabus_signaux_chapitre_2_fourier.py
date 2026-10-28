"""Le syllabus de signaux et systemes, chapitre 2 : l'analyse frequentielle.

Source : `SignauxSystemes/SYLLABUS/Syllabus_2019_Chapitre_02_Transformees.pdf`,
section 2.1 (pages 1 a 23), lue sur les pages rendues.

Les oracles sont les exemples du chapitre : `rect <-> sinc` (2.1),
`tri <-> sinc^2` (2.2), `sinc <-> rect` (2.3), le cosinus et ses deux
raies d'amplitude 1/2 (2.6), le rectangle et le triangle periodiques
(2.8, 2.9), les sommets 200 et 100 des estimations numeriques (2.5, 2.6)
et les 123 Hz du segment de parole (2.10).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.maths.domain.comparaison import (
    egales,
    proches,
    proches_relativement,
    suite_nulle,
    suites_proches,
)
from polytech.maths.domain.fourier.distributions import transformee_du_cosinus
from polytech.maths.domain.fourier.transformee import (
    PULSATION,
    transformee_de_fourier,
)
from polytech.maths.domain.symboles import VARIABLE
from polytech.signaux.domain.echantillons import convoluer
from polytech.signaux.domain.frequence import (
    FREQUENCE,
    INSTANT,
    en_frequence,
    rect,
    signal_depuis_le_spectre,
    sinc,
    transformee_en_frequence,
    tri,
)
from polytech.signaux.domain.proprietes_de_fourier import (
    de_la_derivee,
    de_la_dualite,
    de_la_reflexion,
    du_changement_d_echelle,
    du_decalage_temporel,
)
from polytech.signaux.domain.raies import (
    frequence_fondamentale,
    raie,
    taux_de_distorsion_harmonique,
    valeur_efficace,
    valeur_efficace_d_un_signal_reel,
)
from polytech.signaux.domain.spectre_estime import raies_estimees, spectre_estime

F, T = FREQUENCE, INSTANT


def test_l_exemple_2_1_la_porte_devient_un_sinus_cardinal() -> None:
    """(2.10) : `rect(t) <-> sinc(f)`, nul en tout `f = k` entier non nul."""
    image = transformee_en_frequence(rect(T))
    assert egales(image, sinc(F))
    assert sinc(F).subs(F, 3) == 0
    assert sinc(F).subs(F, 0) == 1


def test_la_valeur_en_zero_est_la_surface_du_signal() -> None:
    """(2.9) : `F(0)` est l'integrale de `f(t)`."""
    image = transformee_en_frequence(3 * rect(T / 2))
    assert sp.limit(image, F, 0) == sp.integrate(3 * rect(T / 2), (T, -sp.oo, sp.oo))


def test_l_exemple_2_2_le_triangle_devient_un_sinus_cardinal_au_carre() -> None:
    assert egales(transformee_en_frequence(tri(T)), sinc(F) ** 2)


def test_la_convolution_de_deux_portes_est_un_triangle() -> None:
    """Fig. 2.12 : convoluer en temps, c'est multiplier en frequence."""
    pas = 0.001
    instants = np.arange(-0.5, 0.5 + pas / 2, pas)
    porte = np.ones_like(instants)
    produit = convoluer(porte, porte) * pas
    attendus = sp.lambdify(T, tri(T), "numpy")(np.arange(-1.0, 1.0 + pas / 2, pas))
    assert suites_proches(produit, attendus, 2 * pas)


def test_l_exemple_2_3_le_sinus_cardinal_devient_une_porte() -> None:
    """Par dualite : `F(t) <-> f(-f)`, et la porte est paire."""
    assert de_la_dualite(rect(T)) == rect(-F)
    retour = signal_depuis_le_spectre(rect(F))
    ecart = sp.lambdify(T, retour - sinc(T), "numpy")
    assert suite_nulle(ecart(np.array([-2.3, 0.4, 1.5])), 1e-12)


def test_la_dualite_retourne_l_axe() -> None:
    """Tableau 2.1 : `F(t) <-> f(-f)`, sur une porte decalee, donc non paire."""
    retard = sp.Rational(3, 4)
    decalee = rect(T - retard)
    assert de_la_dualite(decalee) == rect(-F - retard)
    image = du_decalage_temporel(sinc(F), retard).subs(F, T)
    retour = signal_depuis_le_spectre(de_la_dualite(decalee))
    ecart = sp.lambdify(T, retour - image, "numpy")
    assert suite_nulle(ecart(np.array([-1.3, 0.4, 2.2])), 1e-12)


def test_l_exemple_2_4_l_impulsion_de_dirac_contient_toutes_les_frequences() -> None:
    assert transformee_en_frequence(sp.DiracDelta(T)) == 1


def test_le_decalage_temporel_n_ajoute_qu_une_phase_lineaire() -> None:
    """Tableau 2.1 : `f(t - t_0) <-> F(f) exp(-j 2 pi f t_0)`."""
    retard = sp.Rational(3, 4)
    directe = transformee_en_frequence(rect(T - retard))
    assert egales(directe, du_decalage_temporel(sinc(F), retard))
    module = sp.Abs(du_decalage_temporel(sinc(F), retard))
    assert egales(module, sp.Abs(sinc(F)))


def test_la_reflexion_conjugue_la_transformee() -> None:
    """Tableau 2.1 : `f(-t) <-> F*(f)`, sur une porte decalee."""
    retard = sp.Rational(3, 4)
    image = transformee_en_frequence(rect(T - retard))
    assert egales(transformee_en_frequence(rect(-T - retard)), de_la_reflexion(image))


def test_dilater_un_signal_comprime_sa_transformee() -> None:
    """Fig. 2.8 : la porte deux fois plus large a un spectre deux fois plus haut."""
    large = transformee_en_frequence(rect(T / 2))
    assert egales(large, du_changement_d_echelle(sinc(F), sp.Rational(1, 2)))
    assert sp.limit(large, F, 0) == 2
    with pytest.raises(ValueError, match="facteur d'echelle nul"):
        du_changement_d_echelle(sinc(F), 0)


def test_deriver_multiplie_par_j_omega() -> None:
    """Fig. 2.9 : la pente du triangle est une porte montante puis descendante."""
    demi = sp.Rational(1, 2)
    pente = rect(T + demi) - rect(T - demi)
    directe = transformee_en_frequence(pente)
    assert egales(directe, de_la_derivee(sinc(F) ** 2))


def test_parseval_l_energie_se_lit_dans_les_deux_domaines() -> None:
    """Tableau 2.1 : `<f, f> = <F, F>`, soit 1 pour la porte."""
    temporelle = sp.integrate(rect(T) ** 2, (T, -sp.oo, sp.oo))
    frequences = np.linspace(-400, 400, 800_001)
    spectre = np.sinc(frequences) ** 2
    frequentielle = float(np.sum(spectre) * (frequences[1] - frequences[0]))
    assert temporelle == 1
    assert proches(frequentielle, 1.0, 1e-3)


def test_la_notation_en_pulsation_ne_change_que_l_axe() -> None:
    """2.1.6 et fig. 2.22 : `F(omega) = F(f)`, soit `sinc(omega / 2 pi)`."""
    porte = sp.Piecewise((1, sp.Abs(VARIABLE) <= sp.Rational(1, 2)), (0, True))
    en_pulsation = transformee_de_fourier(porte)
    assert egales(en_frequence(en_pulsation, PULSATION), sinc(F))


def test_l_impulsion_de_dirac_change_de_poids_avec_l_axe() -> None:
    """Fig. 2.23 : `delta(f) = 2 pi delta(omega)`, donc `pi` devient `1/2`."""
    pulsation = sp.Symbol("omega_0", positive=True)
    en_pulsation = transformee_du_cosinus(pulsation)
    frequence = pulsation / (2 * sp.pi)
    attendu = (sp.DiracDelta(F - frequence) + sp.DiracDelta(F + frequence)) / 2
    assert egales(en_frequence(en_pulsation, PULSATION), attendu)


def test_l_exemple_2_8_les_raies_du_rectangle_periodique() -> None:
    """Periode 2 s : raies en `k / 2` de valeur `sinc(f) / 2`, impaires en `1 / k`."""
    raies = [raie(sinc(F), 2, rang) for rang in range(6)]
    assert raies[0] == sp.Rational(1, 2)
    assert raies[2] == 0
    assert raies[4] == 0
    assert egales(raies[1], 1 / sp.pi)
    assert egales(raies[3] * 3, -(raies[1]))
    assert egales(raies[5] * 5, raies[1])


def test_la_raie_est_le_coefficient_de_la_serie_de_fourier() -> None:
    """(2.12) : `F_k = (1 / T_0) integrale du motif fois exp(-j k omega_0 t)`."""
    for rang in (0, 1, 3):
        noyau = sp.exp(-sp.I * rang * sp.pi * T)
        directe = sp.integrate(rect(T) * noyau, (T, -1, 1)) / 2
        assert egales(directe, raie(sinc(F), 2, rang))


def test_une_raie_se_prolonge_la_ou_l_image_s_ecrit_zero_sur_zero() -> None:
    """La transformee calculee de la porte s'ecrit `sin(pi f) / (pi f)`."""
    brute = transformee_en_frequence(rect(T))
    assert brute.subs(F, 0) is sp.nan
    assert raie(brute, 2, 0) == sp.Rational(1, 2)


def test_l_exemple_2_9_les_raies_du_triangle_decroissent_en_un_sur_k_carre() -> None:
    """Le syllabus ecrit « decroissant en 1/k », recopie de l'exemple 2.8.

    Les raies valent `sinc^2(k / 2) / 2`, soit `2 / (pi k)^2` aux rangs
    impairs : elles decroissent comme `1 / k^2`.
    """
    raies = [raie(sinc(F) ** 2, 2, rang) for rang in range(6)]
    assert raies[0] == sp.Rational(1, 2)
    assert raies[2] == 0
    assert egales(raies[1], 2 / sp.pi**2)
    assert egales(raies[3] * 9, raies[1])
    assert egales(raies[5] * 25, raies[1])


def test_l_exemple_2_7_le_peigne_de_dirac_a_des_raies_d_amplitude_un_sur_t0() -> None:
    periode = sp.Symbol("T_0", positive=True)
    assert raie(transformee_en_frequence(sp.DiracDelta(T)), periode, 4) == 1 / periode
    with pytest.raises(ValueError, match="periode nulle ou negative"):
        raie(sinc(F), 0, 1)


def test_la_valeur_efficace_d_un_sinus_par_ses_deux_raies() -> None:
    """Fig. 2.14 : deux raies d'amplitude 1/2, valeur efficace `racine(2) / 2`."""
    demi = sp.Rational(1, 2)
    assert valeur_efficace((demi, demi)) == sp.sqrt(2) / 2
    assert valeur_efficace_d_un_signal_reel((demi,)) == sp.sqrt(2) / 2
    carre = sp.integrate(sp.sin(2 * sp.pi * T) ** 2, (T, 0, 1))
    assert sp.sqrt(carre) == sp.sqrt(2) / 2


def test_le_taux_de_distorsion_du_rectangle_periodique() -> None:
    """Le rapport de deux valeurs efficaces : 43,5 % pour l'onde carree.

    « Le rapport entre la valeur efficace de l'ensemble des harmoniques
    d'ordre superieur a 1 et la valeur efficace du signal entier », dit le
    texte. La formule (2.16) n'a pas la racine : elle rend le rapport des
    puissances, `1 - 8 / pi^2`, soit 18,9 %.
    """
    rangs = np.arange(1, 4000)
    amplitudes = list(np.abs(np.sinc(rangs / 2)) / 2)
    for rang in (1, 2, 3):
        attendue = abs(float(raie(sinc(F), 2, rang)))
        assert proches(amplitudes[rang - 1], attendue, 1e-12)
    taux = taux_de_distorsion_harmonique(amplitudes)
    assert proches(taux, 100 * math.sqrt(1 - 8 / math.pi**2), 0.05)
    assert proches(taux, 43.5, 0.1)
    assert proches((taux / 100) ** 2 * 100, 18.9, 0.1)
    assert taux_de_distorsion_harmonique([1.0]) == 0
    with pytest.raises(ValueError, match="fondamental nul"):
        taux_de_distorsion_harmonique([0.0, 1.0])


def test_l_estimation_numerique_de_la_porte_a_un_facteur_un_sur_te_pres() -> None:
    """2.1.3 : deux secondes de `rect(t)` a 10 Hz, sommet « 10 fois » la theorie.

    Le code du cours, tel quel : `arange(-1, 1, 1 / fe)` puis
    `where(abs(x) <= 0.5, 1, 0)`. Dix echantillons valent un, et non
    onze : l'arithmetique flottante place les bords en -0,5000000000000001
    et 0,4999999999999997, de sorte qu'un seul des deux est retenu. Le
    sommet vaut donc exactement dix, comme le syllabus le dit.

    Ce test a d'abord affirme le contraire -- onze, avec une tolerance
    ajoutee a la comparaison. L'ecart etait le mien, non celui du cours.
    """
    cadence = 10.0
    instants = np.arange(-1, 1, 1 / cadence)
    porte = np.where(np.abs(instants) <= 0.5, 1.0, 0.0)
    assert porte.sum() == 10
    assert instants[5] < -0.5 < 0.5 > instants[15] > 0.4999
    frequences = np.array([0.0, 1.5, 2.5])
    estime = spectre_estime(porte, cadence, frequences)
    assert proches_relativement(estime[0] * cadence, 10.0, 1e-6)
    assert proches_relativement(estime[0], 1.0, 1e-6)
    assert suites_proches(estime[1:], np.abs(np.sinc(frequences[1:])), 0.03)


def test_l_exemple_2_5_l_exponentielle_imaginaire_a_un_sommet_de_200() -> None:
    """2.1.5 : `N F_k`, soit 200 pour deux secondes a 100 Hz."""
    cadence = 100.0
    instants = np.arange(0, 2, 1 / cadence)
    signal = np.exp(2j * np.pi * instants)
    brut = raies_estimees(signal, cadence, np.array([1.0])) * len(instants)
    assert proches_relativement(brut[0], 200.0, 1e-6)
    assert proches_relativement(
        raies_estimees(signal, cadence, np.array([1.0]))[0], 1.0, 1e-6
    )


def test_l_exemple_2_6_le_cosinus_n_a_que_la_moitie_de_l_amplitude() -> None:
    cadence = 100.0
    instants = np.arange(0, 2, 1 / cadence)
    signal = np.cos(2 * np.pi * instants)
    assert proches_relativement(
        raies_estimees(signal, cadence, np.array([1.0]))[0], 0.5, 1e-6
    )
    with pytest.raises(ValueError, match="aucun echantillon"):
        raies_estimees(np.array([]), cadence, np.array([1.0]))


def test_l_exemple_2_10_la_frequence_fondamentale_de_la_voix() -> None:
    """« Une periode de l'ordre de 65 echantillons », soit `8000 / 65 = 123 Hz`."""
    assert proches(frequence_fondamentale(65, 8000.0), 123.0, 0.1)
    with pytest.raises(ValueError, match="periode nulle ou negative"):
        frequence_fondamentale(0, 8000.0)
