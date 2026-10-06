"""Mathematique, exercices 2, seance II : transformees de Laplace.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 11
a 15. Chaque reponse de la feuille est recopiee et controlee par un autre
chemin : une image par la transformee de sa reponse, une solution par son
report dans l'equation.

Deux reponses de la feuille ne passent pas ce controle, et les tests le
disent : la valeur finale de II.3, et le regime sinusoidal de II.7.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import laplace_exs_02 as la
from polytech.maths.application import problemes_exs_02 as pb
from polytech.maths.application import reponses_exs_02 as re
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

W, PHI = la.PULSATION, la.PHASE

IMAGES_OFFICIELLES = {
    "a": 9 * sp.factorial(7) / P**8 - 3 * sp.factorial(5) / P**6 + 2 / P**3 - 3 / P,
    "b": (P + 2) / ((P + 2) ** 2 + 1) + 4 / (P + 2) ** 3 - 1 / (P + 2),
    "c": 2 / (P**2 - 4),
    "d": (P * sp.cos(PHI) - W * sp.sin(PHI)) / (P**2 + W**2),
    "e": (W * (sp.cos(PHI) + sp.sin(PHI)) + P * sp.sin(PHI))
    / (P**2 + 2 * W * P + 2 * W**2),
}
"""Page 11, « Solutions », exercice II.1.1."""

ORIGINAUX_OFFICIELS = {
    1: sp.sin(T),
    2: 2 / sp.sqrt(3) * sp.exp(-T / 2) * sp.sin(sp.sqrt(3) * T / 2),
    3: sp.exp(-sp.Rational(4, 3) * T) / 3,
    4: sp.Rational(2, 3) * sp.exp(-T) * (1 - T),
    5: sp.Rational(3, 4) * sp.exp(-3 * T) + sp.exp(T) / 4,
    6: sp.exp(T) / 4 * (5 - 2 * T) - sp.exp(-T) / 4,
    7: 3 * sp.cos(3 * T) + sp.Rational(4, 3) * sp.sin(3 * T),
}
"""Page 12, exercice II.2."""


@pytest.mark.parametrize("lettre", sorted(IMAGES_OFFICIELLES))
def test_ii1_les_images_de_la_feuille(lettre: str) -> None:
    assert sp.simplify(la.images()[lettre] - IMAGES_OFFICIELLES[lettre]) == 0


def test_ii1_translater_n_est_pas_decaler_l_argument() -> None:
    """Page 11 : `(t - 1)^3 H(t)` et `tau_1(t^3 H)` n'ont pas la meme image."""
    tronquee, retardee = la.cube_tronque_et_cube_retarde()
    assert sp.simplify(tronquee - (6 / P**4 - 6 / P**3 + 3 / P**2 - 1 / P)) == 0
    assert sp.simplify(retardee - sp.exp(-P) * 6 / P**4) == 0


def test_ii1_les_deux_fonctions_periodiques() -> None:
    """Page 11 : la dent de scie et le creneau."""
    scie, creneau = la.periodiques()
    attendue = 1 / P**2 - sp.exp(-P) / P / (1 - sp.exp(-P))
    assert sp.simplify(scie - attendue) == 0
    assert sp.simplify(creneau - 1 / P / (1 + sp.exp(-P))) == 0


@pytest.mark.parametrize("numero", sorted(ORIGINAUX_OFFICIELS))
def test_ii2_les_originaux_de_la_feuille(numero: int) -> None:
    image, original = la.ORIGINAUX[numero]()
    assert sp.simplify(original - ORIGINAUX_OFFICIELS[numero]) == 0
    assert sp.simplify(Transformee.de(original).image - image) == 0


def test_ii2_les_fractions_simples_de_la_6() -> None:
    """`(p^3 + p^2 - 2p - 2)/(p^2 - 1)^2` : un pole double en 1, un simple en -1."""
    termes = set(la.fractions_de_la_6())
    attendus = {5 / (4 * (P - 1)), -1 / (2 * (P - 1) ** 2), -1 / (4 * (P + 1))}
    assert {sp.simplify(t) for t in termes} == {sp.simplify(t) for t in attendus}


# --- II.3 : la valeur finale -----------------------------------------------


def test_ii3_la_formule_rend_la_valeur_de_la_feuille() -> None:
    """Page 12 : `lim p F(p) = 1/a^2` -- c'est ce que la formule rend."""
    assert pb.valeur_par_la_formule() == 1 / pb.A**2


def test_ii3_mais_le_theoreme_ne_s_applique_pas() -> None:
    """Ecart : avec `a > 0`, `F` a un pole double en `p = a`, a droite.

    L'originale est `(1 - e^(at) + a t e^(at))/a^2`, qui tend vers l'infini :
    `f` n'a pas de limite, et la formule rend un nombre qui n'en est pas une.
    """
    assert pb.poles_a_droite() == (pb.A,)
    with pytest.raises(ValueError, match="inapplicable"):
        pb.valeur_finale_par_le_theoreme()
    originale = pb.originale_du_3()
    attendue = (1 - sp.exp(pb.A * T) + pb.A * T * sp.exp(pb.A * T)) / pb.A**2
    assert sp.simplify(originale - attendue) == 0
    assert sp.limit(originale, T, sp.oo) == sp.oo


def test_ii3_le_theoreme_vaut_pour_un_pole_a_gauche() -> None:
    """Avec `a < 0`, l'enonce serait juste : la limite existe et vaut `1/a^2`."""
    limite, par_le_theoreme = pb.cas_stable()
    assert limite == par_le_theoreme == sp.Rational(1, 4)


# --- II.4 : la convolution --------------------------------------------------


def test_ii4_l_image_et_ses_fractions_simples() -> None:
    """Page 13 : `1/p + 2/p^2 - 1/(1 + p^2) - 1/(p - 1)`."""
    image, termes = pb.image_de_la_convolution()
    assert sp.simplify(image - (1 / (1 + P**2) - 1 / (P - 1)) / P**2) == 0
    attendus = {1 / P, 2 / P**2, -1 / (1 + P**2), -1 / (P - 1)}
    assert {sp.simplify(t) for t in termes} == attendus


def test_ii4_le_produit_par_les_deux_chemins() -> None:
    """Par l'image, puis par l'integrale de convolution elle-meme."""
    par_l_image, par_l_integrale = pb.convolution()
    attendue = 1 + 2 * T - sp.sin(T) - sp.exp(T)
    assert sp.simplify(par_l_image - attendue) == 0
    assert sp.simplify(par_l_integrale - attendue) == 0


# --- II.5 : equations a coefficients constants -----------------------------


def test_ii5_les_deux_premieres_solutions() -> None:
    premier, deuxieme = pb.probleme_1(), pb.probleme_2()
    assert sp.simplify(premier.solution() - (2 * sp.exp(2 * T) - sp.exp(3 * T))) == 0
    attendue = (2 + T + 2 * T**2) * sp.exp(-T)
    assert sp.simplify(deuxieme.solution() - attendue) == 0
    assert premier.solution_verifiee()
    assert deuxieme.solution_verifiee()


def test_ii5_les_images_des_inconnues() -> None:
    """Ce que la feuille ne montre pas : l'equation algebrique resolue."""
    assert sp.simplify(pb.probleme_1().image() - (P - 4) / ((P - 2) * (P - 3))) == 0
    attendue = 2 / (P + 1) + 1 / (P + 1) ** 2 + 4 / (P + 1) ** 3
    assert sp.simplify(pb.probleme_2().image() - attendue) == 0


def test_ii5_la_rampe_ecretee() -> None:
    """Page 13 : `g(t) - g(t - 1)`, avec `g = (2t - sin 2t)/8`."""
    avant, apres = pb.rampe_ecretee()
    assert sp.simplify(avant - (2 * T - sp.sin(2 * T)) / 8) == 0
    assert sp.simplify(apres - (avant - avant.subs(T, T - 1))) == 0
    assert sp.simplify(avant.diff(T, 2) + 4 * avant - T) == 0
    assert sp.simplify(apres.diff(T, 2) + 4 * apres - 1) == 0
    assert (avant.subs(T, 0), avant.diff(T).subs(T, 0)) == (0, 0)
    for ordre in (0, 1):
        saut = (apres - avant).diff(T, ordre).subs(T, 1)
        assert sp.simplify(saut) == 0


# --- II.6 : reponse impulsionnelle, reponse indicielle ---------------------


def test_ii6_la_reponse_indicielle() -> None:
    attendue = sp.Rational(1, 3) - sp.exp(-T) / 2 + sp.exp(-3 * T) / 6
    assert sp.simplify(re.indicielle() - attendue) == 0


def test_ii6_la_reponse_au_creneau_et_sa_limite() -> None:
    """Page 14 : `h_n`, puis `h = (e^(-t) - e^(-3t))/2` quand `n` tend vers l'infini."""
    n = re.HAUTEUR
    pendant, apres = re.au_creneau()
    indicielle = re.indicielle()
    assert sp.simplify(pendant - n * indicielle) == 0
    assert sp.simplify(apres - n * (indicielle - indicielle.subs(T, T - 1 / n))) == 0
    impulsionnelle = (sp.exp(-T) - sp.exp(-3 * T)) / 2
    assert sp.simplify(sp.limit(apres, n, sp.oo) - impulsionnelle) == 0
    assert sp.simplify(re.impulsionnelle() - impulsionnelle) == 0


def test_ii6_l_impulsion_equivaut_a_une_vitesse_initiale() -> None:
    """Question 4 : la reponse libre a `y'(0) = 1` est la reponse impulsionnelle."""
    assert re.conditions_equivalentes() == (0, 1)
    assert sp.simplify(re.libre() - re.impulsionnelle()) == 0


def test_ii6_la_derivee_de_l_indicielle_est_l_impulsionnelle() -> None:
    assert sp.simplify(re.indicielle().diff(T) - re.impulsionnelle()) == 0


# --- II.7 : le circuit RLC --------------------------------------------------


def test_ii7_l_admittance_du_circuit() -> None:
    """`U = (L p + R + 1/(C p)) I`, avec `R = 3`, `L = 1`, `C = 1/2`."""
    assert sp.simplify(re.admittance() - P / ((P + 1) * (P + 2))) == 0


def test_ii7_les_reponses_indicielle_et_impulsionnelle() -> None:
    indicielle, impulsionnelle, _ = re.courants()
    assert sp.simplify(indicielle - (sp.exp(-T) - sp.exp(-2 * T))) == 0
    assert sp.simplify(impulsionnelle - (2 * sp.exp(-2 * T) - sp.exp(-T))) == 0
    assert sp.simplify(indicielle.diff(T) - impulsionnelle) == 0


def test_ii7_la_reponse_sinusoidale_verifie_l_equation() -> None:
    """Derivee une fois, l'equation du circuit est `i'' + 3 i' + 2 i = u'`."""
    courant = re.courants()[2]
    reste = courant.diff(T, 2) + 3 * courant.diff(T) + 2 * courant - sp.cos(T)
    assert sp.simplify(reste) == 0
    assert courant.subs(T, 0) == 0
    attendue = -sp.exp(-T) / 2 + 2 * sp.exp(-2 * T) / 5
    assert sp.simplify(courant - attendue - (sp.cos(T) + 3 * sp.sin(T)) / 10) == 0


def test_ii7_la_feuille_echange_les_deux_exponentielles_complexes() -> None:
    """Ecart : la feuille attache `1/(2(1 + 3j))` a `e^(-jt)`.

    Le residu de `I(p) e^(pt)` en `p = j` est `e^(jt)/(2(1 + 3j))`. Les deux
    termes echanges, le regime etabli devient `(cos t - 3 sin t)/10`, qui
    ne verifie pas l'equation du circuit.
    """
    j = sp.I
    assert sp.simplify(re.residu_en_j() - 1 / (2 * (1 + 3 * j))) == 0
    officiel = (
        -sp.exp(-T) / 2
        + 2 * sp.exp(-2 * T) / 5
        + sp.exp(-j * T) / (2 * (1 + 3 * j))
        + sp.exp(j * T) / (2 * (1 - 3 * j))
    )
    reel = sp.simplify(sp.re(sp.expand_complex(officiel)))
    etabli = reel + sp.exp(-T) / 2 - 2 * sp.exp(-2 * T) / 5
    assert sp.simplify(etabli - (sp.cos(T) - 3 * sp.sin(T)) / 10) == 0
    reste = reel.diff(T, 2) + 3 * reel.diff(T) + 2 * reel - sp.cos(T)
    assert sp.simplify(reste) != 0


def test_ii7_le_regime_etabli_par_l_impedance_complexe() -> None:
    """Second chemin : `Z(j) = 3 - j`, donc `|I| = 1/sqrt(10)` et un retard nul.

    `I = U/Z = 1/(3 - j) = (3 + j)/10` : la partie reelle va au sinus (la
    tension est un sinus), l'imaginaire au cosinus.
    """
    amplitude, etabli = re.regime_etabli()
    assert amplitude == 1 / sp.sqrt(10)
    assert sp.simplify(etabli - (3 * sp.sin(T) + sp.cos(T)) / 10) == 0


def test_ii7_la_reponse_pour_une_pulsation_quelconque() -> None:
    """L'enonce ecrit `sin(omega t)` ; la feuille ne traite que `omega = 1`."""
    courant = re.courant_sinusoidal(W)
    reste = courant.diff(T, 2) + 3 * courant.diff(T) + 2 * courant
    assert sp.simplify(reste - W * sp.cos(W * T)) == 0
    assert sp.simplify(courant.subs(W, 1) - re.courants()[2]) == 0


def test_ii7_l_impulsion_donne_un_courant_initial() -> None:
    """Le courant vaut `1/L = 1` juste apres l'impulsion, et non zero."""
    assert re.courants()[1].subs(T, 0) == 1 / re.INDUCTANCE


def test_ii7_la_resonance_du_circuit() -> None:
    """L'amplitude etablie culmine en `1/sqrt(LC)`, ou elle vaut `1/R`."""
    pulsation, amplitude = re.resonance_du_circuit()
    assert pulsation == 1 / sp.sqrt(re.INDUCTANCE * re.CAPACITE)
    assert pulsation == sp.sqrt(2)
    assert amplitude == 1 / re.RESISTANCE
