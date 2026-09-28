"""Les reponses du corrige manuscrit de l'examen blanc 2021-2022, recalculees.

La transcription de `ElectroniqueInstrumentation/EXAMENS/TheorieCorr.pdf`
porte des resultats et deux ecarts. Ces tests refont chaque raisonnement
depuis les donnees de l'enonce, sans reprendre le chemin du manuscrit.
"""

import math

import pytest
import sympy as sp
from sympy.physics.units import ampere, convert_to, ohm, siemens, volt

from polytech.electronique.domain.amplification import (
    diviseur_d_entree,
    gain_inverseur,
    gain_non_inverseur,
)
from polytech.electronique.domain.boucle import (
    le_gain_de_boucle_est_sans_dimension,
    unite_de_la_transconductance,
    unite_de_la_transimpedance,
)
from polytech.electronique.domain.bruit import densite_spectrale_unilaterale
from polytech.electronique.domain.pole_zero import pente_asymptotique
from polytech.electronique.domain.quadripole import (
    matrice_y_en_l,
    transfert_a_vide_depuis_y,
)
from polytech.electronique.domain.second_ordre import (
    PasseBasCanonique,
    qualite_de_coupure_a_trois_decibels,
)

KILOHM = 1e3

# --- Question 1 : parametres y -------------------------------------------


def test_la_matrice_y_de_la_cellule_rc_est_celle_du_cours() -> None:
    """Question 1.1 : `I = Y V`, ecrite ici pour la cellule de la question 5."""
    s, r, c = sp.symbols("s R C", positive=True)
    y = matrice_y_en_l(r, s * c)
    attendue = sp.Matrix([[1 / r, -1 / r], [-1 / r, 1 / r + s * c]])
    assert sp.simplify(y - attendue) == sp.zeros(2, 2)


def test_la_matrice_y_redonne_le_transfert_a_vide_de_la_question_5() -> None:
    """Question 1.2 et 5.1 : `-y_21/y_22` et le diviseur donnent le meme filtre.

    Les deux questions ne se parlent pas dans le corrige ; qu'elles
    coincident est ce qui prouve que la lecture des parametres y est la
    bonne.
    """
    s, r, c = sp.symbols("s R C", positive=True)
    transfert = transfert_a_vide_depuis_y(matrice_y_en_l(r, s * c))
    assert sp.simplify(transfert - 1 / (s * r * c + 1)) == 0


# --- Question 2 : la cascade et son diviseur d'entree ---------------------


def test_la_cascade_telle_qu_elle_est_cablee_donne_moins_vingt_cinq() -> None:
    """Question 2.1 : l'inverseur charge la source et lui prend la moitie."""
    entree = diviseur_d_entree(1 * KILOHM, 1 * KILOHM)
    premier = gain_inverseur(1 * KILOHM, 5 * KILOHM)
    second = gain_non_inverseur(1 * KILOHM, 9 * KILOHM)
    assert entree == pytest.approx(0.5)
    assert premier == pytest.approx(-5.0)
    assert second == pytest.approx(10.0)
    assert entree * premier * second == pytest.approx(-25.0)


def test_mettre_le_non_inverseur_en_premier_donne_moins_cinquante() -> None:
    """Question 2.2 : son impedance d'entree infinie supprime le diviseur."""
    premier = gain_non_inverseur(1 * KILOHM, 9 * KILOHM)
    second = gain_inverseur(1 * KILOHM, 5 * KILOHM)
    assert premier * second == pytest.approx(-50.0)


def test_une_impedance_d_entree_infinie_ne_divise_rien() -> None:
    """Question 2.2 : c'est la raison invoquee, et elle se chiffre."""
    assert diviseur_d_entree(1 * KILOHM, 1e12) == pytest.approx(1.0, rel=1e-8)


# --- Question 3 : le passe-bas canonique ---------------------------------


def test_le_gain_continu_du_passe_bas_vaut_g() -> None:
    """Question 3.1."""
    gain, pulsation, qualite = sp.symbols("G omega_0 Q", positive=True)
    filtre = PasseBasCanonique(gain, pulsation, qualite)
    assert sp.simplify(filtre.gain_continu() - gain) == 0


def test_la_reponse_a_la_pulsation_propre_vaut_moins_j_g_q() -> None:
    """Question 3.2 : la partie reelle s'annule, il ne reste que le terme en Q."""
    gain, pulsation, qualite = sp.symbols("G omega_0 Q", positive=True)
    filtre = PasseBasCanonique(gain, pulsation, qualite)
    obtenue = filtre.reponse_a_la_pulsation_propre()
    assert sp.simplify(obtenue + sp.I * gain * qualite) == 0


def test_la_coupure_a_trois_decibels_impose_q_egal_un_sur_racine_de_deux() -> None:
    """Question 3.3 : `|H(j omega_0)| / |H(j0)| = Q` doit valoir 1/racine(2)."""
    assert qualite_de_coupure_a_trois_decibels() == pytest.approx(
        1 / math.sqrt(2),
    )


def test_le_rapport_des_modules_vaut_bien_q() -> None:
    """Question 3.3 : c'est cette egalite qui rend la condition si simple."""
    filtre = PasseBasCanonique(3.0, 1000.0, 0.5)
    rapport = abs(complex(filtre.reponse_a_la_pulsation_propre())) / abs(
        complex(filtre.gain_continu()),
    )
    assert rapport == pytest.approx(0.5)


@pytest.mark.parametrize("qualite", [0.1, 0.25, 0.49])
def test_sous_un_demi_les_poles_sont_reels(qualite: float) -> None:
    """Question 3.4 : le discriminant est positif."""
    assert PasseBasCanonique(1.0, 1000.0, qualite).poles_sont_reels()


def test_a_un_demi_exactement_les_poles_sont_encore_reels() -> None:
    """Question 3.4 : le corrige exclut ce point, le discriminant y est nul.

    A `Q = 1/2` les deux poles sont confondus en `-omega_0`, donc reels :
    la condition exacte est `Q <= 1/2`, pas `Q < 1/2`.
    """
    filtre = PasseBasCanonique(1.0, 1000.0, 0.5)
    assert filtre.poles_sont_reels()
    premier, second = filtre.poles()
    assert complex(premier) == pytest.approx(complex(second))
    assert complex(premier).real == pytest.approx(-1000.0)
    assert complex(premier).imag == pytest.approx(0.0)


@pytest.mark.parametrize("qualite", [0.51, 1 / math.sqrt(2), 10.0])
def test_au_dessus_d_un_demi_les_poles_sont_complexes(qualite: float) -> None:
    """Question 3.4 : au-dela, la racine devient imaginaire."""
    assert not PasseBasCanonique(1.0, 1000.0, qualite).poles_sont_reels()


@pytest.mark.parametrize("qualite", [0.01, 0.5, 0.7071, 5.0, 1000.0])
def test_aucun_q_positif_ne_rend_le_filtre_instable(qualite: float) -> None:
    """Question 3.5 : la partie reelle des poles reste negative partout."""
    filtre = PasseBasCanonique(1.0, 1000.0, qualite)
    assert filtre.est_stable()


# --- Question 4 : les quatre poles du Butterworth ------------------------


def test_la_frequence_de_coupure_vaut_700_hertz() -> None:
    """Question 4.2 : le module commun des poles, divise par deux pi."""
    assert 4.398e3 / (2 * math.pi) == pytest.approx(700.0, rel=1e-3)


def test_sept_cents_hertz_font_zero_virgule_sept_kilohertz() -> None:
    """Question 4.2 : la question demande des kilohertz, la reponse des hertz."""
    assert (4.398e3 / (2 * math.pi)) / 1e3 == pytest.approx(0.7, rel=1e-3)


def test_quatre_poles_sans_zero_donnent_moins_quatre_vingts_decibels() -> None:
    """Question 4.4."""
    assert pente_asymptotique(zeros=0, poles=4) == pytest.approx(-80.0)


def test_les_paires_conjuguees_sont_bien_p1_p4_et_p2_p3() -> None:
    """Question 4.1 : seules ces paires donnent des coefficients reels.

    Le produit `(s - p)(s - p*)` est a coefficients reels ; celui de deux
    poles non conjugues ne l'est pas, et c'est exactement l'obstacle que le
    corrige invoque.
    """
    s = sp.Symbol("s")
    p1 = sp.Float(-1700) + sp.I * sp.Float(4050)
    p2 = sp.Float(-4060) + sp.I * sp.Float(1680)
    conjuguee = sp.expand((s - p1) * (s - sp.conjugate(p1)))
    croisee = sp.expand((s - p1) * (s - sp.conjugate(p2)))
    assert sp.Poly(conjuguee, s).all_coeffs() == [
        sp.re(c) for c in sp.Poly(conjuguee, s).all_coeffs()
    ]
    assert any(sp.im(c) != 0 for c in sp.Poly(croisee, s).all_coeffs())


# --- Question 5 : le facteur de bruit de la cellule RC -------------------


def test_le_facteur_de_bruit_vaut_un_plus_r_sur_rs() -> None:
    """Question 5.2 : les deux PSD passent par le meme filtre, qui se simplifie."""
    r, rs, t, k, module = sp.symbols("R R_S T k_B H", positive=True)
    au_total = module * densite_spectrale_unilaterale(r + rs, t, k)
    de_la_source = module * densite_spectrale_unilaterale(rs, t, k)
    assert sp.simplify(au_total / de_la_source - (1 + r / rs)) == 0


def test_le_facteur_de_bruit_ne_depend_pas_de_la_frequence() -> None:
    """Question 5.2 : « frequency-selective » et pourtant plat.

    Les deux bruits traversent le meme filtre, qui se simplifie dans le
    rapport : le facteur ne garde aucune trace de la frequence.
    """
    f, r, rs, t, k, c = sp.symbols("f R R_S T k_B C", positive=True)
    module = 1 / (1 + 4 * sp.pi**2 * f**2 * r**2 * c**2)
    facteur = (module * densite_spectrale_unilaterale(r + rs, t, k)) / (
        module * densite_spectrale_unilaterale(rs, t, k)
    )
    assert sp.simplify(sp.diff(facteur, f)) == 0


# --- Question 6 : les unites de la boucle --------------------------------


def test_la_transconductance_est_un_siemens() -> None:
    """Question 6.2 : `g_ol` convertit une tension en un courant."""
    assert convert_to(unite_de_la_transconductance(), siemens) == siemens
    assert convert_to(unite_de_la_transconductance(), ampere / volt) == ampere / volt


def test_la_transimpedance_est_un_ohm_et_non_un_siemens() -> None:
    """Question 6.2 : `beta` convertit un courant en une tension.

    Le corrige justifie sa reponse par « conversion d'un courant en une
    tension » -- la description de `beta` -- mais donne l'unite de
    `g_ol`. L'ohm manque.
    """
    assert convert_to(unite_de_la_transimpedance(), ohm) == ohm
    assert convert_to(unite_de_la_transimpedance(), siemens) != siemens


def test_le_gain_de_boucle_est_sans_dimension() -> None:
    """Question 6.2 : c'est ce qui force les deux unites a etre inverses."""
    assert le_gain_de_boucle_est_sans_dimension(
        unite_de_la_transconductance(),
        unite_de_la_transimpedance(),
    )
