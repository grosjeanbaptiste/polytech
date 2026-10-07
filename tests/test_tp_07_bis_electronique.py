"""Electronique : l'annexe du TP 7, verifiee table par table.

Source : `ElectroniqueInstrumentation/TP/TP_7_Annexe.pdf`, neuf pages. C'est
un formulaire, sans exercice : mise a l'echelle d'impedance, trois
transformations en frequence, deux tables de filtres en echelle, le
filtre de Sallen-Key et ses quatre methodes. Chaque table et chaque
formule y est recopiee comme oracle, et recalculee par la bibliotheque.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.application import annexe_tp_07 as an
from polytech.electronique.application import annexe_tp_07_actifs as ac
from polytech.electronique.domain.methodes_sallen_key import Methode

S = sp.Symbol("s")

# --- Section 1 : la mise a l'echelle d'impedance ----------------------------


def test_l_exemple_de_la_page_1() -> None:
    """Page 1 : pour un facteur 10, 5 microhenrys en font 50, et 20 nF en font 2."""
    _, inductance, capacite = an.exemple_de_mise_a_l_echelle()
    assert inductance == pytest.approx(50e-6)
    assert capacite == pytest.approx(2e-9)


def test_la_mise_a_l_echelle_ne_change_pas_le_transfert() -> None:
    """Page 1 : les tensions des noeuds restent inchangees."""
    assert sp.simplify(an.echelle_mise_a_l_echelle(50) - an.prototype()) == 0


# --- Section 2 : les trois transformations en frequence ---------------------


def test_passe_bas_vers_passe_bas() -> None:
    """Page 1 : `L -> (Oc/wc) L`, `C -> (Oc/wc) C` deplace la coupure de 1 a `wc`."""
    coupure = 1000
    attendu = an.prototype().subs(S, S / coupure)
    assert sp.simplify(an.vers_passe_bas(coupure) - attendu) == 0


def test_passe_bas_vers_passe_haut() -> None:
    """Page 2, figure 1 : `C -> L = 1/(wc Oc C)`, `L -> C = 1/(wc Oc L)`."""
    coupure = 1000
    attendu = an.prototype().subs(S, coupure / S)
    assert sp.simplify(an.vers_passe_haut(coupure) - attendu) == 0


def test_passe_bas_vers_passe_bande() -> None:
    """Page 3, figure 2 : chaque element devient un circuit accorde sur `w0`."""
    centre, largeur = 1000, 100
    attendu = an.prototype().subs(S, (S**2 + centre**2) / (largeur * S))
    assert sp.simplify(an.vers_passe_bande(centre, largeur) - attendu) == 0


def test_le_passe_bande_garde_le_gain_au_centre() -> None:
    """En `s = j w0` le passe-bande vaut ce que le prototype vaut en continu."""
    au_centre = an.vers_passe_bande(1000, 100).subs(S, 1000 * sp.I)
    assert sp.simplify(au_centre - an.prototype().subs(S, 0)) == 0


# --- Section 3 : les deux tables de filtres en echelle ----------------------

BUTTERWORTH = {
    1: (2.000,),
    2: (1.4142, 1.4142),
    3: (1.0000, 2.0000, 1.0000),
    4: (0.7654, 1.8478, 1.8478, 0.7654),
    5: (0.6180, 1.6180, 2.000, 1.6180, 0.6180),
}
CHEBYSHEV = {
    1: (0.3052, 1.0),
    2: (0.8431, 0.6220, 1.3554),
    3: (1.0316, 1.1474, 1.0316, 1.0),
    4: (1.1088, 1.3062, 1.7704, 0.8181, 1.3554),
    5: (1.1468, 1.3712, 1.9750, 1.3712, 1.1468, 1.0),
}


@pytest.mark.parametrize("ordre", [1, 2, 3, 4, 5])
def test_la_table_1_de_butterworth(ordre: int) -> None:
    """Page 5, table 1 : `g_1` a `g_N` ; la charge `g_(N+1)` vaut toujours un."""
    assert an.elements_de_butterworth(ordre) == pytest.approx(
        BUTTERWORTH[ordre], abs=5e-5
    )


@pytest.mark.parametrize("ordre", [1, 2, 3, 4, 5])
def test_la_table_2_de_chebyshev(ordre: int) -> None:
    """Page 5, table 2, ondulation de 0,1 dB : `g_1` a `g_(N+1)`.

    A un dix-millieme pres : la bibliotheque emploie la constante 17,37
    du cours, arrondie de `40/ln 10 = 17,3718`.
    """
    assert an.elements_de_chebyshev(ordre) == pytest.approx(CHEBYSHEV[ordre], abs=1e-4)


def test_le_premier_coefficient_de_chebyshev_sans_table() -> None:
    """Second chemin : a l'ordre 1, `g_1 = 2 epsilon = 0,30524`, la table dit 0,3052."""
    assert an.premier_element_exact() == pytest.approx(0.30524, abs=5e-6)
    assert an.premier_element_exact() == pytest.approx(CHEBYSHEV[1][0], abs=5e-5)
    ecart = an.elements_de_chebyshev(1)[0] - an.premier_element_exact()
    assert ecart == pytest.approx(1.7e-5, abs=5e-6)


def test_la_charge_de_chebyshev_differe_aux_ordres_pairs() -> None:
    """Page 5 : « on peut avoir `g_(N+1)` different de un », 1,3554 aux ordres pairs."""
    assert [
        an.elements_de_chebyshev(n)[-1] == pytest.approx(1) for n in range(1, 6)
    ] == [
        True,
        False,
        True,
        False,
        True,
    ]


def test_le_prototype_de_butterworth_coupe_a_un_radian_par_seconde() -> None:
    """Second chemin : a `w = 1` le module a perdu 3 dB sur sa valeur en continu."""
    continu, coupure = an.modules_du_prototype()
    assert continu == pytest.approx(0.5)
    assert coupure / continu == pytest.approx(2**-0.5)


# --- Section 4 : le filtre de Sallen-Key ------------------------------------


def test_la_fonction_de_transfert_de_la_page_7() -> None:
    """Page 7 : celle que l'annexe dit obtenue « par analyse nodale modifiee »."""
    assert (
        sp.simplify(ac.transfert_par_l_analyse_nodale() - ac.transfert_de_l_annexe())
        == 0
    )


def test_la_pulsation_et_la_qualite_de_la_page_7() -> None:
    """Page 7 : `w0 = 1/racine(R1 R2 C1 C2)`, et `Q` ; `H(0) = mu`."""
    pulsation, qualite = ac.pulsation_et_qualite()
    reference = ac.MU * pulsation**2 / (S**2 + pulsation / qualite * S + pulsation**2)
    assert sp.simplify(ac.transfert_de_l_annexe() - reference) == 0
    assert ac.transfert_de_l_annexe().subs(S, 0) == ac.MU


@pytest.mark.parametrize("methode", list(Methode))
def test_les_quatre_methodes_donnent_la_pulsation_et_la_qualite_voulues(
    methode: Methode,
) -> None:
    """Pages 7 et 8 : les composants de chaque methode rendent bien `w0` et `Q`."""
    assert ac.pulsation_et_qualite_de_la_methode(methode) == (ac.W0, ac.Q)


def test_les_gains_des_quatre_methodes() -> None:
    """Pages 7 et 8 : `3 - 1/Q`, 2, 4/3 et 1."""
    gains = [ac.composants(methode).gain for methode in Methode]
    assert gains == [3 - 1 / ac.Q, 2, sp.Rational(4, 3), 1]


def test_les_deux_rapports_que_l_annexe_signale() -> None:
    """Page 8 : `R1/R2 = Q^2` (methode 2), `C1/C2 = 4 Q^2` (methode 4)."""
    deux = ac.composants(Methode.CAPACITES_EGALES)
    quatre = ac.composants(Methode.SENSIBILITE_MINIMALE)
    assert sp.simplify(deux.r1 / deux.r2 - ac.Q**2) == 0
    assert sp.simplify(quatre.c1 / quatre.c2 - 4 * ac.Q**2) == 0


def test_la_contrainte_des_courants_de_polarisation() -> None:
    """Page 9 : `R1 + R2 = Ra // Rb`, pour un gain et des resistances donnes."""
    ra, rb = ac.pont_de_gain(gain=2.0, r1=4000.0, r2=6000.0)
    assert 1 + rb / ra == pytest.approx(2.0)
    assert ra * rb / (ra + rb) == pytest.approx(10000.0)


def test_mettre_le_pont_a_l_echelle_ne_change_pas_le_gain() -> None:
    """Page 7 : `mu = 1 + Rb/Ra` reste inchange si les deux sont multipliees."""
    ra, rb = ac.pont_de_gain(gain=2.0, r1=4000.0, r2=6000.0)
    assert 1 + (7 * rb) / (7 * ra) == pytest.approx(1 + rb / ra)
