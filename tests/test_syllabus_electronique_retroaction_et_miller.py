"""Le syllabus d'electronique : la retroaction en detail, et Miller a un pole.

Source : `Syllabus_Electronics_and_Instrumentation.pdf`, dans
`ElectroniqueInstrumentation/SYLLABUS/`, sections 3.6.2 a 3.8.3 (pages
114 a 138), lues sur leur couche de texte. Ce fichier porte ce que le
syllabus ajoute aux planches du chapitre 2.

Les oracles : les relations generales de la section 3.6.2 (`A_cl`,
`Z_in,cl`, `Z_out,cl`, `g_12`), les matrices (3.10) a (3.14) de
l'inverseur, son impedance de sortie (3.15), et les poles de
l'amplificateur a retour capacitif de la section 3.8.3.

Trois ecarts.

* La section 3.6.3 refait, comme la planche 50 du chapitre 2, l'erreur
  de facteur sur `Z_in(R_L)` : elle y remplace `1 / Z_in,ol` par `(A_cl /
  A_ol) / Z_in,cl`, l'inverse du rapport.
  `tests/test_planches_electronique_2_1_aop.py` l'etablit par deux chemins.
* La matrice (3.14), annoncee « with as few approximations as possible »,
  neglige `R_o` devant `A_v R_2` dans `g_21`. C'est un millionieme sur
  `g_21`, mais un millieme sur l'impedance de sortie qui en est tiree :
  la lecon de la page 127 sur les petites differences vaut pour elle.
* La page 129 ecrit « beta ~ R_1 / R_2 ~ |A_cl| » : c'est `1 / |A_cl|`,
  ce que sa propre equation (3.16) emploie.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.boucle_fermee import distorsion_apres_retroaction
from polytech.electronique.domain.compensation_de_miller import (
    poles_a_retour_capacitif,
    poles_separes,
    transfert_a_retour_capacitif,
)
from polytech.electronique.domain.conversions_de_quadripole import (
    Jeu,
    associer,
    convertir,
)
from polytech.electronique.domain.netliste import (
    Admittance,
    SourceDeCourant,
    SourceDeTensionCommandee,
    resoudre,
)
from polytech.electronique.domain.retroaction import (
    parametres_h_amplificateur,
    parametres_h_reseau_resistif,
)
from polytech.electronique.domain.retroaction_parallele import (
    impedance_de_sortie_de_l_inverseur,
    parametres_y_de_l_amplificateur,
    parametres_y_de_la_contre_reaction,
)

RIN, RO, AV, R1, R2, BETA = sp.symbols("R_in R_o A_v R_1 R_2 beta", positive=True)
S = sp.Symbol("s")

ORDRES = {RIN: 1e6, RO: 100.0, AV: 1e4, R1: 1e3, R2: 1e4}
"""Page 124 : 100 ohm de sortie, 1 a 100 kohm au pont, `A_v >= 1e4`, 1 Mohm d'entree."""


def test_les_relations_generales_de_la_retroaction_serie_parallele() -> None:
    """Section 3.6.2 : avec un reseau de retour ideal, tout se lit sur `G`."""
    ideale = sp.Matrix([[0, BETA], [0, 0]])
    globale = associer(Jeu.H, parametres_h_amplificateur(RIN, RO, AV), ideale)
    inverse = convertir(globale, depuis=Jeu.H, vers=Jeu.G)
    boucle = 1 + BETA * AV
    assert sp.simplify(inverse[1, 0] - AV / boucle) == 0
    assert sp.simplify(1 / inverse[0, 0] - RIN * boucle) == 0
    assert sp.simplify(inverse[1, 1] - RO / boucle) == 0
    assert sp.simplify(inverse[0, 1] + BETA * (RO / RIN) / boucle) == 0


def test_le_reseau_resistif_vaut_le_reseau_ideal_aux_ordres_de_grandeur() -> None:
    """Pages 116 et 124 : `h_11`, `h_21`, `h_22` restent ceux de l'amplificateur."""
    amplificateur = parametres_h_amplificateur(RIN, RO, AV)
    reseau = parametres_h_reseau_resistif(R1, R2)
    assert sp.simplify(reseau[0, 1] - R1 / (R1 + R2)) == 0
    assert sp.simplify(reseau[0, 0] - R1 * R2 / (R1 + R2)) == 0
    assert sp.simplify(reseau[1, 0] + R1 / (R1 + R2)) == 0
    assert sp.simplify(reseau[1, 1] - 1 / (R1 + R2)) == 0
    for ligne, colonne in ((0, 0), (1, 0), (1, 1)):
        rapport = (reseau[ligne, colonne] / amplificateur[ligne, colonne]).subs(ORDRES)
        assert abs(float(rapport)) < 0.01


def test_le_coefficient_de_retour_en_courant_de_la_page_120() -> None:
    """`g_12 ~ -(Z_out,ol / Z_in,ol) / A_ol` : 1 Mohm, 100 ohm, 1e4 donnent -1e-8."""
    exact = -BETA * (RO / RIN) / (1 + BETA * AV)
    chiffres = {RIN: 1e6, RO: 100.0, AV: 1e4, BETA: sp.Rational(1, 10)}
    assert float(exact.subs(chiffres)) == pytest.approx(-1e-8, rel=2e-3, abs=0)


def test_les_matrices_y_de_l_inverseur() -> None:
    """(3.10) et (3.11) : la topologie parallele-parallele additionne les `Y`."""
    amplificateur = parametres_y_de_l_amplificateur(RIN, RO, AV)
    assert amplificateur == sp.Matrix([[1 / RIN, 0], [AV / RO, 1 / RO]])
    retour = parametres_y_de_la_contre_reaction(R2)
    assert retour == sp.Matrix([[1 / R2, -1 / R2], [-1 / R2, 1 / R2]])
    globale = associer(Jeu.Y, amplificateur, retour)
    attendue = sp.Matrix(
        [[1 / RIN + 1 / R2, -1 / R2], [AV / RO - 1 / R2, 1 / RO + 1 / R2]]
    )
    assert sp.simplify(globale - attendue) == sp.zeros(2, 2)
    with pytest.raises(ValueError, match="resistance de sortie nulle"):
        parametres_y_de_l_amplificateur(RIN, 0, AV)


def _g_de_l_inverseur() -> sp.Matrix:
    """La matrice `G` exacte de l'inverseur sans `R_1`, pour `R_in` infinie."""
    globale = associer(
        Jeu.Y,
        parametres_y_de_l_amplificateur(RIN, RO, AV),
        parametres_y_de_la_contre_reaction(R2),
    )
    inverse = convertir(globale, depuis=Jeu.Y, vers=Jeu.G)
    return inverse.applyfunc(lambda terme: sp.limit(terme, RIN, sp.oo))


G_DU_SYLLABUS = sp.Matrix(
    [
        [(1 + AV) / (R2 + RO), -RO / (R2 + RO)],
        [-AV * R2 / (R2 + RO), RO * R2 / (RO + R2)],
    ]
)
"""(3.14), telle qu'imprimee."""


def test_la_matrice_g_de_l_inverseur() -> None:
    """(3.13) et (3.14) : exactes a `R_in` infinie, sauf `R_o` neglige dans `g_21`."""
    exacte = _g_de_l_inverseur()
    for ligne, colonne in ((0, 0), (0, 1), (1, 1)):
        ecart = exacte[ligne, colonne] - G_DU_SYLLABUS[ligne, colonne]
        assert sp.simplify(ecart) == 0
    assert sp.simplify(exacte[1, 0] - (RO - AV * R2) / (R2 + RO)) == 0
    relatif = (exacte[1, 0] / G_DU_SYLLABUS[1, 0] - 1).subs(ORDRES)
    assert float(relatif) == pytest.approx(-1e-6, rel=1e-6, abs=0)
    grossiere = sp.Matrix([[(1 + AV) / R2, -RO / R2], [-AV, RO * R2 / (RO + R2)]])
    for ligne in range(2):
        for colonne in range(2):
            rapport = exacte[ligne, colonne] / grossiere[ligne, colonne]
            assert float(rapport.subs(ORDRES)) == pytest.approx(1.0, abs=0.011)


def test_le_gain_et_l_impedance_d_entree_de_l_inverseur_par_sa_matrice_g() -> None:
    """Page 126 : `V_1 / V_in` est un diviseur ; `A_cl ~ -R_2 / R_1`, `Z_in ~ R_1`."""
    vue = R2 / (1 + AV)
    gain = -AV * vue / (vue + R1)
    assert sp.limit(gain, AV, sp.oo) == -R2 / R1
    assert float(gain.subs(ORDRES)) == pytest.approx(-10.0, rel=2e-3)
    assert float((vue + R1).subs(ORDRES)) == pytest.approx(1e3, rel=2e-3)


def _sortie_par_la_matrice(inverse: sp.Matrix) -> sp.Expr:
    """`g_22 - g_21 g_12 / (g_11 + 1 / R_1)` : l'entree fermee sur `R_1`."""
    retour = inverse[1, 0] * inverse[0, 1] / (inverse[0, 0] + 1 / R1)
    return sp.simplify(inverse[1, 1] - retour)


def test_l_impedance_de_sortie_de_l_inverseur_exige_un_calcul_exact() -> None:
    """Pages 127 a 129 : `R_o (R_1 + R_2) / (R_o + R_1 + R_2 + A_v R_1)`."""
    exacte = impedance_de_sortie_de_l_inverseur(RO, AV, vers_l_entree=R1, retour=R2)
    assert sp.simplify(exacte - RO * (R1 + R2) / (RO + R1 + R2 + AV * R1)) == 0
    elements = (
        SourceDeCourant(0, 2, sp.Integer(1)),
        Admittance(1, 0, 1 / R1),
        Admittance(1, 2, 1 / R2),
        Admittance(3, 2, 1 / RO),
        SourceDeTensionCommandee(3, 0, -AV, (1, 0), sp.Symbol("i_a")),
    )
    assert sp.simplify(resoudre(elements, 3)[1] - exacte) == 0
    assert sp.simplify(_sortie_par_la_matrice(_g_de_l_inverseur()) - exacte) == 0
    assert float(exacte.subs(ORDRES)) == pytest.approx(1.1e6 / 10_011_100, rel=1e-9)


def test_l_impedance_de_sortie_imprimee_porte_l_approximation_de_3_14() -> None:
    """(3.15) : `(R_o // R_2) / (1 + R_1 A_v / (R_1 + R_2 + R_o))`, a un millieme."""
    exacte = impedance_de_sortie_de_l_inverseur(RO, AV, vers_l_entree=R1, retour=R2)
    imprimee = RO * R2 / (RO + R2) / (R1 * AV / (R1 + R2 + RO) + 1)
    assert sp.simplify(_sortie_par_la_matrice(G_DU_SYLLABUS) - imprimee) == 0
    rapport = sp.simplify(imprimee / exacte)
    assert sp.simplify(rapport - (1 - R1 * RO / ((RO + R2) * (R1 + R2)))) == 0
    assert float(rapport.subs(ORDRES)) == pytest.approx(1 - 9e-4, abs=2e-5)
    finale = RO / (1 + R1 / (R1 + R2) * AV)
    assert float((finale / exacte).subs(ORDRES)) == pytest.approx(1.0, abs=2e-3)


def test_une_petite_difference_ne_se_calcule_pas_par_approximation() -> None:
    """Page 127 : `A_0 / (1 - eps) - A_0 = A_0 eps / (1 - eps)`, et non zero."""
    a0, eps = sp.symbols("A_0 epsilon", positive=True)
    difference = a0 / (1 - eps) - a0
    assert sp.simplify(difference - a0 * eps / (1 - eps)) == 0
    assert difference.subs(eps, sp.Rational(1, 1000)) != 0
    assert sp.limit(difference / (a0 * eps), eps, 0) == 1


def test_la_retroaction_linearise_un_amplificateur_non_lineaire() -> None:
    """Page 131 : `A_v = 1e4 + 10 (V+ - V-)` ; le gain ferme n'en depend plus."""
    retour = 0.1
    fermes = [gain / (1 + retour * gain) for gain in (1e4, 1e4 + 10 * 0.5)]
    assert fermes[1] / fermes[0] - 1 == pytest.approx(5e-7, rel=0.01, abs=0)
    ouverts = (1e4 + 10 * 0.5) / 1e4 - 1
    assert ouverts == pytest.approx(5e-4, rel=1e-9, abs=0)
    attenuee = distorsion_apres_retroaction(ouverts, 1e4, retour)
    assert attenuee == pytest.approx(ouverts / (1 + retour * 1e4), rel=1e-9, abs=0)


ADC, WC, TAU = 1e4, 100.0, 1e-3
"""Section 3.8.3 : gain continu, pulsation de coupure, et `beta = C_f R_S`."""


def test_le_transfert_de_l_amplificateur_a_retour_capacitif() -> None:
    """Page 137 : `-A_DC / ((beta / w_c) s^2 + s (beta (1 + A_DC) + 1 / w_c) + 1)`."""
    a, w, b = sp.symbols("A_DC omega_c beta", positive=True)
    transfert = transfert_a_retour_capacitif(S, continu=a, coupure=w, constante=b)
    attendu = -a / (b / w * S**2 + S * (b * (1 + a) + 1 / w) + 1)
    assert sp.simplify(transfert - attendu) == 0
    amplificateur = a / (1 + S / w)
    par_miller = -amplificateur / (1 + S * b * (1 + amplificateur))
    assert sp.simplify(transfert - par_miller) == 0
    parfait = sp.limit(transfert, w, sp.oo)
    assert sp.simplify(parfait + a / (S * b * (1 + a) + 1)) == 0


def test_les_poles_se_separent() -> None:
    """Page 138 : `p_1 ~ -1 / (beta (1 + A_DC))`, `p_2 ~ -w_c (1 + A_DC)`, par Viete."""
    lent, rapide = poles_a_retour_capacitif(continu=ADC, coupure=WC, constante=TAU)
    assert lent * rapide == pytest.approx(WC / TAU, rel=1e-9)
    somme = -(WC * (1 + ADC) + 1 / TAU)
    assert lent + rapide == pytest.approx(somme, rel=1e-9)
    approche_lent, approche_rapide = poles_separes(
        continu=ADC, coupure=WC, constante=TAU
    )
    assert approche_lent == pytest.approx(-1 / (TAU * (1 + ADC)), rel=1e-12)
    assert approche_rapide == pytest.approx(-WC * (1 + ADC), rel=1e-12)
    assert lent == pytest.approx(approche_lent, rel=2e-3)
    assert rapide == pytest.approx(approche_rapide, rel=2e-3)
    assert approche_lent * approche_rapide == pytest.approx(WC / TAU, rel=1e-9)
    with pytest.raises(ValueError, match="grandeur nulle ou negative"):
        poles_a_retour_capacitif(continu=ADC, coupure=0.0, constante=TAU)


def test_l_approximation_de_miller_vaut_sous_le_dixieme_de_la_coupure() -> None:
    """Page 138 : pour `w < 0,1 w_c`, `Z_1` est la capacite `C_f (1 + A_DC)`."""

    def capacite_relative(pulsation: float) -> complex:
        amplificateur = ADC / (1 + 1j * pulsation / WC)
        return (1 + amplificateur) / (1 + ADC)

    assert abs(capacite_relative(0.01 * WC) - 1) < 0.0101
    assert abs(capacite_relative(0.1 * WC) - 1) < 0.101
    assert abs(capacite_relative(10 * WC) - 1) > 0.9


def test_le_transfert_survit_a_l_approximation_bien_au_dela() -> None:
    """Le second pole est en `w_c (1 + A_DC)` : le transfert tient jusque-la."""
    a, w, b = sp.symbols("A_DC omega_c beta", positive=True)
    transfert = transfert_a_retour_capacitif(S, continu=a, coupure=w, constante=b)
    miller = -a / (S * b * (1 + a) + 1)
    chiffres = {a: ADC, w: WC, b: TAU}

    def ecart(pulsation: float) -> float:
        exact = complex(transfert.subs(chiffres).subs(S, 1j * pulsation))
        approche = complex(miller.subs(chiffres).subs(S, 1j * pulsation))
        return abs(exact / approche - 1)

    assert ecart(10 * WC) < 2e-3
    assert ecart(WC * (1 + ADC)) > 0.25


def test_les_gardes_de_la_retroaction_parallele_et_de_miller() -> None:
    """Une resistance nulle ou un gain nul ne decrivent aucun des deux montages."""
    with pytest.raises(ValueError, match="d'entree nulle"):
        parametres_y_de_l_amplificateur(0, RO, AV)
    with pytest.raises(ValueError, match="resistance de contre-reaction nulle"):
        parametres_y_de_la_contre_reaction(0)
    with pytest.raises(ValueError, match="grandeur nulle ou negative"):
        poles_separes(continu=0.0, coupure=WC, constante=TAU)
