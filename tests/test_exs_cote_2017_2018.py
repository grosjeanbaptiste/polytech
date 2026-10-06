"""Electromagnetisme : l'exercice cote du 16 mai 2018, cinq questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2017_2018_Solution.pdf`, douze
pages -- enonce imprime, solutions manuscrites --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2017_2018_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle et recoupe par un
autre chemin : la boucle d'Ampere refermee, le produit vectoriel au lieu
des angles, Millman, l'equation differentielle du circuit, les equations
des bobines couplees dans le temps.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2018_bobines as bo
from polytech.electromagnetisme.application import cote_2018_commutation as cm
from polytech.electromagnetisme.application import cote_2018_fils as fi
from polytech.electromagnetisme.application import cote_2018_thevenin as th
from polytech.electromagnetisme.application import cote_2018_tore as to

# --- Question 1 : deux demi-tores de materiaux differents -------------------


def test_q1_le_champ_d_entrefer() -> None:
    """Page 1 : `H_e = 875 352,187` A/m pour 1,1 T."""
    assert to.champ_d_entrefer() == pytest.approx(875_352.187, rel=1e-6, abs=0)


def test_q1_l_epaisseur_de_l_entrefer() -> None:
    """Page 2 : `l_e = 0,496` mm."""
    assert to.epaisseur() * 1e3 == pytest.approx(0.496, abs=5e-4)


def test_q1_la_boucle_d_ampere_se_referme() -> None:
    """Le circuit reconstruit avec cette epaisseur redonne les 1000 A."""
    circuit = to.circuit()
    assert circuit.entrefers == 2
    assert circuit.amperes_tours() == pytest.approx(1000.0, rel=1e-12, abs=0)


def test_q1_le_partage_des_amperes_tours() -> None:
    """L'hipersil en prend 6,6 ; les toles 125,3 ; l'air le reste."""
    hipersil, toles, air = to.partage()
    assert hipersil == pytest.approx(6.597, abs=5e-4)
    assert toles == pytest.approx(125.35, abs=5e-3)
    assert hipersil + toles + air == pytest.approx(1000.0, rel=1e-12, abs=0)
    assert toles / hipersil == pytest.approx(19.0, rel=1e-12, abs=0)


# --- Question 2 : deux fils, champ aligne sur l'axe y -----------------------


def test_q2_la_composante_horizontale_du_manuscrit() -> None:
    """Page 6 : `(I2/(x2^2+y^2) - I1/(x1^2+y^2)) y / (2 pi)`."""
    y = fi.Y
    attendue = (fi.I2 / (fi.X2**2 + y**2) - fi.I1 / (fi.X1**2 + y**2)) * y / (2 * sp.pi)
    assert sp.simplify(fi.champ()[0] - attendue) == 0


def test_q2_les_points_d_alignement() -> None:
    """Page 6 : `y = 0`, et `y^2 = (I1 x2^2 - I2 x1^2)/(I2 - I1)`."""
    nul, carre = fi.points_d_alignement()
    assert nul == 0
    attendu = (fi.I1 * fi.X2**2 - fi.I2 * fi.X1**2) / (fi.I2 - fi.I1)
    assert sp.simplify(carre - attendu) == 0


def test_q2_la_condition_d_existence() -> None:
    """Page 6 : il faut `x2^2/x1^2 > I2/I1`."""
    valeurs = {fi.I1: 1, fi.I2: 2, fi.X1: 1}
    assert fi.points_d_alignement()[1].subs({**valeurs, fi.X2: 2}) > 0
    assert fi.points_d_alignement()[1].subs({**valeurs, fi.X2: 1}) < 0
    assert fi.seuil_d_existence() == sp.sqrt(fi.I2 / fi.I1)


def test_q2_sur_un_exemple_le_champ_y_est_vertical() -> None:
    """En ces points la composante horizontale est nulle, pas la verticale."""
    valeurs = {fi.I1: 1, fi.I2: 2, fi.X1: 1, fi.X2: 2}
    hauteur = sp.sqrt(fi.points_d_alignement()[1].subs(valeurs))
    assert hauteur == sp.sqrt(2)
    horizontal, vertical = (c.subs(valeurs).subs(fi.Y, hauteur) for c in fi.champ())
    assert sp.simplify(horizontal) == 0
    assert sp.simplify(vertical) > 0


def test_q2_a_l_origine_les_deux_champs_s_ajoutent() -> None:
    """En `y = 0` : `H_y = I1/(2 pi x1) + I2/(2 pi x2)`."""
    vertical = fi.champ()[1].subs(fi.Y, 0)
    attendu = fi.I1 / (2 * sp.pi * fi.X1) + fi.I2 / (2 * sp.pi * fi.X2)
    assert sp.simplify(vertical - attendu) == 0


# --- Question 3 : Thevenin avec une source de courant -----------------------


def test_q3_le_generateur_equivalent() -> None:
    """Page 7 : `R_TH = 2,5` ohms, `E_TH = 7,5` V."""
    tension, resistance = th.thevenin()
    assert resistance == sp.Rational(5, 2)
    assert tension == sp.Rational(15, 2)


def test_q3_millman_et_superposition() -> None:
    """Deux autres chemins vers la meme tension."""
    assert th.tension_par_millman() == th.thevenin()[0]
    de_la_tension, du_courant = th.tension_par_superposition()
    assert (de_la_tension, du_courant) == (5, sp.Rational(5, 2))


def test_q3_le_courant_dans_la_charge() -> None:
    """Page 8 : `I = 1` A, retrouve sur le circuit entier."""
    assert th.courant_de_charge() == 1
    assert th.courant_sur_le_reseau_complet() == 1


# --- Question 4 : commutation, systeme en Laplace ---------------------------


def test_q4_la_tension_initiale_du_condensateur() -> None:
    """Page 10 : `E racine(2)/racine(1+(wRC)^2) cos(theta - arctan(wRC))`."""
    attendue = (cm.E * sp.sqrt(2) / sp.sqrt(1 + (cm.W * cm.R * cm.C) ** 2)) * sp.cos(
        cm.THETA - sp.atan(cm.W * cm.R * cm.C)
    )
    valeurs = {cm.E: 3, cm.W: 2, cm.R: 5, cm.C: sp.Rational(1, 7), cm.THETA: 1}
    ecart = (cm.tension_initiale() - attendue).subs(valeurs)
    assert abs(sp.N(ecart, 30)) < sp.Float("1e-25")


def test_q4_cette_tension_verifie_l_equation_du_circuit() -> None:
    """Second chemin, sans phaseur : `R C du/dt + u = e(t)`."""
    avant = cm.tension_avant()
    source = cm.E * sp.sqrt(2) * sp.cos(cm.W * cm.T + cm.THETA)
    reste = cm.R * cm.C * avant.diff(cm.T) + avant - source
    assert sp.simplify(reste) == 0


def test_q4_le_systeme_du_manuscrit() -> None:
    """Page 9 : `L i(0) = (R + pL) I - U_C`, `C u_C(0) = I + pC U_C`."""
    premiere, seconde = cm.systeme()
    p, courant, tension = cm.P, cm.IP, cm.UC
    attendue_1 = (cm.R + p * cm.L) * courant - tension - cm.L * cm.I0
    attendue_2 = courant + p * cm.C * tension - cm.C * cm.U0
    assert sp.simplify(premiere - attendue_1) == 0
    assert sp.simplify(seconde - attendue_2) == 0


def test_q4_l_image_du_courant() -> None:
    """Le systeme resolu, avec `i(0) = 0` : `I = C u0 / (L C p^2 + R C p + 1)`."""
    p = cm.P
    attendue = cm.C * cm.U0 / (cm.L * cm.C * p**2 + cm.R * cm.C * p + 1)
    assert sp.simplify(cm.image_du_courant() - attendue) == 0


def test_q4_le_courant_part_de_zero_avec_la_pente_u0_sur_l() -> None:
    assert cm.depart_du_courant() == (0, cm.U0 / cm.L)


# --- Question 5 : dephasage de deux courants couples ------------------------


def test_q5_le_rapport_des_tensions_mutuelles() -> None:
    """Page 11 : `U21/U12 = -I1/I2`."""
    assert sp.simplify(bo.rapport_des_tensions() + bo.I1 / bo.I2) == 0


def test_q5_le_dephasage() -> None:
    """Page 11 : `theta + pi`."""
    assert bo.dephasage() == bo.THETA + sp.pi


def test_q5_les_tensions_mutuelles_viennent_des_equations_des_bobines() -> None:
    """Dans le temps : `u12 = -L12 di2/dt`, `u21 = +L12 di1/dt`."""
    u12, u21 = bo.tensions_mutuelles_instantanees()
    t = bo.T
    assert sp.simplify(u12 + bo.L12 * bo.courant(2).diff(t)) == 0
    assert sp.simplify(u21 - bo.L12 * bo.courant(1).diff(t)) == 0


def test_q5_verification_dans_le_temps() -> None:
    """Avec `i1` en avance de `theta + pi`, `u21` avance bien de `theta` sur `u12`."""
    assert bo.u21_est_u12_avancee_de(bo.THETA)
    assert not bo.u21_est_u12_avancee_de(bo.THETA + sp.pi)


def test_q5_les_deux_courants_ont_les_amplitudes_de_l_enonce() -> None:
    """A un sixieme de periode du sommet, chacun vaut la moitie de son amplitude."""
    premiere, seconde = bo.AMPLITUDES
    sixieme = sp.pi / (3 * bo.W)
    assert bo.courant(2).subs(bo.T, sixieme) == seconde / 2
    sommet = sp.solve(sp.Eq(bo.W * bo.T + bo.dephasage(), 0), bo.T)[0]
    ecart = bo.courant(1).subs(bo.T, sommet + sixieme) - premiere / 2
    assert sp.simplify(ecart) == 0
