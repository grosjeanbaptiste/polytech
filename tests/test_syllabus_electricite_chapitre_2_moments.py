"""Syllabus d'electricite generale, chapitre 2 : centres, moments, multipoles.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphes 2.2 a 2.5 et 2.7, pages 2-2 a 2-12 et 2-24 a 2-36, lues sur les
pages rendues.

Un ecart, constate en agrandissant la page 2-7 : l'exemple 2.3.3 conclut
`p = (4/3) pi a^2 sigma_0 u_z`, alors que sa ligne precedente porte `a^3`
et que l'integrale `2 pi sigma_0 a^3 * 2/3` le donne. Un moment est une
charge fois une longueur : `sigma_0 a^3`, et non `sigma_0 a^2`.

La legende de la figure 2.5.1 parle de « la deuxieme propriete » dans le
paragraphe de la premiere.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import (
    CHARGE_ELEMENTAIRE,
    PERMITTIVITE_DU_VIDE,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.lois import (
    est_electrostatique,
    rotationnel_cartesien,
)
from polytech.electromagnetisme.domain.electrostatique.moments import (
    SystemeDeCharges,
    moment_d_un_doublet,
    moment_d_un_filet,
    moment_d_une_coque,
)
from polytech.electromagnetisme.domain.electrostatique.multipoles import (
    champ_du_dipole,
    contribution_dipolaire,
    contribution_quadrupolaire,
    contribution_unipolaire,
    potentiel_du_dipole,
)
from polytech.electromagnetisme.domain.electrostatique.potentiel import (
    COULOMB,
    champ_du_potentiel,
    est_une_ligne_de_champ,
    potentiel_des_charges,
)
from polytech.electromagnetisme.domain.symboles import (
    ABSCISSE,
    CARTESIENNES,
    COTE,
    ORDONNEE,
)

X, Y, Z = CARTESIENNES
TROIS = SystemeDeCharges(((2, (0, 0, 0)), (2, (2, 0, 0)), (-1, (0, 3, 0))))
MEMES = SystemeDeCharges(((1, (0, 0, 1)), (1, (0, 0, -1))))
OPPOSEES = SystemeDeCharges(((1, (0, 0, 1)), (-1, (0, 0, -1))))
QUADRUPOLE = SystemeDeCharges(((1, (0, 0, 1)), (-2, (0, 0, 0)), (1, (0, 0, -1))))
K = COULOMB


def _en(expression: sp.Expr, point: tuple[float, float, float]) -> float:
    valeurs = dict(zip(CARTESIENNES, point, strict=True))
    return float(expression.subs(valeurs).subs(PERMITTIVITE, PERMITTIVITE_DU_VIDE))


# --- 2.2 Centre electrique -----------------------------------------------------


def test_les_charges_totales_positive_et_negative() -> None:
    """(2.2.2) : Q+ = 4, Q- = -1, Q = 3."""
    assert (TROIS.charge_positive(), TROIS.charge_negative()) == (4, -1)
    assert TROIS.charge_totale() == 3


def test_les_centres_des_charges_positives_et_negatives() -> None:
    """(2.2.1) : r+ = somme des q r / Q+ sur les seules charges positives."""
    assert TROIS.centre_des_positives() == (1, 0, 0)
    assert TROIS.centre_des_negatives() == (0, 3, 0)


def test_le_centre_electrique_du_systeme() -> None:
    """(2.2.3) : r_ce = (Q+ r+ + Q- r-) / (Q+ + Q-)."""
    assert TROIS.centre_electrique() == (sp.Rational(4, 3), -1, 0)


def test_un_systeme_neutre_n_a_pas_de_centre_electrique() -> None:
    """Remarque de la page 2-3 : la charge totale « peut etre nulle »."""
    with pytest.raises(ValueError, match="charge totale nulle"):
        OPPOSEES.centre_electrique()


def test_un_systeme_sans_charge_negative_n_a_pas_de_centre_negatif() -> None:
    """Q- est nulle : le quotient (2.2.1b) n'existe pas."""
    with pytest.raises(ValueError, match="aucune charge de ce signe"):
        MEMES.centre_des_negatives()


# --- 2.3 Moment electrique, dipole ---------------------------------------------


def test_le_moment_electrique_par_rapport_a_un_point() -> None:
    """(2.3.1) : p(r_A) = somme des q_i (r_i - r_A)."""
    assert TROIS.moment_electrique() == (4, -3, 0)
    assert TROIS.moment_electrique((1, 1, 1)) == (1, -6, -3)


def test_le_moment_change_avec_le_point_de_reference() -> None:
    """Remarque de la page 2-3 : p(A') = p(A) - Q AA'."""
    depuis_a, vers = TROIS.moment_electrique((1, 0, 0)), (5, -2, 7)
    attendu = tuple(
        p - TROIS.charge_totale() * (b - a)
        for p, a, b in zip(depuis_a, (1, 0, 0), vers, strict=True)
    )
    assert TROIS.moment_electrique(vers) == attendu


def test_le_moment_est_nul_par_rapport_au_centre_electrique() -> None:
    """Remarque de la page 2-5 : p(r_ce) = 0 quand Q n'est pas nulle."""
    assert TROIS.moment_electrique(TROIS.centre_electrique()) == (0, 0, 0)


def test_la_decomposition_du_moment_en_deux_termes() -> None:
    """(2.3.7) : Q/2 (r+ + r- - 2 r_A) + (Q+ - Q-)/2 (r+ - r-)."""
    plus, moins = TROIS.centre_des_positives(), TROIS.centre_des_negatives()
    q, ecart = TROIS.charge_totale(), TROIS.charge_positive() - TROIS.charge_negative()
    reference = (1, 1, 1)
    somme = tuple(
        sp.Rational(q, 2) * (p + m - 2 * a) + sp.Rational(ecart, 2) * (p - m)
        for p, m, a in zip(plus, moins, reference, strict=True)
    )
    assert TROIS.moment_electrique(reference) == somme


def test_le_moment_dipolaire_d_un_doublet() -> None:
    """(2.3.4) : p = q d, d allant de la charge negative a la positive."""
    assert moment_d_un_doublet(3, negative=(0, 0, -1), positive=(0, 0, 1)) == (0, 0, 6)
    assert OPPOSEES.moment_electrique() == (0, 0, 2)


def test_le_moment_d_un_systeme_neutre_ne_depend_pas_du_point() -> None:
    """Page 2-3 : le moment du doublet « est independant du choix du point A »."""
    assert OPPOSEES.moment_electrique((7, -3, 2)) == OPPOSEES.moment_electrique()


def test_la_molecule_d_eau_de_l_exemple_2_3_1() -> None:
    """O-H de 97 pm, angle de 104,30 degres : 6,1 . 10^-30 C.m pour alpha = 0,32."""
    liaison, demi_angle = 97e-12, math.radians(104.30 / 2)
    alpha = 0.32
    hydrogenes = [
        (
            alpha * CHARGE_ELEMENTAIRE,
            (
                liaison * math.cos(demi_angle),
                signe * liaison * math.sin(demi_angle),
                0.0,
            ),
        )
        for signe in (1, -1)
    ]
    eau = SystemeDeCharges(
        ((-2 * alpha * CHARGE_ELEMENTAIRE, (0.0, 0.0, 0.0)), *hydrogenes)
    )
    px, py, pz = eau.moment_electrique()
    assert math.isclose(px, 6.1e-30, rel_tol=0.01)
    assert (math.isclose(py, 0.0, abs_tol=1e-40), pz) == (True, 0)
    assert math.isclose(eau.charge_totale(), 0.0, abs_tol=1e-30)


def test_le_segment_charge_de_l_exemple_2_3_2() -> None:
    """+lambda_0 de -l a 0, -lambda_0 de 0 a l : p = -lambda_0 l^2 u_x."""
    lambda_0, demi = sp.symbols("lambda_0 l", positive=True)
    densite = sp.Piecewise((lambda_0, ABSCISSE < 0), (-lambda_0, True))
    assert moment_d_un_filet(densite, -demi, demi) == -lambda_0 * demi**2
    assert sp.integrate(densite, (ABSCISSE, -demi, demi)) == 0


def test_la_coque_spherique_de_l_exemple_2_3_3() -> None:
    """sigma_0 cos(theta) sur une sphere de rayon a : (4/3) pi a^3 sigma_0 u_z."""
    sigma_0, rayon = sp.symbols("sigma_0 a", positive=True)
    theta = sp.Symbol("theta", real=True)
    moment = moment_d_une_coque(sigma_0 * sp.cos(theta), rayon, theta)
    assert moment == (0, 0, sp.Rational(4, 3) * sp.pi * rayon**3 * sigma_0)
    assert moment[2] != sp.Rational(4, 3) * sp.pi * rayon**2 * sigma_0


def test_la_coque_de_l_exemple_2_3_3_est_neutre() -> None:
    """« A titre d'exercice, on verifiera que la charge totale est bien nulle »."""
    sigma_0, rayon = sp.symbols("sigma_0 a", positive=True)
    theta = sp.Symbol("theta", real=True)
    element = sigma_0 * sp.cos(theta) * rayon**2 * sp.sin(theta)
    assert sp.integrate(element, (theta, 0, sp.pi)) * 2 * sp.pi == 0


# --- 2.5 Premiere propriete fondamentale ---------------------------------------


def test_le_champ_de_l_exemple_2_5_1_n_est_pas_electrostatique() -> None:
    """Son rotationnel vaut (E_0 / a^2) (-2y, x, 0) : il ne derive d'aucun potentiel."""
    e_0, a = sp.symbols("E_0 a", positive=True)
    champ = (e_0 / a**2 * X * Z, e_0 / a**2 * 2 * Y * Z, sp.Integer(0))
    assert rotationnel_cartesien(champ) == (-2 * e_0 * Y / a**2, e_0 * X / a**2, 0)
    assert not est_electrostatique(champ)


def test_les_lignes_de_champ_de_l_exemple_2_5_1_sont_des_paraboles() -> None:
    """(2.5.8) : dl ^ E = 0 le long de y = C x^2, a z constant."""
    parametre, constante, cote = sp.symbols("t C z_0", real=True)
    champ = (X * Z, 2 * Y * Z, sp.Integer(0))
    parabole = (parametre, constante * parametre**2, cote)
    assert est_une_ligne_de_champ(champ, parabole, parametre)
    droite = (parametre, constante * parametre, cote)
    assert not est_une_ligne_de_champ(champ, droite, parametre)


def test_le_champ_derive_du_potentiel() -> None:
    """(2.5.5) : E = -grad V ; un tel champ est irrotationnel."""
    potentiel = X**2 * Y - Z
    champ = champ_du_potentiel(potentiel)
    assert champ == (-2 * X * Y, -(X**2), 1)
    assert est_electrostatique(champ)


# --- 2.6 Deuxieme propriete : le potentiel newtonien ------------------------


def test_le_potentiel_d_une_charge_ponctuelle() -> None:
    """Exemple 2.6.1 : V = q / (4 pi eps_0 r) -- 8,99 V a un metre d'un nanocoulomb."""
    seule = SystemeDeCharges(((1e-9, (0, 0, 0)),))
    potentiel = potentiel_des_charges(seule)
    attendu = 1e-9 / (4 * math.pi * PERMITTIVITE_DU_VIDE * 5.0)
    assert math.isclose(_en(potentiel, (3, 0, 4)), attendu)
    assert math.isclose(_en(potentiel, (1, 0, 0)), 8.99, rel_tol=1e-3)


def test_le_champ_d_une_charge_ponctuelle_est_radial_et_en_inverse_du_carre() -> None:
    """Exemple 2.6.1 : E = q u_r / (4 pi eps_0 r^2), vers l'exterieur si q > 0."""
    seule = SystemeDeCharges(((1e-9, (0, 0, 0)),))
    ex, ey, ez = (
        _en(c, (0, 0, 2)) for c in champ_du_potentiel(potentiel_des_charges(seule))
    )
    assert (ex, ey) == (0, 0)
    assert math.isclose(ez, 1e-9 / (4 * math.pi * PERMITTIVITE_DU_VIDE * 4.0))


def test_la_permittivite_du_vide_du_syllabus() -> None:
    """Page 2-13 : 8,854 187 817 . 10^-12, soit 8,85 . 10^-12 F/m."""
    assert math.isclose(PERMITTIVITE_DU_VIDE, 8.854187817e-12, rel_tol=1e-9)
    assert math.isclose(PERMITTIVITE_DU_VIDE, 8.85e-12, rel_tol=1e-3)


# --- 2.7 Multipoles -----------------------------------------------------------


def test_le_developpement_de_l_inverse_de_la_distance() -> None:
    """Page 2-25 : (1 + x)^(-1/2) = 1 - x/2 + 3 x^2 / 8 - ..."""
    x = sp.Symbol("x")
    assert sp.series((1 + x) ** sp.Rational(-1, 2), x, 0, 3).removeO() == (
        1 - x / 2 + 3 * x**2 / 8
    )


def test_deux_charges_identiques_vues_de_loin() -> None:
    """Exemple 2.7.1 : V ~ 2q / (4 pi eps_0 r), et le terme dipolaire est nul."""
    rayon = sp.sqrt(X**2 + Y**2 + Z**2)
    assert sp.simplify(contribution_unipolaire(MEMES) - 2 * K / rayon) == 0
    assert contribution_dipolaire(MEMES) == 0


def test_l_approximation_unipolaire_s_ameliore_avec_la_distance() -> None:
    """Figure 2.7.2 : loin des charges, les equipotentielles deviennent des spheres."""
    exact, approche = potentiel_des_charges(MEMES), contribution_unipolaire(MEMES)
    ecarts = [
        abs(_en(approche, (0, d, d)) / _en(exact, (0, d, d)) - 1) for d in (2, 20)
    ]
    assert ecarts[1] < ecarts[0] / 50
    assert ecarts[1] < 1e-3


def test_deux_charges_opposees_vues_de_loin() -> None:
    """Exemple 2.7.2 : p = 2 u_z, V ~ q c cos(theta) / (2 pi eps_0 r^2)."""
    rayon = sp.sqrt(X**2 + Y**2 + Z**2)
    assert contribution_unipolaire(OPPOSEES) == 0
    attendu = 2 * K * Z / rayon**3
    assert sp.simplify(contribution_dipolaire(OPPOSEES) - attendu) == 0
    assert contribution_quadrupolaire(OPPOSEES) == 0


def test_le_plan_mediateur_de_deux_charges_opposees_est_au_potentiel_nul() -> None:
    """Page 2-32 : « le plan xOy est aussi une equipotentielle zero »."""
    assert potentiel_des_charges(OPPOSEES).subs(Z, 0) == 0


def test_le_terme_quadrupolaire_d_un_quadrupole_lineaire() -> None:
    """(2.7.6c) : Q = 0 et p = 0, le premier terme non nul est V_2."""
    assert QUADRUPOLE.charge_totale() == 0
    assert QUADRUPOLE.moment_electrique() == (0, 0, 0)
    exact, approche = (
        potentiel_des_charges(QUADRUPOLE),
        contribution_quadrupolaire(QUADRUPOLE),
    )
    assert abs(_en(approche, (0, 30, 40)) / _en(exact, (0, 30, 40)) - 1) < 1e-2


def test_le_moment_quadrupolaire_est_un_tenseur_symetrique() -> None:
    """(2.7.18d) : p_uv = somme des q_i u_i v_i, donc p_uv = p_vu."""
    tenseur = TROIS.moment_quadrupolaire()
    assert tenseur == tenseur.T
    assert tenseur == sp.Matrix([[8, 0, 0], [0, -9, 0], [0, 0, 0]])
    assert QUADRUPOLE.moment_quadrupolaire() == sp.diag(0, 0, 2)


def test_le_quadrupole_ne_depend_pas_de_l_origine_si_q_et_p_sont_nuls() -> None:
    """Page 2-36 : « le moment quadrupolaire est independant du choix de l'origine »."""
    deplace = SystemeDeCharges(
        tuple((q, (x + 5, y - 2, z + 1)) for q, (x, y, z) in QUADRUPOLE.charges)
    )
    assert deplace.moment_quadrupolaire() == QUADRUPOLE.moment_quadrupolaire()


def test_les_deux_ecritures_du_terme_quadrupolaire_s_accordent() -> None:
    """(2.7.6c) et le troisieme terme de (2.7.17) : la meme fonction du point."""
    rayon = sp.sqrt(X**2 + Y**2 + Z**2)
    tenseur = TROIS.moment_quadrupolaire()
    par_derivees = (
        K
        / 2
        * sum(
            tenseur[i, j] * sp.diff(1 / rayon, CARTESIENNES[i], CARTESIENNES[j])
            for i in range(3)
            for j in range(3)
        )
    )
    assert sp.simplify(contribution_quadrupolaire(TROIS) - par_derivees) == 0


def test_le_potentiel_du_dipole() -> None:
    """(2.7.8) : V = p . u_r / (4 pi eps_0 r^2) = p cos(theta) / (4 pi eps_0 r^2)."""
    rayon = sp.sqrt(X**2 + Y**2 + Z**2)
    assert sp.simplify(potentiel_du_dipole((0, 0, 2)) - 2 * K * Z / rayon**3) == 0


def test_le_champ_du_dipole_derive_de_son_potentiel() -> None:
    """(2.7.9) : E = [3 (p . r) r / r^5 - p / r^3] / (4 pi eps_0)."""
    moment = (1, -2, 3)
    par_le_gradient = champ_du_potentiel(potentiel_du_dipole(moment))
    ecarts = [
        sp.simplify(a - b)
        for a, b in zip(champ_du_dipole(moment), par_le_gradient, strict=True)
    ]
    assert ecarts == [0, 0, 0]


def test_les_composantes_spheriques_du_champ_du_dipole() -> None:
    """Page 2-30 : E_r = 2 p cos(theta) / (4 pi eps_0 r^3), E_theta en sinus."""
    r, theta = sp.symbols("r theta", positive=True)
    point = {ABSCISSE: r * sp.sin(theta), ORDONNEE: 0, COTE: r * sp.cos(theta)}
    ex, _, ez = (c.subs(point) for c in champ_du_dipole((0, 0, 1)))
    radial = sp.simplify(ex * sp.sin(theta) + ez * sp.cos(theta))
    orthoradial = sp.simplify(ex * sp.cos(theta) - ez * sp.sin(theta))
    assert sp.simplify(radial - 2 * K * sp.cos(theta) / r**3) == 0
    assert sp.simplify(orthoradial - K * sp.sin(theta) / r**3) == 0


def test_les_lignes_de_champ_du_dipole() -> None:
    """(2.7.11) : r = K sin^2(theta), dans le plan y = 0."""
    theta, constante = sp.symbols("theta K_0", positive=True)
    rayon = constante * sp.sin(theta) ** 2
    ligne = (rayon * sp.sin(theta), sp.Integer(0), rayon * sp.cos(theta))
    assert est_une_ligne_de_champ(champ_du_dipole((0, 0, 1)), ligne, theta)


def test_les_equipotentielles_du_dipole() -> None:
    """(2.7.10) : r^2 = C cos(theta) -- le potentiel y est constant."""
    theta, constante = sp.symbols("theta C_0", positive=True)
    rayon = sp.sqrt(constante * sp.cos(theta))
    point = {ABSCISSE: rayon * sp.sin(theta), ORDONNEE: 0, COTE: rayon * sp.cos(theta)}
    sur_la_courbe = sp.simplify(potentiel_du_dipole((0, 0, 1)).subs(point))
    assert sp.simplify(sp.diff(sur_la_courbe, theta)) == 0
