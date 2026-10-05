"""Syllabus d'electricite generale, chapitre 8 : l'induction magnetique.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphes 8.1 a 8.9, pages 8-1 a 8-20, lues sur les pages rendues.

Chaque formule fermee des exemples 8.9.1 et 8.9.2 est recoupee par
l'integrale de Biot et Savart (8.9.1) menee numeriquement le long du
circuit, et par le rotationnel du potentiel vecteur.

Le segment de courant et le segment charge de l'exemple 2.6.2 ont la meme
geometrie : un test les relie, `B_phi / E_rho = mu_0 eps_0 I / lambda`.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import (
    PERMEABILITE_DU_VIDE,
    PERMITTIVITE_DU_VIDE,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.potentiel import circulation
from polytech.electromagnetisme.domain.electrostatique.segment import champ_du_segment
from polytech.electromagnetisme.domain.magnetostatique.actions import (
    force_electromagnetique,
    travail_a_courant_constant,
)
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
    induction_d_un_element,
    induction_d_une_charge,
    integrale_de_biot_et_savart,
)
from polytech.electromagnetisme.domain.magnetostatique.boucle import (
    induction_de_la_boucle,
    induction_sur_l_axe,
)
from polytech.electromagnetisme.domain.magnetostatique.lois import (
    est_un_champ_magnetique,
    flux_magnetique,
    force_de_lorentz,
    force_sur_un_element_de_courant,
    induction_du_fil_infini,
    potentiel_vecteur_du_fil_infini,
)
from polytech.electromagnetisme.domain.magnetostatique.potentiel_vecteur import (
    changement_de_jauge,
    est_en_jauge_de_coulomb,
    induction_du_potentiel,
)
from polytech.electromagnetisme.domain.magnetostatique.segment_de_courant import (
    induction_au_centre_d_une_spire_carree,
    induction_de_la_droite,
    induction_du_segment,
    potentiel_vecteur_de_la_droite,
    potentiel_vecteur_du_segment,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES, COTE, RAYON

X, Y, Z = CARTESIENNES
T = sp.Symbol("t", real=True)
COURANT, BAS, HAUT = 3.0, 0.3, 0.5
"""Un segment de 3 A, de z = -0,3 m a z = +0,5 m."""
POINTS = [(0.2, 0.1), (0.4, -0.6), (1.5, 0.9), (0.05, 0.45)]
AMPERE = PERMEABILITE_DU_VIDE / (4 * math.pi)


def _nombre(expression: sp.Expr, rho: float, z: float) -> float:
    valeurs = {RAYON: rho, COTE: z, PERMEABILITE: PERMEABILITE_DU_VIDE}
    return float(sp.sympify(expression).subs(valeurs))


# --- 8.3 Vecteur induction magnetique ------------------------------------------


def test_la_force_de_coulomb_lorentz() -> None:
    """(8.3.4) : F = q (E + v ^ B)."""
    force = force_electromagnetique(
        2.0, (1.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 0.5)
    )
    magnetique = force_de_lorentz(2.0, (0.0, 3.0, 0.0), (0.0, 0.0, 0.5))
    assert (
        force == (2.0 + magnetique[0], magnetique[1], magnetique[2]) == (5.0, 0.0, 0.0)
    )


def test_sans_vitesse_il_ne_reste_que_la_force_electrique() -> None:
    """(8.3.4) au repos : la relation (2.4.1), F = q E."""
    assert force_electromagnetique(
        2.0, (1.0, -2.0, 0.5), (0.0, 0.0, 0.0), (9.0, 9.0, 9.0)
    ) == (
        2.0,
        -4.0,
        1.0,
    )


def test_la_force_sur_un_troncon_droit_normal_au_champ() -> None:
    """(8.3.5) : F = B l I."""
    force = force_sur_un_element_de_courant(2.0, (0.0, 0.0, 0.5), (0.3, 0.0, 0.0))
    assert force == (0.0, 0.3, 0.0)
    assert math.isclose(math.hypot(*force), 0.3 * 0.5 * 2.0)


# --- 8.5 et 8.7 Travail des forces du champ -------------------------------------


def test_le_travail_est_le_courant_par_le_flux_coupe() -> None:
    """(8.5.1) : dT = I d(phi) -- F . dr pour un troncon qui balaie l'aire l dr."""
    induction, longueur, courant, pas = 0.3, 0.5, 2.0, 0.01
    force = force_sur_un_element_de_courant(
        courant, (0.0, 0.0, longueur), (induction, 0.0, 0.0)
    )
    flux_coupe = flux_magnetique(induction, longueur * pas)
    assert math.isclose(
        travail_a_courant_constant(courant, 0.0, flux_coupe), force[1] * pas
    )


def test_le_travail_a_courant_constant_suit_l_accroissement_du_flux() -> None:
    """8.7 : T = I (phi_2 - phi_1) ; negatif si le flux a travers le circuit diminue."""
    assert travail_a_courant_constant(2.0, 0.5, 0.75) == 0.5
    assert travail_a_courant_constant(2.0, 0.75, 0.5) == -0.5


# --- 8.6 et 8.8 Les deux proprietes fondamentales -------------------------------


def test_l_induction_derive_d_un_potentiel_vecteur() -> None:
    """(8.6.3) : B = rot A ; sa divergence est alors nulle (8.6.2)."""
    potentiel = (Y * Z**2, sp.sin(X) * Z, X * Y)
    induction = induction_du_potentiel(potentiel)
    attendu = (X - sp.sin(X), 2 * Y * Z - Y, sp.cos(X) * Z - Z**2)
    assert [sp.simplify(b - a) for b, a in zip(induction, attendu, strict=True)] == [
        0
    ] * 3
    assert est_un_champ_magnetique(induction)


def test_le_flux_est_la_circulation_du_potentiel_vecteur() -> None:
    """(8.6.4) : phi = circulation de A -- B pi a^2 pour un champ uniforme."""
    induction, rayon = sp.symbols("B a", positive=True)
    potentiel = (-induction * Y / 2, induction * X / 2, sp.Integer(0))
    assert induction_du_potentiel(potentiel) == (0, 0, induction)
    cercle = (rayon * sp.cos(T), rayon * sp.sin(T), sp.Integer(0))
    assert (
        circulation(potentiel, cercle, T, 0, 2 * sp.pi) == induction * sp.pi * rayon**2
    )


def test_un_changement_de_jauge_ne_change_pas_l_induction() -> None:
    """Complement de la page 8-12 : A' = A + grad(phi) donne le meme B."""
    potentiel = (-Y, X, sp.Integer(0))
    jauge = X**2 * Y + sp.cos(Z)
    autre = changement_de_jauge(potentiel, jauge)
    assert autre == (2 * X * Y - Y, X**2 + X, -sp.sin(Z))
    assert induction_du_potentiel(autre) == induction_du_potentiel(potentiel)


def test_la_jauge_de_coulomb() -> None:
    """(8.8.2) : div A = 0 ; une fonction harmonique la preserve, une autre non."""
    potentiel = (-Y, X, sp.Integer(0))
    assert est_en_jauge_de_coulomb(potentiel)
    assert est_en_jauge_de_coulomb(changement_de_jauge(potentiel, X**2 - Y**2))
    assert not est_en_jauge_de_coulomb(changement_de_jauge(potentiel, X**2 + Y**2))


# --- 8.9 Formule de Biot et Savart ---------------------------------------------


def test_l_induction_d_un_element_de_courant() -> None:
    """(8.9.1) : dB = (mu_0 / 4 pi) dI dl ^ R / R^3 -- la regle du tire-bouchon."""
    courant = sp.Symbol("I", positive=True)
    element = induction_d_un_element(courant, (0, 0, 1), (2, 0, 0))
    assert element == (0, PERMEABILITE * courant / (16 * sp.pi), 0)


def test_l_induction_d_un_element_decroit_comme_le_carre_de_la_distance() -> None:
    """(8.9.1) : R / R^3, soit 1 / R^2 dans la direction normale a l'element."""
    pres = induction_d_un_element(1, (0, 0, 1), (1, 0, 0))[1]
    loin = induction_d_un_element(1, (0, 0, 1), (3, 0, 0))[1]
    assert sp.simplify(pres / loin) == 9
    assert induction_d_un_element(1, (0, 0, 1), (0, 0, 5)) == (0, 0, 0)


def test_l_induction_d_une_charge_en_mouvement() -> None:
    """(8.8.6), un seul terme : (mu_0 / 4 pi) q v ^ R / R^3."""
    charge, vitesse = sp.symbols("q v", positive=True)
    induction = induction_d_une_charge(charge, (0, 0, vitesse), (1, 0, 0))
    assert induction == (0, PERMEABILITE * charge * vitesse / (4 * sp.pi), 0)


def test_un_point_sur_l_element_n_a_pas_d_induction_definie() -> None:
    """A distance nulle, R / R^3 n'est pas defini."""
    with pytest.raises(ValueError, match="point confondu avec la source"):
        induction_d_un_element(1, (0, 0, 1), (0, 0, 0))


# --- Exemple 8.9.1 : segment de droite parcouru par un courant ------------------


@pytest.mark.parametrize(("rho", "z"), POINTS)
def test_l_induction_du_segment_recoupe_biot_et_savart(rho: float, z: float) -> None:
    """(8.9.4), contre l'integrale (8.9.3) menee numeriquement."""
    droite = (sp.Integer(0), sp.Integer(0), T)
    _, selon_y, _ = integrale_de_biot_et_savart(droite, T, -BAS, HAUT, (rho, 0.0, z))
    assert math.isclose(
        _nombre(induction_du_segment(COURANT, BAS, HAUT), rho, z),
        AMPERE * COURANT * selon_y,
        rel_tol=1e-9,
    )


@pytest.mark.parametrize(("rho", "z"), POINTS)
def test_l_induction_du_segment_par_les_angles(rho: float, z: float) -> None:
    """(8.9.5) : (mu_0 I / 4 pi rho) (sin t1 - sin t2), avec t1 > 0 > t2."""
    theta_1, theta_2 = math.atan((z + BAS) / rho), math.atan((z - HAUT) / rho)
    attendu = AMPERE * COURANT / rho * (math.sin(theta_1) - math.sin(theta_2))
    assert math.isclose(
        _nombre(induction_du_segment(COURANT, BAS, HAUT), rho, z), attendu
    )


@pytest.mark.parametrize(("rho", "z"), POINTS)
def test_l_induction_du_segment_derive_de_son_potentiel_vecteur(
    rho: float, z: float
) -> None:
    """(8.9.6) : B_phi = -dA_z/d rho, le rotationnel en coordonnees cylindriques."""
    potentiel = potentiel_vecteur_du_segment(COURANT, BAS, HAUT)
    assert math.isclose(
        _nombre(induction_du_segment(COURANT, BAS, HAUT), rho, z),
        _nombre(-sp.diff(potentiel, RAYON), rho, z),
    )


@pytest.mark.parametrize(("rho", "z"), POINTS)
def test_le_segment_de_courant_et_le_segment_charge_ont_la_meme_geometrie(
    rho: float, z: float
) -> None:
    """(8.9.4) et (2.6.10) : B_phi / E_rho = mu_0 eps_0 I / lambda, partout."""
    lineique = 2e-9
    e_rho, _ = champ_du_segment(lineique, BAS, HAUT)
    electrique = float(
        e_rho.subs({RAYON: rho, COTE: z, PERMITTIVITE: PERMITTIVITE_DU_VIDE})
    )
    magnetique = _nombre(induction_du_segment(COURANT, BAS, HAUT), rho, z)
    rapport = PERMEABILITE_DU_VIDE * PERMITTIVITE_DU_VIDE * COURANT / lineique
    assert math.isclose(magnetique / electrique, rapport)


def test_la_droite_infinie_est_la_limite_du_segment() -> None:
    """(8.9.7) : B_phi = mu_0 I / (2 pi rho)."""
    courant, demi = sp.symbols("I l", positive=True)
    limite = sp.limit(
        induction_du_segment(courant, demi, demi).subs(COTE, 0), demi, sp.oo
    )
    assert sp.simplify(limite - induction_de_la_droite(courant)) == 0
    assert induction_de_la_droite(courant) == PERMEABILITE * courant / (
        2 * sp.pi * RAYON
    )


def test_la_droite_infinie_recoupe_le_fil_du_fascicule() -> None:
    """(8.9.7) et (8.9.8), que `magnetostatique.lois` portait deja en nombres."""
    assert math.isclose(
        _nombre(induction_de_la_droite(COURANT), 0.25, 0.0),
        induction_du_fil_infini(COURANT, 0.25),
    )
    assert math.isclose(
        _nombre(potentiel_vecteur_de_la_droite(COURANT, 1.0), 0.25, 0.0),
        potentiel_vecteur_du_fil_infini(COURANT, 0.25),
    )


def test_le_rotationnel_du_potentiel_de_la_droite_redonne_l_induction() -> None:
    """Page 8-17 : « le rotationnel de (8.9.8) redonne bien (8.9.7) »."""
    courant, reference = sp.symbols("I rho_0", positive=True)
    potentiel = potentiel_vecteur_de_la_droite(courant, reference)
    assert (
        sp.simplify(-sp.diff(potentiel, RAYON) - induction_de_la_droite(courant)) == 0
    )


def test_l_induction_au_centre_d_une_spire_carree() -> None:
    """Remarque de la page 8-17 : B = 2 sqrt(2) mu_0 I / (pi l)."""
    courant, cote = sp.symbols("I l", positive=True)
    au_centre = induction_au_centre_d_une_spire_carree(courant, cote)
    assert au_centre == 2 * sp.sqrt(2) * PERMEABILITE * courant / (sp.pi * cote)
    un_cote = induction_du_segment(courant, cote / 2, cote / 2).subs(
        {RAYON: cote / 2, COTE: 0}
    )
    assert sp.simplify(au_centre - 4 * un_cote) == 0


def test_la_spire_carree_recoupe_biot_et_savart() -> None:
    """Les quatre cotes, parcourus dans le sens direct, integres numeriquement."""
    sommets = [(0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]
    total = 0.0
    for (x_0, y_0), (x_1, y_1) in pairwise(sommets):
        cote = (x_0 + (x_1 - x_0) * T, y_0 + (y_1 - y_0) * T, sp.Integer(0))
        total += integrale_de_biot_et_savart(cote, T, 0, 1, (0.0, 0.0, 0.0))[2]
    attendu = float(
        induction_au_centre_d_une_spire_carree(1, 1).subs(PERMEABILITE, 4 * sp.pi)
    )
    assert math.isclose(total, attendu, rel_tol=1e-9)


# --- Exemple 8.9.2 : boucle de rayon a ------------------------------------------


def test_l_induction_sur_l_axe_d_une_boucle() -> None:
    """(8.9.14) : B_z = mu_0 I a^2 / [2 (z^2 + a^2)^(3/2)], maximale en z = 0."""
    courant, rayon, cote = sp.symbols("I a z", positive=True)
    sur_l_axe = induction_sur_l_axe(courant, rayon, cote)
    attendu = (
        PERMEABILITE
        * courant
        * rayon**2
        / (2 * (cote**2 + rayon**2) ** sp.Rational(3, 2))
    )
    assert sp.simplify(sur_l_axe - attendu) == 0
    assert (
        sp.simplify(sur_l_axe.subs(cote, 0) - PERMEABILITE * courant / (2 * rayon)) == 0
    )
    assert sp.limit(sur_l_axe, cote, sp.oo) == 0


@pytest.mark.parametrize("cote", [0.0, 0.4, 2.0])
def test_l_induction_sur_l_axe_recoupe_biot_et_savart(cote: float) -> None:
    """La boucle parcourue dans le sens anti-horlogique, integree numeriquement."""
    cercle = (0.7 * sp.cos(T), 0.7 * sp.sin(T), sp.Integer(0))
    b_x, b_y, b_z = integrale_de_biot_et_savart(
        cercle, T, 0, 2 * math.pi, (0.0, 0.0, cote)
    )
    attendu = float(induction_sur_l_axe(1, 0.7, cote).subs(PERMEABILITE, 4 * sp.pi))
    assert math.isclose(b_z, attendu, rel_tol=1e-9)
    assert math.isclose(b_x, 0.0, abs_tol=1e-12)
    assert math.isclose(b_y, 0.0, abs_tol=1e-12)


@pytest.mark.parametrize(("rho", "cote"), [(0.3, 0.4), (1.2, -0.5), (0.69, 0.02)])
def test_l_induction_hors_de_l_axe_recoupe_biot_et_savart(
    rho: float, cote: float
) -> None:
    """(8.9.9) et (8.9.11), « a integrer numeriquement », contre l'integrale directe."""
    b_rho, b_z = induction_de_la_boucle(COURANT, 0.7, rho, cote)
    cercle = (0.7 * sp.cos(T), 0.7 * sp.sin(T), sp.Integer(0))
    direct = integrale_de_biot_et_savart(cercle, T, 0, 2 * math.pi, (rho, 0.0, cote))
    assert math.isclose(b_rho, AMPERE * COURANT * direct[0], rel_tol=1e-8)
    assert math.isclose(b_z, AMPERE * COURANT * direct[2], rel_tol=1e-8)


def test_sur_l_axe_les_integrales_de_la_boucle_donnent_la_formule_fermee() -> None:
    """Page 8-19 : rho = 0, donc K = 0, et seul B_z est different de zero."""
    b_rho, b_z = induction_de_la_boucle(COURANT, 0.7, 0.0, 0.4)
    attendu = float(
        induction_sur_l_axe(COURANT, 0.7, 0.4).subs(PERMEABILITE, PERMEABILITE_DU_VIDE)
    )
    assert math.isclose(b_rho, 0.0, abs_tol=1e-15)
    assert math.isclose(b_z, attendu)


def test_dans_le_plan_de_la_boucle_l_induction_change_de_sens_au_dela_du_fil() -> None:
    """Figure 8.9.4 : les lignes d'induction entourent le conducteur."""
    _, dedans = induction_de_la_boucle(COURANT, 0.7, 0.3, 0.0)
    _, dehors = induction_de_la_boucle(COURANT, 0.7, 1.5, 0.0)
    assert dedans > 0 > dehors


def test_un_point_sur_le_fil_de_la_boucle_est_refuse() -> None:
    """En rho = a et z = 0, K vaut 1 et les integrales divergent."""
    with pytest.raises(ValueError, match="point sur le conducteur"):
        induction_de_la_boucle(COURANT, 0.7, 0.7, 0.0)
