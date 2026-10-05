"""Syllabus d'electricite generale, chapitre 3 : conducteurs et images electriques.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphes 3.1 a 3.8, pages 3-1 a 3-22, lues sur les pages rendues ; les
formules de capacite des pages 3-21 et 3-22 ont ete relues agrandies.

Les formules fermees de l'exemple 3.8.2 sont toutes rederivees ici a partir
du seul potentiel (3.8.9) de deux droites opposees : le champ par le
gradient, les cercles equipotentiels par substitution, la capacite par la
difference de potentiel entre les deux cylindres materialises.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMITTIVITE_DU_VIDE
from polytech.electromagnetisme.domain.electrostatique.avancee import laplacien
from polytech.electromagnetisme.domain.electrostatique.conducteurs import (
    champ_le_long_de_la_surface,
    champ_sur_la_surface,
    charge_lineique_pour_un_potentiel,
    charge_surfacique_maximale,
    charges_des_faces_de_l_enveloppe,
    force_surfacique,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import (
    PERMITTIVITE,
    champ_par_gauss,
    densite_surfacique,
)
from polytech.electromagnetisme.domain.electrostatique.cylindres_paralleles import (
    DeuxCylindres,
    capacite_lineique_de_deux_cylindres_egaux,
    capacite_lineique_des_petits_rayons,
    capacite_lineique_presque_coaxiale,
)
from polytech.electromagnetisme.domain.electrostatique.droites_paralleles import (
    cercle_de_champ,
    cercle_equipotentiel,
    parametre_de_l_equipotentielle,
    potentiel_de_deux_droites,
)
from polytech.electromagnetisme.domain.electrostatique.potentiel import (
    champ_du_potentiel,
    est_une_ligne_de_champ,
)
from polytech.electromagnetisme.domain.electrostatique.segment import (
    tension_entre_deux_rayons,
)
from polytech.electromagnetisme.domain.symboles import ABSCISSE, ORDONNEE

X, Y = ABSCISSE, ORDONNEE
LAMBDA, DEMI = sp.symbols("lambda_1 a", positive=True)
DEUX_PI_EPS = 2 * math.pi * PERMITTIVITE_DU_VIDE


def _nombre(expression: sp.Expr) -> float:
    return float(sp.sympify(expression).subs(PERMITTIVITE, PERMITTIVITE_DU_VIDE))


# --- 3.1 a 3.4 : le champ dans, le long de et sur un conducteur ----------------


def test_le_champ_le_long_de_la_surface_d_un_conducteur() -> None:
    """(3.3.3) : E_ext = sigma / eps_0, normal a la surface."""
    sigma = sp.Symbol("sigma", positive=True)
    assert champ_le_long_de_la_surface(sigma) == sigma / PERMITTIVITE


def test_le_champ_sur_la_surface_est_la_moitie_du_champ_exterieur() -> None:
    """(3.4.2) : E_sur = E_ext / 2 = sigma / (2 eps_0)."""
    sigma = sp.Symbol("sigma", positive=True)
    assert champ_sur_la_surface(sigma) == champ_le_long_de_la_surface(sigma) / 2


def test_la_force_par_unite_de_surface_d_un_conducteur_charge() -> None:
    """(3.4.1) : dF = sigma dS E_sur, soit sigma^2 / (2 eps_0) par unite de surface."""
    sigma = sp.Symbol("sigma", real=True)
    pression = force_surfacique(sigma)
    assert pression == sigma**2 / (2 * PERMITTIVITE)
    assert pression.subs(sigma, -3) == pression.subs(sigma, 3)


def test_la_charge_surfacique_que_l_air_supporte() -> None:
    """Remarque de la page 3-5 : 3 MV/m dans l'air, donc environ 26 microC/m^2."""
    assert math.isclose(_nombre(charge_surfacique_maximale(3e6)), 26e-6, rel_tol=0.03)


def test_le_conducteur_cylindrique_de_l_exemple_3_4_1() -> None:
    """(3.4.4) en rho = R : E_ext = sigma / eps_0, avec sigma = Q / (2 pi R l)."""
    charge, longueur, rayon = sp.symbols("Q l R", positive=True)
    sigma = densite_surfacique(charge, longueur, rayon)
    a_la_surface = champ_par_gauss(charge, longueur, rayon)
    assert sp.simplify(a_la_surface - champ_le_long_de_la_surface(sigma)) == 0


def test_le_champ_sur_le_conducteur_cylindrique() -> None:
    """(3.4.10) : E_sur(R) = Q / (4 pi eps_0 l R)."""
    charge, longueur, rayon = sp.symbols("Q l R", positive=True)
    sigma = densite_surfacique(charge, longueur, rayon)
    attendu = charge / (4 * sp.pi * PERMITTIVITE * longueur * rayon)
    assert sp.simplify(champ_sur_la_surface(sigma) - attendu) == 0


def test_la_charge_lineique_qu_impose_un_potentiel() -> None:
    """(3.8.2) : lambda = Q / l = 2 pi eps_0 V(R) / ln(rho_0 / R)."""
    potentiel, rayon, reference = sp.symbols("V R rho_0", positive=True)
    lineique = charge_lineique_pour_un_potentiel(potentiel, rayon, reference)
    assert lineique == 2 * sp.pi * PERMITTIVITE * potentiel / sp.log(reference / rayon)
    retrouve = tension_entre_deux_rayons(lineique, rayon, reference)
    assert sp.simplify(retrouve - potentiel) == 0


def test_le_potentiel_de_reference_ne_peut_pas_etre_la_surface() -> None:
    """Si rho_0 = R, le logarithme est nul : aucune charge ne donne ce potentiel."""
    with pytest.raises(ValueError, match="rayon de reference egal au rayon"):
        charge_lineique_pour_un_potentiel(5, 2, 2)


# --- 3.6 Systemes fermes, systemes ouverts -------------------------------------


def test_les_charges_des_deux_faces_de_l_enveloppe() -> None:
    """(3.6.1) et (3.6.2) : Q_0i = -(Q_1 + Q_2 + Q_3), Q_0e = Q_0 + Q_1 + Q_2 + Q_3."""
    interieure, exterieure = charges_des_faces_de_l_enveloppe(10, [1, 2, 3])
    assert (interieure, exterieure) == (-6, 16)
    assert interieure + exterieure == 10


def test_l_influence_totale_d_une_enveloppe_neutre() -> None:
    """Une enveloppe sans charge renvoie a l'exterieur la charge qu'elle entoure."""
    assert charges_des_faces_de_l_enveloppe(0, [4, -1]) == (-3, 3)
    assert charges_des_faces_de_l_enveloppe(7, []) == (0, 7)


# --- 3.8 Images electriques, exemple 3.8.2 --------------------------------------


def test_le_potentiel_de_deux_droites_opposees() -> None:
    """(3.8.9) : (lambda_1 / 4 pi eps_0) ln{[(x-a)^2 + y^2] / [(x+a)^2 + y^2]}."""
    potentiel = potentiel_de_deux_droites(LAMBDA, DEMI)
    haut, bas = (X - DEMI) ** 2 + Y**2, (X + DEMI) ** 2 + Y**2
    attendu = LAMBDA / (4 * sp.pi * PERMITTIVITE) * sp.log(haut / bas)
    assert sp.simplify(potentiel - attendu) == 0


def test_les_deux_droites_somment_deux_potentiels_logarithmiques() -> None:
    """(3.8.7) : (lambda_1 / 2 pi eps_0) ln(rho_2 / rho_1), la droite 1 en x = -a."""
    rho_1 = sp.sqrt((X + DEMI) ** 2 + Y**2)
    rho_2 = sp.sqrt((X - DEMI) ** 2 + Y**2)
    somme = tension_entre_deux_rayons(LAMBDA, rho_1, rho_2)
    ecart = sp.expand_log(potentiel_de_deux_droites(LAMBDA, DEMI) - somme, force=True)
    assert sp.simplify(ecart) == 0


def test_le_plan_mediateur_des_deux_droites_est_au_potentiel_zero() -> None:
    """Page 3-15 : « le plan x = 0 est l'equipotentielle zero »."""
    assert sp.simplify(potentiel_de_deux_droites(LAMBDA, DEMI).subs(X, 0)) == 0


def test_le_potentiel_des_deux_droites_est_harmonique() -> None:
    """(3.1.3) : hors des droites, le potentiel verifie l'equation de Laplace."""
    assert sp.simplify(laplacien(potentiel_de_deux_droites(LAMBDA, DEMI))) == 0


def test_le_champ_des_deux_droites_en_coordonnees_cartesiennes() -> None:
    """(3.8.11) : E_x et E_y, obtenus ici par le gradient de (3.8.9)."""
    e_x, e_y, e_z = champ_du_potentiel(potentiel_de_deux_droites(LAMBDA, DEMI))
    produit = ((X - DEMI) ** 2 + Y**2) * ((X + DEMI) ** 2 + Y**2)
    facteur = LAMBDA * DEMI / (sp.pi * PERMITTIVITE * produit)
    assert sp.simplify(e_x + facteur * (X**2 - Y**2 - DEMI**2)) == 0
    assert sp.simplify(e_y + 2 * facteur * X * Y) == 0
    assert e_z == 0


def test_le_champ_des_deux_droites_en_coordonnees_polaires() -> None:
    """(3.8.10) : E_rho en (rho^2 - a^2) cos(phi), E_phi en (rho^2 + a^2) sin(phi)."""
    rho, phi = sp.symbols("rho phi", positive=True)
    polaire = {X: rho * sp.cos(phi), Y: rho * sp.sin(phi)}
    e_x, e_y, _ = (
        c.subs(polaire)
        for c in champ_du_potentiel(potentiel_de_deux_droites(LAMBDA, DEMI))
    )
    radial = e_x * sp.cos(phi) + e_y * sp.sin(phi)
    orthoradial = -e_x * sp.sin(phi) + e_y * sp.cos(phi)
    moins = DEMI**2 + rho**2 - 2 * DEMI * rho * sp.cos(phi)
    plus = DEMI**2 + rho**2 + 2 * DEMI * rho * sp.cos(phi)
    facteur = LAMBDA * DEMI / (sp.pi * PERMITTIVITE * moins * plus)
    assert sp.simplify(radial + facteur * (rho**2 - DEMI**2) * sp.cos(phi)) == 0
    assert sp.simplify(orthoradial + facteur * (rho**2 + DEMI**2) * sp.sin(phi)) == 0


def test_le_parametre_d_une_equipotentielle() -> None:
    """(3.8.6) : p = exp(2 pi eps_0 V / lambda_1) ; 1 sur le plan mediateur."""
    tension = sp.Symbol("V", real=True)
    p = parametre_de_l_equipotentielle(tension, LAMBDA)
    assert p == sp.exp(2 * sp.pi * PERMITTIVITE * tension / LAMBDA)
    assert p.subs(tension, 0) == 1
    assert sp.simplify(p * parametre_de_l_equipotentielle(-tension, LAMBDA)) == 1


def test_les_equipotentielles_sont_des_cercles() -> None:
    """(3.8.13) : centre x_0 = -a (p^2 + 1) / (p^2 - 1), rayon |2 a p / (p^2 - 1)|."""
    assert cercle_equipotentiel(3, 4) == (-5, 3)
    assert cercle_equipotentiel(sp.Rational(1, 3), 4) == (5, 3)


def test_le_potentiel_est_constant_sur_un_cercle_equipotentiel() -> None:
    """Sur le cercle de parametre p, V = (lambda_1 / 2 pi eps_0) ln p -- (3.8.6)."""
    angle = sp.Symbol("t", real=True)
    centre, rayon = cercle_equipotentiel(3, 4)
    sur_le_cercle = {X: centre + rayon * sp.cos(angle), Y: rayon * sp.sin(angle)}
    potentiel = sp.simplify(potentiel_de_deux_droites(LAMBDA, 4).subs(sur_le_cercle))
    assert sp.simplify(potentiel - LAMBDA * sp.log(3) / (2 * sp.pi * PERMITTIVITE)) == 0


def test_le_centre_et_le_rayon_d_une_equipotentielle_sont_lies() -> None:
    """(3.8.16) : x_0^2 - R^2 = a^2, quel que soit p."""
    p = sp.Symbol("p", positive=True)
    centre, rayon = cercle_equipotentiel(p, DEMI)
    assert sp.simplify(centre**2 - rayon**2 - DEMI**2) == 0


def test_le_plan_mediateur_n_est_pas_un_cercle() -> None:
    """Pour p = 1, le « cercle » est le plan x = 0, de rayon infini."""
    with pytest.raises(ValueError, match="plan mediateur"):
        cercle_equipotentiel(1, 4)


def test_les_lignes_de_champ_sont_des_cercles_passant_par_les_deux_droites() -> None:
    """(3.8.15) : centre (0, K a / 2), rayon a sqrt(1 + K^2 / 4)."""
    centre, rayon = cercle_de_champ(sp.Rational(3, 2), 4)
    assert (centre, rayon) == ((0, 3), 5)
    for abscisse in (-4, 4):
        assert (abscisse - centre[0]) ** 2 + (0 - centre[1]) ** 2 == rayon**2


def test_un_cercle_de_champ_est_bien_une_ligne_de_champ() -> None:
    """(2.5.8) appliquee au champ (3.8.11) le long du cercle (3.8.15)."""
    angle = sp.Symbol("t", real=True)
    (x_c, y_c), rayon = cercle_de_champ(sp.Rational(3, 2), 4)
    cercle = (x_c + rayon * sp.cos(angle), y_c + rayon * sp.sin(angle), sp.Integer(0))
    champ = champ_du_potentiel(potentiel_de_deux_droites(1, 4))
    assert est_une_ligne_de_champ(champ, cercle, angle)


# --- Exemple 3.8.2, suite : deux conducteurs cylindriques paralleles -----------

EXTERIEURS = DeuxCylindres(premier_rayon=1.0, second_rayon=2.0, demi_entraxe=2.5)
EMBOITES = DeuxCylindres(premier_rayon=5.0, second_rayon=2.0, demi_entraxe=0.5)


def test_les_axes_de_deux_cylindres_exterieurs() -> None:
    """(3.8.17) : x_02 - x_01 = 2b et x_01^2 - R_1^2 = x_02^2 - R_2^2 = a^2."""
    x_1, x_2 = EXTERIEURS.axes()
    assert math.isclose(x_2 - x_1, 5.0)
    assert math.isclose(x_1**2 - 1.0, x_2**2 - 4.0)
    assert math.isclose(x_1**2 - 1.0, EXTERIEURS.demi_ecart_des_droites() ** 2)
    assert x_1 < 0 < x_2


def test_le_demi_ecart_des_droites_equivalentes() -> None:
    """(3.8.18) : a^2 = [(R_2+R_1)^2 / 4b - b] [(R_2-R_1)^2 / 4b - b]."""
    attendu = (9 / 10 - 2.5) * (1 / 10 - 2.5)
    assert math.isclose(EXTERIEURS.demi_ecart_des_droites() ** 2, attendu)


def test_deux_cylindres_qui_se_coupent_sont_refuses() -> None:
    """Ni exterieurs (2b > R_1 + R_2), ni emboites (2b < |R_1 - R_2|)."""
    with pytest.raises(ValueError, match="les deux cylindres se coupent"):
        DeuxCylindres(premier_rayon=1.0, second_rayon=2.0, demi_entraxe=1.0)


@pytest.mark.parametrize("demi_entraxe", [1.5, 0.5])
def test_deux_cylindres_tangents_sont_refuses(demi_entraxe: float) -> None:
    """Au contact, exterieur ou interieur, les deux droites se confondent (a = 0)."""
    with pytest.raises(ValueError, match="les deux cylindres se coupent"):
        DeuxCylindres(premier_rayon=1.0, second_rayon=2.0, demi_entraxe=demi_entraxe)


def test_les_parametres_des_deux_surfaces() -> None:
    """(3.8.19) : p^2 = (x_0 - a) / (x_0 + a) = (R / (x_0 + a))^2."""
    p_1, p_2 = EXTERIEURS.parametres()
    a = EXTERIEURS.demi_ecart_des_droites()
    for p, x_0, rayon in zip((p_1, p_2), EXTERIEURS.axes(), (1.0, 2.0), strict=True):
        assert math.isclose(p**2, (x_0 - a) / (x_0 + a))
        assert math.isclose(p**2, (rayon / (x_0 + a)) ** 2)
    assert p_1 > 1 > p_2


def test_la_capacite_lineique_par_les_deux_ecritures() -> None:
    """(3.8.20) et (3.8.21) : ln(p_1 / p_2) = ln(k + sqrt(k^2 - 1))."""
    p_1, p_2 = EXTERIEURS.parametres()
    k = (25.0 - 1.0 - 4.0) / (2 * 1.0 * 2.0)
    par_k = DEUX_PI_EPS / math.log(k + math.sqrt(k**2 - 1))
    assert math.isclose(
        EXTERIEURS.capacite_lineique(), DEUX_PI_EPS / math.log(p_1 / p_2)
    )
    assert math.isclose(EXTERIEURS.capacite_lineique(), par_k)


def test_la_capacite_lineique_recoupe_le_potentiel_des_deux_droites() -> None:
    """C / l = lambda / U_12, U_12 lue dans (3.8.9) sur les deux surfaces."""
    a = EXTERIEURS.demi_ecart_des_droites()
    x_1, x_2 = EXTERIEURS.axes()
    potentiel = potentiel_de_deux_droites(1.0, a).subs(
        PERMITTIVITE, PERMITTIVITE_DU_VIDE
    )
    sur_le_premier = float(potentiel.subs({X: x_1 + 1.0, Y: 0}))
    sur_le_second = float(potentiel.subs({X: x_2 - 2.0, Y: 0}))
    ailleurs_sur_le_second = float(potentiel.subs({X: x_2, Y: 2.0}))
    assert math.isclose(sur_le_second, ailleurs_sur_le_second)
    assert math.isclose(
        EXTERIEURS.capacite_lineique(), 1.0 / (sur_le_premier - sur_le_second)
    )


def test_la_capacite_de_deux_cylindres_emboites() -> None:
    """(3.8.21), signe inferieur : k = (R_1^2 + R_2^2 - 4b^2) / (2 R_1 R_2)."""
    k = (25.0 + 4.0 - 1.0) / (2 * 5.0 * 2.0)
    attendu = DEUX_PI_EPS / math.log(k + math.sqrt(k**2 - 1))
    assert math.isclose(EMBOITES.capacite_lineique(), attendu)
    x_1, x_2 = EMBOITES.axes()
    assert (x_1, x_2) == (11.0, 10.0)


def test_la_capacite_ne_depend_pas_de_l_ordre_des_deux_cylindres() -> None:
    """Emboites ou non, ln(p_1 / p_2) reste positif : (3.8.20) vaut telle quelle."""
    retourne = DeuxCylindres(premier_rayon=2.0, second_rayon=5.0, demi_entraxe=0.5)
    assert retourne.axes() == (-10.0, -11.0)
    p_1, p_2 = retourne.parametres()
    assert p_1 > p_2 > 1
    assert math.isclose(retourne.capacite_lineique(), EMBOITES.capacite_lineique())
    echange = DeuxCylindres(premier_rayon=2.0, second_rayon=1.0, demi_entraxe=2.5)
    assert math.isclose(echange.capacite_lineique(), EXTERIEURS.capacite_lineique())


def test_deux_cylindres_de_meme_rayon() -> None:
    """(3.8.23) : C / l = pi eps_0 / ln[b/R + sqrt((b/R)^2 - 1)]."""
    egaux = DeuxCylindres(premier_rayon=1.5, second_rayon=1.5, demi_entraxe=4.0)
    assert math.isclose(
        egaux.capacite_lineique(), capacite_lineique_de_deux_cylindres_egaux(1.5, 4.0)
    )
    rapport = 4.0 / 1.5
    attendu = (
        math.pi * PERMITTIVITE_DU_VIDE / math.log(rapport + math.sqrt(rapport**2 - 1))
    )
    assert math.isclose(capacite_lineique_de_deux_cylindres_egaux(1.5, 4.0), attendu)


def test_deux_cylindres_de_petits_rayons() -> None:
    """(3.8.24) : C / l ~ 2 pi eps_0 / ln(4 b^2 / R_1 R_2), a mieux que le millieme."""
    fins = DeuxCylindres(premier_rayon=0.01, second_rayon=0.02, demi_entraxe=2.5)
    approchee = capacite_lineique_des_petits_rayons(0.01, 0.02, 2.5)
    assert math.isclose(approchee, DEUX_PI_EPS / math.log(25.0 / 0.0002))
    assert math.isclose(fins.capacite_lineique(), approchee, rel_tol=1e-3)
    assert not math.isclose(
        EXTERIEURS.capacite_lineique(),
        capacite_lineique_des_petits_rayons(1.0, 2.0, 2.5),
        rel_tol=0.05,
    )


def test_deux_cylindres_presque_coaxiaux() -> None:
    """(3.8.25) et (3.8.26) : ln[(R_1/R_2) (2a - b)/(2a + b)], a par (3.8.26)."""
    proches = DeuxCylindres(premier_rayon=5.0, second_rayon=2.0, demi_entraxe=0.05)
    approchee = capacite_lineique_presque_coaxiale(5.0, 2.0, 0.05)
    assert math.isclose(proches.capacite_lineique(), approchee, rel_tol=1e-6)
    coaxiale = DEUX_PI_EPS / math.log(5.0 / 2.0)
    assert math.isclose(approchee, coaxiale, rel_tol=1e-3)
    assert approchee > coaxiale


def test_le_champ_maximal_entre_deux_cylindres() -> None:
    """(3.8.22) : (p_1+1)/(p_1-1) U_12 / [R_1 ln(p_1/p_2)], sur le petit cylindre."""
    tension = 1000.0
    p_1, p_2 = EXTERIEURS.parametres()
    attendu = (p_1 + 1) / (p_1 - 1) * tension / (1.0 * math.log(p_1 / p_2))
    assert math.isclose(EXTERIEURS.champ_maximal(tension), attendu)


def test_le_champ_maximal_recoupe_le_gradient_du_potentiel() -> None:
    """Le champ (3.8.11) au point du petit cylindre le plus proche de l'autre."""
    tension = 1000.0
    a = EXTERIEURS.demi_ecart_des_droites()
    lineique = EXTERIEURS.capacite_lineique() * tension
    potentiel = potentiel_de_deux_droites(lineique, a).subs(
        PERMITTIVITE, PERMITTIVITE_DU_VIDE
    )
    e_x = champ_du_potentiel(potentiel)[0]
    x_1, _ = EXTERIEURS.axes()
    au_plus_pres = abs(float(e_x.subs({X: x_1 + 1.0, Y: 0})))
    a_l_oppose = abs(float(e_x.subs({X: x_1 - 1.0, Y: 0})))
    assert math.isclose(EXTERIEURS.champ_maximal(tension), au_plus_pres)
    assert au_plus_pres > a_l_oppose


def test_le_champ_maximal_dans_deux_cylindres_emboites() -> None:
    """(3.8.22) vaut aussi pour le cylindre interieur, face a la paroi proche."""
    tension = 1000.0
    a = EMBOITES.demi_ecart_des_droites()
    lineique = EMBOITES.capacite_lineique() * tension
    potentiel = potentiel_de_deux_droites(lineique, a).subs(
        PERMITTIVITE, PERMITTIVITE_DU_VIDE
    )
    e_x = champ_du_potentiel(potentiel)[0]
    _, x_2 = EMBOITES.axes()
    au_plus_pres = abs(float(e_x.subs({X: x_2 - 2.0, Y: 0})))
    a_l_oppose = abs(float(e_x.subs({X: x_2 + 2.0, Y: 0})))
    assert math.isclose(EMBOITES.champ_maximal(tension), au_plus_pres)
    assert au_plus_pres > a_l_oppose
