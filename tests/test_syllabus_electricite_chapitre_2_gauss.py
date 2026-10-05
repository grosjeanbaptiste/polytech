"""Syllabus d'electricite generale, chapitre 2 : polarisation, Gauss, Poisson.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphes 2.8 a 2.18, pages 2-36 a 2-64, lues sur les pages rendues.

La jonction pn de l'exemple 2.15.1 releve de la physique des
semi-conducteurs ; le syllabus la donne comme application de l'equation de
Poisson, et c'est a ce titre qu'elle est portee ici.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.avancee import laplacien
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.double_couche import (
    angle_solide_d_un_disque,
    potentiel_de_la_double_couche,
    saut_de_potentiel,
)
from polytech.electromagnetisme.domain.electrostatique.gauss import (
    charge_dans_une_boule_uniforme,
)
from polytech.electromagnetisme.domain.electrostatique.jonction_pn import (
    JonctionAbrupte,
)
from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
)
from polytech.electromagnetisme.domain.electrostatique.moments import SystemeDeCharges
from polytech.electromagnetisme.domain.electrostatique.passage import (
    residu_tangentiel,
    saut_normal,
)
from polytech.electromagnetisme.domain.electrostatique.plan_charge import (
    champ_du_plan,
    potentiel_du_plan,
)
from polytech.electromagnetisme.domain.electrostatique.polarisation import (
    charge_surfacique_fictive,
    charge_volumique_fictive,
    deplacement,
    electrisation,
    polarisation,
)
from polytech.electromagnetisme.domain.electrostatique.potentiel import (
    COULOMB,
    champ_du_potentiel,
    potentiel_des_charges,
)
from polytech.electromagnetisme.domain.electrostatique.sphere_uniforme import (
    champ_interieur,
    deplacement_interieur,
    deplacement_radial,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES

X, Y, Z = CARTESIENNES
HAUT, BAS = (0, 0, 1), (0, 0, -1)


# --- 2.8 Expression macroscopique du potentiel ---------------------------------


def test_la_polarisation_est_la_somme_des_moments_par_unite_de_volume() -> None:
    """(2.8.9) : P = somme des p_i / Delta V."""
    assert polarisation([(0, 0, 2), (0, 0, 4), (3, 0, 0)], 2) == (
        sp.Rational(3, 2),
        0,
        3,
    )


def test_des_dipoles_orientes_au_hasard_ne_polarisent_pas() -> None:
    """Page 2-39 : dans l'eau, « la polarisation y est nulle, bien que p_i != 0 »."""
    moments = [(1, 0, 0), (-1, 0, 0), (0, 2, 0), (0, -2, 0)]
    assert polarisation(moments, 5) == (0, 0, 0)


def test_un_volume_nul_n_a_pas_de_polarisation() -> None:
    """Le quotient (2.8.4) se prend sur un element de volume, donc non nul."""
    with pytest.raises(ValueError, match="volume strictement positif attendu"):
        polarisation([(0, 0, 1)], 0)


def test_le_developpement_au_premier_ordre_de_l_inverse_de_la_distance() -> None:
    """(2.8.2) : 1/R_i = 1/R_N + (r_i - r_N) . grad_N(1/R_N), a l'ordre 2 pres."""
    ecart = sp.Symbol("h", positive=True)
    distance = sp.Symbol("R", positive=True)
    exact = 1 / (distance - ecart)
    approche = 1 / distance + ecart * sp.diff(1 / (distance - ecart), ecart).subs(
        ecart, 0
    )
    assert sp.series(exact - approche, ecart, 0, 2).removeO() == 0


# --- 2.9 Charges fictives ------------------------------------------------------


def test_une_polarisation_uniforme_n_a_pas_de_charge_volumique_fictive() -> None:
    """Page 2-42 : pour qu'apparaisse eta', il faudrait que P ne soit pas uniforme."""
    assert charge_volumique_fictive((0, 0, 7)) == 0


def test_la_charge_volumique_fictive_est_l_oppose_de_la_divergence() -> None:
    """(2.9.1a) : eta' = -div P."""
    pente = sp.Symbol("k", positive=True)
    assert charge_volumique_fictive((0, 0, pente * Z)) == -pente
    assert charge_volumique_fictive((X**2, Y, 0)) == -2 * X - 1


def test_la_charge_surfacique_fictive_suit_la_normale_interieure() -> None:
    """(2.9.1b) : sigma' = -u_n . P, u_n pointant vers l'interieur du milieu."""
    assert charge_surfacique_fictive((0, 0, 5), normale_interieure=BAS) == 5
    assert charge_surfacique_fictive((0, 0, 5), normale_interieure=HAUT) == -5
    assert charge_surfacique_fictive((0, 0, 5), normale_interieure=(1, 0, 0)) == 0


def test_la_charge_fictive_totale_d_un_barreau_uniforme_est_nulle() -> None:
    """Figure 2.9.1 et (2.9.4) : +P sur une face, -P sur l'autre, rien dedans."""
    polarise = (0, 0, sp.Symbol("P", positive=True))
    faces = charge_surfacique_fictive(
        polarise, normale_interieure=BAS
    ) + charge_surfacique_fictive(polarise, normale_interieure=HAUT)
    assert faces == 0
    assert charge_volumique_fictive(polarise) == 0


def test_la_charge_fictive_totale_d_une_tranche_non_uniforme_est_nulle() -> None:
    """(2.9.4) : P = k z u_z entre z = 0 et z = h -- le volume compense la surface."""
    pente, hauteur = sp.symbols("k h", positive=True)
    polarise = (0, 0, pente * Z)
    en_volume = sp.integrate(charge_volumique_fictive(polarise), (Z, 0, hauteur))
    dessus = charge_surfacique_fictive(polarise, normale_interieure=BAS).subs(
        Z, hauteur
    )
    dessous = charge_surfacique_fictive(polarise, normale_interieure=HAUT).subs(Z, 0)
    assert (dessus, dessous) == (pente * hauteur, 0)
    assert en_volume + dessus + dessous == 0


def test_une_normale_non_unitaire_est_refusee() -> None:
    """La relation (2.9.1b) projette sur un vecteur unitaire."""
    with pytest.raises(ValueError, match="normale unitaire attendue"):
        charge_surfacique_fictive((0, 0, 5), normale_interieure=(0, 0, 2))


# --- 2.13 Theoreme de Gauss, deplacement electrique ----------------------------


def test_le_deplacement_electrique() -> None:
    """(2.13.3) : D = eps_0 E + P."""
    assert deplacement((1, 2, 3), (4, 0, -1)) == (
        PERMITTIVITE + 4,
        2 * PERMITTIVITE,
        3 * PERMITTIVITE - 1,
    )


def test_l_electrisation() -> None:
    """Remarque 3 de la page 2-50 : E_i = P / eps_0, et E = D / eps_0 - E_i."""
    champ, polarise = (1, 2, 3), (4, 0, -1)
    retrouve = tuple(
        d / PERMITTIVITE - e_i
        for d, e_i in zip(
            deplacement(champ, polarise), electrisation(polarise), strict=True
        )
    )
    assert tuple(sp.simplify(c) for c in retrouve) == champ


def test_la_divergence_du_deplacement_ne_voit_que_les_charges_reelles() -> None:
    """(2.13.5) et (2.15.1) : div D = eta, alors que eps_0 div E = eta + eta'."""
    pente = sp.Symbol("k", positive=True)
    champ, polarise = (X, 0, 0), (0, 0, pente * Z)
    reelle = divergence_cartesienne(deplacement(champ, polarise))
    apparente = PERMITTIVITE * divergence_cartesienne(champ)
    assert apparente == reelle + charge_volumique_fictive(polarise)


def test_le_deplacement_dans_la_sphere_isolante_de_l_exemple_2_13_1() -> None:
    """(2.13.10) : D_r = eta r / 3 = Q r / (4 pi a^3) ; ici D = eps_0 E."""
    charge, rayon, distance = sp.symbols("Q a r", positive=True)
    d_r = deplacement_interieur(charge, rayon, distance)
    volumique = charge / (sp.Rational(4, 3) * sp.pi * rayon**3)
    assert sp.simplify(d_r - volumique * distance / 3) == 0
    assert (
        sp.simplify(d_r - PERMITTIVITE * champ_interieur(charge, rayon, distance)) == 0
    )


def test_le_flux_du_deplacement_vaut_la_charge_enlacee() -> None:
    """(2.13.6) : 4 pi r^2 D_r = Q_in, recoupe par la charge d'une boule uniforme."""
    charge, rayon, distance = sp.symbols("Q a r", positive=True)
    flux = 4 * sp.pi * distance**2 * deplacement_interieur(charge, rayon, distance)
    assert (
        sp.simplify(flux - charge_dans_une_boule_uniforme(charge, rayon, distance)) == 0
    )


def test_le_deplacement_radial_d_une_charge_a_symetrie_spherique() -> None:
    """(2.13.9) : D_r(r) = (1 / r^2) integrale de 0 a r de eta(s) s^2 ds."""
    source, distance, volumique = sp.symbols("s r eta", positive=True)
    assert deplacement_radial(volumique, source, distance) == volumique * distance / 3
    assert deplacement_radial(source, source, distance) == distance**2 / 4


def test_hors_de_la_sphere_tout_se_passe_comme_pour_une_charge_ponctuelle() -> None:
    """(2.13.8) : D_r = Q / (4 pi r^2), continu en r = a avec (2.13.10)."""
    charge, rayon = sp.symbols("Q a", positive=True)
    a_la_surface = deplacement_interieur(charge, rayon, rayon)
    assert sp.simplify(a_la_surface - charge / (4 * sp.pi * rayon**2)) == 0


# --- 2.10 et exemple 2.13.2 : le plan uniformement charge ----------------------


def test_le_champ_du_plan_charge() -> None:
    """(2.10.5) et (2.13.13) : E = sgn(z) sigma / (2 eps_0) u_z."""
    sigma = sp.Symbol("sigma", positive=True)
    champ = champ_du_plan(sigma)
    assert champ.subs(Z, 3) == sigma / (2 * PERMITTIVITE)
    assert champ.subs(Z, -0.1) == -sigma / (2 * PERMITTIVITE)


def test_le_champ_du_plan_par_l_integrale_directe() -> None:
    """(2.10.4) : l'integrale double du noyau de Coulomb sur tout le plan."""
    source_x, source_y = sp.symbols("x_s y_s", real=True)
    hauteur, sigma = sp.symbols("h sigma", positive=True)
    noyau = hauteur / (source_x**2 + source_y**2 + hauteur**2) ** sp.Rational(3, 2)
    sur_y = sp.integrate(noyau, (source_y, -sp.oo, sp.oo))
    assert sp.simplify(sur_y - 2 * hauteur / (source_x**2 + hauteur**2)) == 0
    total = COULOMB * sigma * sp.integrate(sur_y, (source_x, -sp.oo, sp.oo))
    assert sp.simplify(total - champ_du_plan(sigma).subs(Z, hauteur)) == 0


def test_le_potentiel_du_plan_charge() -> None:
    """(2.10.8) et (2.13.14) : V = -sigma |z| / (2 eps_0), nul sur le plan."""
    sigma, hauteur = sp.symbols("sigma h", positive=True)
    potentiel = potentiel_du_plan(sigma)
    assert potentiel.subs(Z, 0) == 0
    assert potentiel.subs(Z, hauteur) == potentiel.subs(Z, -hauteur)
    au_dessus = -sp.diff(potentiel.subs(Z, hauteur), hauteur)
    assert sp.simplify(au_dessus - champ_du_plan(sigma).subs(Z, hauteur)) == 0


def test_le_potentiel_du_plan_ne_s_annule_pas_a_l_infini() -> None:
    """Page 2-55 : « il y a des charges a l'infini »."""
    sigma = sp.Symbol("sigma", positive=True)
    assert sp.limit(potentiel_du_plan(sigma), Z, sp.oo) == -sp.oo


# --- 2.15 Equation de Poisson, exemple 2.15.1 ----------------------------------

JONCTION = JonctionAbrupte(
    accepteurs=sp.Symbol("N_A", positive=True),
    donneurs=sp.Symbol("N_D", positive=True),
    largeur_p=sp.Symbol("x_p", positive=True),
)
E, EPS = JONCTION.CHARGE, JONCTION.PERMITTIVITE


def test_la_neutralite_globale_fixe_la_largeur_du_cote_n() -> None:
    """« N_A x_p = N_D x_n »."""
    assert (
        JONCTION.largeur_n * JONCTION.donneurs
        == JONCTION.accepteurs * JONCTION.largeur_p
    )
    total = sp.integrate(
        JONCTION.charge_volumique(), (X, -JONCTION.largeur_p, JONCTION.largeur_n)
    )
    assert sp.simplify(total) == 0


def test_la_charge_volumique_de_la_zone_de_depletion() -> None:
    """Figure 2.15.1 : -e N_A du cote p, +e N_D du cote n, rien au-dela."""
    eta = JONCTION.charge_volumique()
    cote_p, cote_n = -JONCTION.largeur_p / 2, JONCTION.largeur_n / 2
    assert eta.subs(X, cote_p) == -E * JONCTION.accepteurs
    assert eta.subs(X, cote_n) == E * JONCTION.donneurs
    assert eta.subs(X, -2 * JONCTION.largeur_p) == 0
    assert eta.subs(X, 2 * JONCTION.largeur_n) == 0


def test_le_champ_dans_la_jonction() -> None:
    """Page 2-57 : -e N_A (x + x_p) / eps du cote p, e N_D (x - x_n) / eps du cote n."""
    cote_p, cote_n = JONCTION.champ_p(), JONCTION.champ_n()
    assert cote_p == -E * JONCTION.accepteurs * (X + JONCTION.largeur_p) / EPS
    assert (
        sp.simplify(cote_n - E * JONCTION.donneurs * (X - JONCTION.largeur_n) / EPS)
        == 0
    )
    assert cote_p.subs(X, -JONCTION.largeur_p) == 0
    assert sp.simplify(cote_n.subs(X, JONCTION.largeur_n)) == 0
    assert sp.simplify(cote_p.subs(X, 0) - cote_n.subs(X, 0)) == 0


def test_le_champ_de_la_jonction_verifie_le_theoreme_de_gauss() -> None:
    """(2.15.3) : dE/dx = eta / eps de chaque cote."""
    assert sp.diff(JONCTION.champ_p(), X) == -E * JONCTION.accepteurs / EPS
    assert (
        sp.simplify(sp.diff(JONCTION.champ_n(), X) - E * JONCTION.donneurs / EPS) == 0
    )


def test_le_potentiel_dans_la_jonction() -> None:
    """Page 2-58 : deux arcs de parabole, raccordes en x = 0."""
    cote_p, cote_n = JONCTION.potentiel_p(), JONCTION.potentiel_n()
    assert (
        sp.simplify(
            cote_p - E * JONCTION.accepteurs * (X + JONCTION.largeur_p) ** 2 / (2 * EPS)
        )
        == 0
    )
    assert sp.simplify(cote_p.subs(X, 0) - cote_n.subs(X, 0)) == 0
    assert sp.simplify(sp.diff(cote_p, X) + JONCTION.champ_p()) == 0
    assert sp.simplify(sp.diff(cote_n, X) + JONCTION.champ_n()) == 0


def test_la_forme_imprimee_du_potentiel_du_cote_n() -> None:
    """Page 2-58 : (e / 2 eps) [N_D (2 x x_n - x^2) + N_A x_p^2]."""
    x_n = JONCTION.largeur_n
    imprime = (
        E
        / (2 * EPS)
        * (
            JONCTION.donneurs * (2 * X * x_n - X**2)
            + JONCTION.accepteurs * JONCTION.largeur_p**2
        )
    )
    assert sp.simplify(JONCTION.potentiel_n() - imprime) == 0


def test_la_hauteur_de_la_barriere() -> None:
    """Page 2-58 : U_b = V(x_n) - V(-x_p) = (e / 2 eps) (N_A x_p^2 + N_D x_n^2)."""
    x_n = JONCTION.largeur_n
    attendu = (
        E
        / (2 * EPS)
        * (JONCTION.accepteurs * JONCTION.largeur_p**2 + JONCTION.donneurs * x_n**2)
    )
    assert sp.simplify(JONCTION.barriere() - attendu) == 0
    assert JONCTION.potentiel_p().subs(X, -JONCTION.largeur_p) == 0


def test_le_potentiel_n_a_pas_d_extremum_hors_des_charges() -> None:
    """Remarque de la page 2-58 : a laplacien nul, les courbures se compensent."""
    seule = SystemeDeCharges(((1, (0, 0, 0)),))
    potentiel = potentiel_des_charges(seule)
    assert sp.simplify(laplacien(potentiel)) == 0
    courbures = [
        sp.diff(potentiel, axe, 2).subs({X: 1, Y: 2, Z: 3}) for axe in CARTESIENNES
    ]
    assert any(c > 0 for c in courbures)
    assert any(c < 0 for c in courbures)


# --- 2.16 Continuite de la fonction potentielle --------------------------------


def test_le_saut_de_potentiel_a_la_traversee_d_une_double_couche() -> None:
    """(2.16.2) et (2.16.5) : V_1 - V_2 = P_S / eps_0."""
    moment = sp.Symbol("P_S", positive=True)
    assert saut_de_potentiel(moment) == moment / PERMITTIVITE


def test_le_potentiel_d_une_double_couche_uniforme() -> None:
    """(2.16.4) : V = P_S Omega / (4 pi eps_0)."""
    moment, angle = sp.symbols("P_S Omega", positive=True)
    assert potentiel_de_la_double_couche(moment, angle) == COULOMB * moment * angle


def test_la_double_couche_fermee_de_la_page_2_61() -> None:
    """(2.16.6) : Omega = 0, 4 pi, 2 pi -- V nul dehors, P_S/eps_0 dedans."""
    moment = sp.Symbol("P_S", positive=True)
    dehors, dedans, dessus = (
        potentiel_de_la_double_couche(moment, angle)
        for angle in (0, 4 * sp.pi, 2 * sp.pi)
    )
    assert dehors == 0
    assert dedans == saut_de_potentiel(moment)
    assert dessus == saut_de_potentiel(moment) / 2


def test_l_angle_solide_d_un_disque_vu_de_son_axe() -> None:
    """Un demi-espace (2 pi) au contact, rien a l'infini, et le signe de la face vue."""
    rayon, distance = sp.symbols("a d", positive=True)
    angle = angle_solide_d_un_disque(rayon, distance)
    assert sp.limit(angle, distance, 0, "+") == 2 * sp.pi
    assert sp.limit(angle, distance, sp.oo) == 0
    assert angle_solide_d_un_disque(3, 4) == sp.Rational(2, 5) * sp.pi
    assert angle_solide_d_un_disque(3, -4) == -sp.Rational(2, 5) * sp.pi


def test_le_disque_retrouve_le_saut_de_la_double_couche() -> None:
    """(2.16.5) : de part et d'autre du disque, 2 pi et -2 pi, donc P_S / eps_0."""
    moment, rayon, distance = sp.symbols("P_S a d", positive=True)
    face_1 = sp.limit(angle_solide_d_un_disque(rayon, distance), distance, 0, "+")
    difference = potentiel_de_la_double_couche(
        moment, face_1
    ) - potentiel_de_la_double_couche(moment, -face_1)
    assert sp.simplify(difference - saut_de_potentiel(moment)) == 0


def test_un_disque_vu_de_son_propre_plan_n_a_pas_d_angle_solide() -> None:
    """A distance nulle, le point est sur la couche : ni face 1, ni face 2."""
    with pytest.raises(ValueError, match="point sur la couche"):
        angle_solide_d_un_disque(3, 0)


# --- 2.17 et 2.18 Relations de passage -----------------------------------------


def test_la_composante_tangentielle_du_champ_est_continue() -> None:
    """(2.17.4) : u_n1 ^ E_1 + u_n2 ^ E_2 = 0 a la traversee du plan charge."""
    sigma = sp.Symbol("sigma", positive=True)
    dessus = (0, 0, champ_du_plan(sigma).subs(Z, 1))
    dessous = (0, 0, champ_du_plan(sigma).subs(Z, -1))
    assert residu_tangentiel(dessus, dessous, normale_1=HAUT) == (0, 0, 0)
    assert residu_tangentiel((1, 0, 5), (3, 0, 5), normale_1=HAUT) == (0, -2, 0)


def test_la_composante_normale_du_deplacement_saute_de_la_charge_surfacique() -> None:
    """(2.18.2) : u_n1 . D_1 + u_n2 . D_2 = sigma, verifie sur l'exemple 2.13.2."""
    sigma = sp.Symbol("sigma", positive=True)
    dessus = (0, 0, PERMITTIVITE * champ_du_plan(sigma).subs(Z, 1))
    dessous = (0, 0, PERMITTIVITE * champ_du_plan(sigma).subs(Z, -1))
    assert saut_normal(dessus, dessous, normale_1=HAUT) == sigma


def test_le_saut_du_champ_a_la_traversee_du_plan_charge() -> None:
    """(2.18.5) et page 2-54 : une discontinuite d'amplitude sigma / eps_0."""
    sigma = sp.Symbol("sigma", positive=True)
    dessus = champ_du_potentiel(potentiel_du_plan(sigma).subs(sp.Abs(Z), Z))
    dessous = champ_du_potentiel(potentiel_du_plan(sigma).subs(sp.Abs(Z), -Z))
    assert saut_normal(dessus, dessous, normale_1=HAUT) == sigma / PERMITTIVITE
