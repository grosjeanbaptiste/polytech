"""Syllabus d'electricite generale, chapitre 8 : aimantation et theoreme d'Ampere.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphes 8.10 a 8.20, pages 8-20 a 8-43, lues sur les pages rendues.

La seconde moitie du chapitre est le miroir de l'electrostatique : moment
magnetique pour moment dipolaire, aimantation pour polarisation, courants
fictifs pour charges fictives, `H` pour `D`. Les tests de lien le
verifient sur ce que les chapitres 2 et 3 portent deja -- le champ du
dipole, les relations de passage, l'angle solide du disque.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import (
    CHARGE_ELEMENTAIRE,
    PERMEABILITE_DU_VIDE,
)
from polytech.electromagnetisme.domain.electrostatique.avancee import laplacien
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.double_couche import (
    angle_solide_d_un_disque,
)
from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
    rotationnel_cartesien,
)
from polytech.electromagnetisme.domain.electrostatique.multipoles import champ_du_dipole
from polytech.electromagnetisme.domain.electrostatique.passage import (
    residu_tangentiel,
    saut_normal,
)
from polytech.electromagnetisme.domain.electrostatique.potentiel import circulation
from polytech.electromagnetisme.domain.magnetostatique.aimantation import (
    aimantation,
    courant_surfacique_fictif,
    courant_volumique_fictif,
    induction_du_moment,
    moment_amperien,
    moment_d_un_circuit,
    moment_d_un_circuit_plan,
    potentiel_vecteur_du_moment,
)
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
)
from polytech.electromagnetisme.domain.magnetostatique.boucle import induction_sur_l_axe
from polytech.electromagnetisme.domain.magnetostatique.champ_magnetique import (
    champ_d_un_conducteur_cylindrique,
    champ_magnetique,
    potentiel_scalaire_d_une_boucle,
    potentiel_scalaire_du_fil,
)
from polytech.electromagnetisme.domain.magnetostatique.lois import (
    champ_depuis_l_induction,
)
from polytech.electromagnetisme.domain.magnetostatique.potentiel_vecteur import (
    induction_du_potentiel,
)
from polytech.electromagnetisme.domain.magnetostatique.regions import courant_enlace
from polytech.electromagnetisme.domain.magnetostatique.segment_de_courant import (
    induction_de_la_droite,
    potentiel_vecteur_de_la_droite,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES, RAYON

X, Y, Z = CARTESIENNES
T = sp.Symbol("t", real=True)
COURANT, RAYON_DU_FIL = sp.symbols("I a", positive=True)
HAUT, BAS = (0, 0, 1), (0, 0, -1)


def _en_cartesiennes(orthoradial: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Un champ `f(rho) u_phi`, ecrit sur les axes x, y, z."""
    distance = sp.sqrt(X**2 + Y**2)
    module = orthoradial.subs(RAYON, distance)
    return (-module * Y / distance, module * X / distance, sp.Integer(0))


# --- 8.10 Moment magnetique amperien, aimantation ------------------------------


def test_le_moment_amperien_d_une_charge_sur_son_orbite() -> None:
    """(8.10.14) : m = (q / T) S u_n -- un courant moyen par une aire."""
    assert moment_amperien(charge=6, periode=2, aire=5, normale=HAUT) == (0, 0, 15)


def test_le_moment_d_un_electron_est_oppose_au_sens_de_son_orbite() -> None:
    """Page 8-25 : q negatif, « ce courant moyen est de sens oppose au parcours »."""
    _, _, selon_z = moment_amperien(
        charge=-CHARGE_ELEMENTAIRE, periode=1.5e-16, aire=8.8e-21, normale=HAUT
    )
    assert selon_z < 0
    assert math.isclose(selon_z, -CHARGE_ELEMENTAIRE / 1.5e-16 * 8.8e-21)


def test_une_periode_nulle_n_est_pas_une_orbite() -> None:
    """Le rapport q / T est un courant moyen : la periode est strictement positive."""
    with pytest.raises(ValueError, match="periode strictement positive attendue"):
        moment_amperien(charge=1, periode=0, aire=1, normale=HAUT)


def test_l_aimantation_est_la_somme_des_moments_par_unite_de_volume() -> None:
    """(8.10.16) : M = somme des m_i / Delta V."""
    assert aimantation([(0, 0, 2), (0, 0, 4), (3, 0, 0)], 2) == (
        sp.Rational(3, 2),
        0,
        3,
    )
    assert aimantation([(0, 0, 2), (0, 0, -2)], 5) == (0, 0, 0)


def test_le_moment_d_un_circuit_plan() -> None:
    """(8.15.12) : m = I S u_n."""
    assert moment_d_un_circuit_plan(COURANT, sp.pi * RAYON_DU_FIL**2, HAUT) == (
        0,
        0,
        COURANT * sp.pi * RAYON_DU_FIL**2,
    )


def test_le_moment_d_une_boucle_par_l_integrale_de_contour() -> None:
    """(8.15.12) : (I / 2) circulation de r ^ dl = I pi a^2 u_z pour un cercle."""
    cercle = (RAYON_DU_FIL * sp.cos(T), RAYON_DU_FIL * sp.sin(T), sp.Integer(0))
    par_le_contour = moment_d_un_circuit(COURANT, cercle, T, 0, 2 * sp.pi)
    assert par_le_contour == moment_d_un_circuit_plan(
        COURANT, sp.pi * RAYON_DU_FIL**2, HAUT
    )


def test_le_moment_d_un_circuit_ne_depend_pas_de_l_origine() -> None:
    """Page 8-36 : « independant de la forme du circuit et du choix de l'origine »."""
    centre = (RAYON_DU_FIL * sp.cos(T), RAYON_DU_FIL * sp.sin(T), sp.Integer(0))
    deplace = (
        5 + RAYON_DU_FIL * sp.cos(T),
        -2 + RAYON_DU_FIL * sp.sin(T),
        sp.Integer(7),
    )
    assert moment_d_un_circuit(COURANT, deplace, T, 0, 2 * sp.pi) == (
        moment_d_un_circuit(COURANT, centre, T, 0, 2 * sp.pi)
    )


def test_un_circuit_parcouru_dans_l_autre_sens_a_le_moment_oppose() -> None:
    """Le sens de u_n est « associe au sens de parcours » du contour."""
    horaire = (RAYON_DU_FIL * sp.cos(T), -RAYON_DU_FIL * sp.sin(T), sp.Integer(0))
    assert moment_d_un_circuit(COURANT, horaire, T, 0, 2 * sp.pi) == (
        0,
        0,
        -COURANT * sp.pi * RAYON_DU_FIL**2,
    )


def test_le_potentiel_vecteur_d_un_moment_magnetique() -> None:
    """(8.10.21) et (8.15.11) : A = (mu_0 / 4 pi) m ^ r / r^3."""
    moment = sp.Symbol("m", positive=True)
    a_x, a_y, a_z = potentiel_vecteur_du_moment((0, 0, moment))
    distance = sp.sqrt(X**2 + Y**2 + Z**2)
    facteur = PERMEABILITE * moment / (4 * sp.pi * distance**3)
    assert (sp.simplify(a_x + facteur * Y), sp.simplify(a_y - facteur * X), a_z) == (
        0,
        0,
        0,
    )


def test_l_induction_d_un_moment_a_la_forme_du_champ_d_un_dipole() -> None:
    """(8.10.21) est « l'analogue de (2.7.8) » : B = mu_0 eps_0 E, de p vers m."""
    moment = (1, -2, 3)
    magnetique = induction_du_moment(moment)
    electrique = champ_du_dipole(moment)
    ecarts = [
        sp.simplify(b - PERMEABILITE * PERMITTIVITE * e)
        for b, e in zip(magnetique, electrique, strict=True)
    ]
    assert ecarts == [0, 0, 0]


def test_l_induction_du_moment_derive_de_son_potentiel_vecteur() -> None:
    """(8.6.3) appliquee a (8.10.21)."""
    moment = (1, -2, 3)
    par_le_rotationnel = induction_du_potentiel(potentiel_vecteur_du_moment(moment))
    ecarts = [
        sp.simplify(a - b)
        for a, b in zip(induction_du_moment(moment), par_le_rotationnel, strict=True)
    ]
    assert ecarts == [0, 0, 0]


def test_loin_sur_son_axe_la_boucle_est_un_moment_magnetique() -> None:
    """8.15 : le premier terme non nul est dipolaire, m = I pi a^2."""
    cote = sp.Symbol("z", positive=True)
    moment = moment_d_un_circuit_plan(COURANT, sp.pi * RAYON_DU_FIL**2, HAUT)
    dipolaire = induction_du_moment(moment)[2].subs({X: 0, Y: 0, Z: cote})
    exacte = induction_sur_l_axe(COURANT, RAYON_DU_FIL, cote)
    assert sp.limit(exacte / dipolaire, cote, sp.oo) == 1
    assert (
        sp.simplify(dipolaire - PERMEABILITE * moment[2] / (2 * sp.pi * cote**3)) == 0
    )


# --- 8.11 Courants fictifs -----------------------------------------------------


def test_une_aimantation_uniforme_n_a_pas_de_courant_volumique_fictif() -> None:
    """Page 8-29 : pour qu'apparaisse i', « il faudrait que M ne soit pas uniforme »."""
    assert courant_volumique_fictif((0, 0, 7)) == (0, 0, 0)


def test_le_courant_volumique_fictif_est_le_rotationnel_de_l_aimantation() -> None:
    """(8.11.1a) : i' = rot M ; sa divergence est nulle (8.11.5)."""
    pente = sp.Symbol("k", positive=True)
    fictif = courant_volumique_fictif((0, 0, pente * X))
    assert fictif == (0, -pente, 0)
    assert (
        divergence_cartesienne(courant_volumique_fictif((Y * Z, X**2, sp.sin(X * Y))))
        == 0
    )


def test_le_courant_surfacique_fictif_d_un_barreau_aimante() -> None:
    """(8.11.1b) et figure 8.11.1 : i'_S = u_n ^ M, u_n vers l'interieur."""
    aimante = (0, 0, sp.Symbol("M", positive=True))
    sur_le_flanc = courant_surfacique_fictif(aimante, normale_interieure=(-1, 0, 0))
    assert sur_le_flanc == (0, aimante[2], 0)
    assert courant_surfacique_fictif(aimante, normale_interieure=BAS) == (0, 0, 0)


def test_le_courant_fictif_total_d_une_tranche_est_nul() -> None:
    """(8.11.4) : M = k x u_z entre x = 0 et x = w -- le volume compense la surface."""
    pente, largeur = sp.symbols("k w", positive=True)
    aimante = (0, 0, pente * X)
    en_volume = sp.integrate(courant_volumique_fictif(aimante)[1], (X, 0, largeur))
    en_surface = courant_surfacique_fictif(aimante, normale_interieure=(-1, 0, 0))[1]
    assert en_volume + en_surface.subs(X, largeur) == 0


def test_la_normale_du_courant_surfacique_est_unitaire() -> None:
    """La relation (8.11.1b) emploie un vecteur unitaire."""
    with pytest.raises(ValueError, match="normale unitaire attendue"):
        courant_surfacique_fictif((0, 0, 1), normale_interieure=(2, 0, 0))


# --- 8.14 Expression globale du potentiel vecteur ------------------------------


def test_hors_des_courants_le_potentiel_vecteur_est_harmonique() -> None:
    """(8.14.5) : lap A = -mu_0 (i + i'), donc nul autour de la droite de (8.9.8)."""
    reference = sp.Symbol("rho_0", positive=True)
    selon_z = potentiel_vecteur_de_la_droite(COURANT, reference).subs(
        RAYON, sp.sqrt(X**2 + Y**2)
    )
    assert sp.simplify(laplacien(selon_z)) == 0


# --- 8.16 Vecteur champ magnetique, theoreme d'Ampere --------------------------


def test_le_vecteur_champ_magnetique() -> None:
    """(8.16.2) : H = B / mu_0 - M."""
    assert champ_magnetique((PERMEABILITE, 0, 3 * PERMEABILITE), (0, 2, 1)) == (
        1,
        -2,
        2,
    )


def test_dans_le_vide_le_champ_recoupe_celui_du_fascicule() -> None:
    """(8.16.2) sans aimantation : H = B / mu_0, que le fascicule portait deja."""
    selon_z = champ_magnetique((0, 0, 0.5), (0, 0, 0))[2]
    assert math.isclose(
        float(selon_z.subs(PERMEABILITE, PERMEABILITE_DU_VIDE)),
        champ_depuis_l_induction(0.5),
    )


def test_le_champ_d_un_conducteur_infiniment_long() -> None:
    """Exemple 8.16.1 : I rho / (2 pi a^2) dedans, I / (2 pi rho) dehors."""
    champ = champ_d_un_conducteur_cylindrique(COURANT, RAYON_DU_FIL)
    dedans = champ.subs(RAYON, RAYON_DU_FIL / 2)
    dehors = champ.subs(RAYON, 3 * RAYON_DU_FIL)
    assert sp.simplify(dedans - COURANT / (4 * sp.pi * RAYON_DU_FIL)) == 0
    assert sp.simplify(dehors - COURANT / (6 * sp.pi * RAYON_DU_FIL)) == 0
    assert (
        sp.simplify(
            champ.subs(RAYON, RAYON_DU_FIL) - COURANT / (2 * sp.pi * RAYON_DU_FIL)
        )
        == 0
    )


def test_dehors_le_conducteur_est_une_droite_placee_sur_son_axe() -> None:
    """Page 8-39 : (8.16.6) redonne (8.9.7), B = mu_0 H."""
    dehors = champ_d_un_conducteur_cylindrique(COURANT, RAYON_DU_FIL).subs(
        RAYON, 3 * RAYON_DU_FIL
    )
    droite = induction_de_la_droite(COURANT).subs(RAYON, 3 * RAYON_DU_FIL)
    assert sp.simplify(PERMEABILITE * dehors - droite) == 0


def test_le_theoreme_d_ampere_sur_un_cercle_interieur() -> None:
    """(8.16.3) et (8.16.5) : la circulation de H vaut le courant enlace."""
    interieur = COURANT * RAYON / (2 * sp.pi * RAYON_DU_FIL**2)
    rayon = sp.Symbol("r", positive=True)
    cercle = (rayon * sp.cos(T), rayon * sp.sin(T), sp.Integer(0))
    enlace = circulation(_en_cartesiennes(interieur), cercle, T, 0, 2 * sp.pi)
    assert sp.simplify(enlace - courant_enlace(COURANT, rayon, RAYON_DU_FIL)) == 0


def test_le_theoreme_d_ampere_sous_sa_forme_locale() -> None:
    """(8.16.4) : rot H = i, soit I / (pi a^2) dedans et rien dehors."""
    interieur = COURANT * RAYON / (2 * sp.pi * RAYON_DU_FIL**2)
    exterieur = COURANT / (2 * sp.pi * RAYON)
    dedans = [
        sp.simplify(c) for c in rotationnel_cartesien(_en_cartesiennes(interieur))
    ]
    dehors = [
        sp.simplify(c) for c in rotationnel_cartesien(_en_cartesiennes(exterieur))
    ]
    assert dedans == [0, 0, COURANT / (sp.pi * RAYON_DU_FIL**2)]
    assert dehors == [0, 0, 0]


# --- 8.18 et 8.19 Relations de passage -----------------------------------------


def _champ_d_une_nappe(courant_surfacique: sp.Expr, abscisse: sp.Expr) -> sp.Expr:
    """`H_y` d'une nappe `K u_z` portee par le plan x = 0, somme de droites."""
    source = sp.Symbol("y_s", real=True)
    par_fil = courant_surfacique / (2 * sp.pi) * abscisse / (abscisse**2 + source**2)
    return sp.integrate(par_fil, (source, -sp.oo, sp.oo))


def test_la_composante_tangentielle_du_champ_saute_du_courant_surfacique() -> None:
    """(8.19.4) : u_n1 ^ H_1 + u_n2 ^ H_2 = i_S, sur une nappe de courant plane."""
    nappe, distance = sp.symbols("K d", positive=True)
    devant = (0, _champ_d_une_nappe(nappe, distance), 0)
    derriere = (0, _champ_d_une_nappe(nappe, -distance), 0)
    assert devant == (0, nappe / 2, 0)
    assert residu_tangentiel(devant, derriere, normale_1=(1, 0, 0)) == (0, 0, nappe)


def test_la_composante_normale_de_l_induction_est_continue() -> None:
    """(8.18.2) : u_n1 . B_1 + u_n2 . B_2 = 0, quelle que soit la nappe."""
    nappe = sp.Symbol("K", positive=True)
    devant = (sp.Integer(3), PERMEABILITE * nappe / 2, 0)
    derriere = (sp.Integer(3), -PERMEABILITE * nappe / 2, 0)
    assert saut_normal(devant, derriere, normale_1=(1, 0, 0)) == 0


# --- 8.20 Potentiel scalaire magnetique ----------------------------------------


def test_le_champ_du_fil_derive_d_un_potentiel_scalaire() -> None:
    """(8.20.1) : H = -grad U hors des courants, pour la droite de (8.9.7)."""
    potentiel = potentiel_scalaire_du_fil(COURANT)
    champ = tuple(-sp.diff(potentiel, axe) for axe in CARTESIENNES)
    attendu = _en_cartesiennes(induction_de_la_droite(COURANT) / PERMEABILITE)
    assert [sp.simplify(a - b) for a, b in zip(champ, attendu, strict=True)] == [
        0,
        0,
        0,
    ]


def test_le_potentiel_scalaire_magnetique_est_multiforme() -> None:
    """(8.20.2) : apres un tour autour du courant, le potentiel a baisse de I."""
    potentiel = potentiel_scalaire_du_fil(COURANT)
    juste_apres = potentiel.subs({X: 1, Y: sp.Rational(1, 10**9)})
    juste_avant = potentiel.subs({X: 1, Y: -sp.Rational(1, 10**9)})
    assert math.isclose(float((juste_avant - juste_apres) / COURANT), 0.0, abs_tol=1e-8)
    oppose_haut = potentiel.subs({X: -1, Y: sp.Rational(1, 10**9)})
    oppose_bas = potentiel.subs({X: -1, Y: -sp.Rational(1, 10**9)})
    assert math.isclose(float((oppose_haut - oppose_bas) / COURANT), -1.0, abs_tol=1e-8)


def test_le_potentiel_scalaire_d_une_boucle_par_l_angle_solide() -> None:
    """(8.20.3) : U = I Omega / (4 pi), Omega etant l'angle solide du disque."""
    cote = sp.Symbol("z", positive=True)
    potentiel = potentiel_scalaire_d_une_boucle(COURANT, RAYON_DU_FIL, cote)
    angle = angle_solide_d_un_disque(RAYON_DU_FIL, cote)
    assert sp.simplify(potentiel - COURANT * angle / (4 * sp.pi)) == 0


def test_le_potentiel_scalaire_de_la_boucle_redonne_l_induction_sur_l_axe() -> None:
    """(8.20.1) et (8.9.14) : -dU/dz = B_z / mu_0."""
    cote = sp.Symbol("z", positive=True)
    champ = -sp.diff(potentiel_scalaire_d_une_boucle(COURANT, RAYON_DU_FIL, cote), cote)
    attendu = induction_sur_l_axe(COURANT, RAYON_DU_FIL, cote) / PERMEABILITE
    assert sp.simplify(champ - attendu) == 0
