"""Syllabus d'electricite generale, chapitre 2 : le segment et la sphere.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphe 2.6, exemples 2.6.2 (pages 2-17 a 2-22) et 2.6.3 (pages 2-22 a
2-24), lus sur les pages rendues.

Chaque formule fermee du syllabus est recoupee par un autre chemin :
l'integrale (2.6.9) calculee numeriquement, le gradient du potentiel, ou
le theoreme de Gauss que porte deja `electrostatique.gauss`.
"""

from __future__ import annotations

import math

import mpmath
import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMITTIVITE_DU_VIDE
from polytech.electromagnetisme.domain.electrostatique.cylindre import (
    PERMITTIVITE,
    potentiel,
)
from polytech.electromagnetisme.domain.electrostatique.gauss import (
    champ_hors_d_une_sphere_uniforme,
    charge_dans_une_boule_uniforme,
)
from polytech.electromagnetisme.domain.electrostatique.segment import (
    champ_de_la_droite_chargee,
    champ_du_segment,
    potentiel_du_segment,
    potentiel_par_les_distances,
    tension_entre_deux_rayons,
)
from polytech.electromagnetisme.domain.electrostatique.sphere_uniforme import (
    champ_interieur,
    potentiel_exterieur,
    potentiel_interieur,
)
from polytech.electromagnetisme.domain.symboles import COTE, RAYON

LAMBDA, BAS, HAUT = 2e-9, 0.3, 0.5
"""Un segment de 2 nC/m, de z = -0,3 m a z = +0,5 m."""

POINTS = [(0.2, 0.1), (0.4, -0.6), (1.5, 0.9), (0.05, 0.45)]
K = 1 / (4 * math.pi * PERMITTIVITE_DU_VIDE)


def _nombre(expression: sp.Expr, rho: float, z: float) -> float:
    valeurs = {RAYON: rho, COTE: z, PERMITTIVITE: PERMITTIVITE_DU_VIDE}
    return float(sp.sympify(expression).subs(valeurs))


def _par_l_integrale(rho: float, z: float) -> tuple[float, float]:
    """(2.6.9), integree numeriquement sur le segment."""

    def noyau(source: float) -> float:
        return float((rho**2 + (z - source) ** 2) ** -1.5)

    radial = mpmath.quad(lambda s: rho * noyau(s), [-BAS, HAUT])
    axial = mpmath.quad(lambda s: (z - s) * noyau(s), [-BAS, HAUT])
    return (float(K * LAMBDA * radial), float(K * LAMBDA * axial))


# --- Exemple 2.6.2 : champ d'un segment de droite ------------------------------


@pytest.mark.parametrize(("rho", "z"), POINTS)
def test_le_champ_du_segment_recoupe_l_integrale(rho: float, z: float) -> None:
    """(2.6.10) et (2.6.11), contre l'integrale (2.6.9) calculee numeriquement."""
    e_rho, e_z = champ_du_segment(LAMBDA, BAS, HAUT)
    radial, axial = _par_l_integrale(rho, z)
    assert math.isclose(_nombre(e_rho, rho, z), radial, rel_tol=1e-9)
    assert math.isclose(_nombre(e_z, rho, z), axial, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize(("rho", "z"), POINTS)
def test_le_champ_du_segment_derive_de_son_potentiel(rho: float, z: float) -> None:
    """Page 2-19 : (2.5.5) appliquee a (2.6.13) redonne (2.6.10) et (2.6.11)."""
    e_rho, e_z = champ_du_segment(LAMBDA, BAS, HAUT)
    v = potentiel_du_segment(LAMBDA, BAS, HAUT)
    assert math.isclose(_nombre(e_rho, rho, z), _nombre(-sp.diff(v, RAYON), rho, z))
    assert math.isclose(
        _nombre(e_z, rho, z), _nombre(-sp.diff(v, COTE), rho, z), abs_tol=1e-12
    )


@pytest.mark.parametrize(("rho", "z"), POINTS)
def test_le_champ_du_segment_par_les_angles(rho: float, z: float) -> None:
    """(2.6.10) et (2.6.11) : -(sin t2 - sin t1) et (cos t2 - cos t1), t1 > 0 > t2."""
    theta_1, theta_2 = math.atan((z + BAS) / rho), math.atan((z - HAUT) / rho)
    e_rho, e_z = champ_du_segment(LAMBDA, BAS, HAUT)
    facteur = K * LAMBDA / rho
    radial = -facteur * (math.sin(theta_2) - math.sin(theta_1))
    axial = facteur * (math.cos(theta_2) - math.cos(theta_1))
    assert math.isclose(_nombre(e_rho, rho, z), radial)
    assert math.isclose(_nombre(e_z, rho, z), axial, abs_tol=1e-12)


def test_dans_le_plan_mediateur_le_champ_est_radial() -> None:
    """A mi-hauteur du segment, les deux moities se compensent selon z."""
    _, e_z = champ_du_segment(LAMBDA, BAS, HAUT)
    assert math.isclose(_nombre(e_z, 0.7, (HAUT - BAS) / 2), 0.0, abs_tol=1e-15)
    assert _nombre(e_z, 0.7, 0.4) > 0
    assert _nombre(e_z, 0.7, -0.2) < 0


@pytest.mark.parametrize(("rho", "z"), POINTS)
def test_les_deux_ecritures_du_potentiel_du_segment(rho: float, z: float) -> None:
    """(2.6.13) et (2.6.15) : ln[(R1 + R2 + l) / (R1 + R2 - l)]."""
    direct = _nombre(potentiel_du_segment(LAMBDA, BAS, HAUT), rho, z)
    par_les_distances = _nombre(potentiel_par_les_distances(LAMBDA, BAS, HAUT), rho, z)
    assert math.isclose(direct, par_les_distances)
    assert direct > 0


def test_les_equipotentielles_du_segment_sont_des_ellipsoides() -> None:
    """Page 2-19 : memes foyers, meme somme R1 + R2, meme potentiel."""
    demi_grand_axe, centre = 0.9, (HAUT - BAS) / 2
    demi_focale = (HAUT + BAS) / 2
    demi_petit_axe = math.sqrt(demi_grand_axe**2 - demi_focale**2)
    v = potentiel_du_segment(LAMBDA, BAS, HAUT)
    sur_l_ellipse = [
        _nombre(v, demi_petit_axe * math.sin(t), centre + demi_grand_axe * math.cos(t))
        for t in (0.3, 1.0, 1.57, 2.5)
    ]
    assert all(math.isclose(valeur, sur_l_ellipse[0]) for valeur in sur_l_ellipse)
    assert not math.isclose(_nombre(v, 0.2, 0.1), sur_l_ellipse[0], rel_tol=0.1)


def test_loin_du_segment_le_potentiel_est_celui_de_sa_charge() -> None:
    """« Lorsque le point P est rejete a l'infini, le potentiel y est nul »."""
    v = potentiel_du_segment(LAMBDA, BAS, HAUT)
    charge = LAMBDA * (BAS + HAUT)
    assert math.isclose(_nombre(v, 300.0, 400.0), K * charge / 500.0, rel_tol=1e-3)


def test_la_droite_infinie_est_la_limite_du_segment() -> None:
    """(2.6.12) : E = lambda / (2 pi eps_0 rho) u_rho, purement radial."""
    densite, demi = sp.symbols("lambda l", positive=True)
    e_rho, e_z = champ_du_segment(densite, demi, demi)
    limite = sp.limit(e_rho.subs(COTE, 0), demi, sp.oo)
    assert sp.simplify(limite - champ_de_la_droite_chargee(densite)) == 0
    assert sp.simplify(e_z.subs(COTE, 0)) == 0
    assert champ_de_la_droite_chargee(densite) == densite / (
        2 * sp.pi * PERMITTIVITE * RAYON
    )


def test_le_potentiel_du_segment_diverge_quand_il_devient_infini() -> None:
    """(2.6.16) et (2.6.17) : le logarithme de 2l / rho n'a pas de limite."""
    densite, demi = sp.symbols("lambda l", positive=True)
    au_milieu = potentiel_du_segment(densite, demi, demi).subs(COTE, 0)
    attendu = (
        densite
        / (2 * sp.pi * PERMITTIVITE)
        * sp.log((demi + sp.sqrt(RAYON**2 + demi**2)) / RAYON)
    )
    valeurs = {densite: 1.0, demi: 3.0, RAYON: 0.7, PERMITTIVITE: 1.0}
    assert math.isclose(float(au_milieu.subs(valeurs)), float(attendu.subs(valeurs)))
    assert sp.limit(au_milieu, demi, sp.oo) == sp.oo


def test_la_tension_entre_deux_cylindres_autour_de_la_droite() -> None:
    """(2.6.19) : U_12 = lambda / (2 pi eps_0) ln(rho_2 / rho_1)."""
    assert math.isclose(
        float(
            tension_entre_deux_rayons(LAMBDA, 0.1, 0.4).subs(
                PERMITTIVITE, PERMITTIVITE_DU_VIDE
            )
        ),
        2 * K * LAMBDA * math.log(4.0),
    )
    assert tension_entre_deux_rayons(LAMBDA, 0.4, 0.4) == 0


def test_la_tension_recoupe_le_potentiel_logarithmique_du_cylindre() -> None:
    """(2.6.21), que porte deja « Statique 1.1 » : le potentiel logarithmique."""
    charge, longueur, reference = sp.symbols("Q L rho_0", positive=True)
    rho_1, rho_2 = sp.symbols("rho_1 rho_2", positive=True)
    difference = potentiel(charge, longueur, rho_1, reference) - potentiel(
        charge, longueur, rho_2, reference
    )
    tension = tension_entre_deux_rayons(charge / longueur, rho_1, rho_2)
    assert sp.simplify(sp.expand_log(difference - tension, force=True)) == 0


def test_la_tension_refuse_un_rayon_nul() -> None:
    """Sur la droite elle-meme, le potentiel logarithmique n'est pas defini."""
    with pytest.raises(ValueError, match="rayon strictement positif attendu"):
        tension_entre_deux_rayons(LAMBDA, 0.0, 0.4)


# --- Exemple 2.6.3 : potentiel d'une sphere ------------------------------------


def test_l_integrale_2_6_24() -> None:
    """Entre -1 et 1 : (|z + r| - |z - r|) / (z r), soit 2/z dehors et 2/r dedans."""
    mu = sp.Symbol("mu", real=True)
    for cote, rayon, attendu in ((3, 2, sp.Rational(2, 3)), (1, 2, 1)):
        integrale = sp.integrate(
            1 / sp.sqrt(cote**2 + rayon**2 - 2 * cote * rayon * mu), (mu, -1, 1)
        )
        assert integrale == attendu
        assert attendu == sp.Rational(
            abs(cote + rayon) - abs(cote - rayon), cote * rayon
        )


def test_le_potentiel_de_la_sphere_est_continu_a_sa_surface() -> None:
    """Page 2-24 : (2.6.25) et (2.6.26) « donnent bien le meme potentiel en z = a »."""
    charge, rayon = sp.symbols("Q a", positive=True)
    dehors = potentiel_exterieur(charge, rayon)
    dedans = potentiel_interieur(charge, rayon, rayon)
    assert sp.simplify(dehors - dedans) == 0
    assert dehors == charge / (4 * sp.pi * PERMITTIVITE * rayon)


def test_le_potentiel_interieur_par_la_charge_volumique() -> None:
    """(2.6.26) : eta (3 a^2 - z^2) / (6 eps_0), avec eta = Q / (4 pi a^3 / 3)."""
    charge, rayon, distance = sp.symbols("Q a r", positive=True)
    volumique = charge / (sp.Rational(4, 3) * sp.pi * rayon**3)
    attendu = volumique * (3 * rayon**2 - distance**2) / (6 * PERMITTIVITE)
    assert sp.simplify(potentiel_interieur(charge, rayon, distance) - attendu) == 0


def test_le_potentiel_au_centre_vaut_une_fois_et_demie_celui_de_la_surface() -> None:
    """Figure 2.6.6 : normalise a son maximum, le potentiel vaut 2/3 en r = a."""
    charge, rayon = sp.symbols("Q a", positive=True)
    au_centre = potentiel_interieur(charge, rayon, 0)
    assert sp.simplify(potentiel_exterieur(charge, rayon) / au_centre) == sp.Rational(
        2, 3
    )


def test_le_champ_de_la_sphere_derive_de_ses_potentiels() -> None:
    """(2.6.27) et (2.6.28) : E = -dV/dr, dehors comme dedans."""
    charge, rayon, distance = sp.symbols("Q a r", positive=True)
    dedans = -sp.diff(potentiel_interieur(charge, rayon, distance), distance)
    assert sp.simplify(dedans - champ_interieur(charge, rayon, distance)) == 0
    dehors = -sp.diff(potentiel_exterieur(charge, distance), distance)
    par_gauss = champ_hors_d_une_sphere_uniforme(charge, rayon, distance)
    assert sp.simplify(dehors - par_gauss) == 0


def test_le_champ_interieur_croit_lineairement_jusqu_a_la_surface() -> None:
    """(2.6.28) : Q r / (4 pi eps_0 a^3), continu en r = a avec le champ exterieur."""
    charge, rayon, distance = sp.symbols("Q a r", positive=True)
    champ = champ_interieur(charge, rayon, distance)
    assert champ == charge * distance / (4 * sp.pi * PERMITTIVITE * rayon**3)
    a_la_surface = -sp.diff(potentiel_exterieur(charge, distance), distance).subs(
        distance, rayon
    )
    assert sp.simplify(champ.subs(distance, rayon) - a_la_surface) == 0


def test_le_champ_interieur_recoupe_le_theoreme_de_gauss() -> None:
    """Le flux a travers la sphere de rayon r vaut la charge qu'elle enlace."""
    charge, rayon, distance = sp.symbols("Q a r", positive=True)
    flux = (
        PERMITTIVITE
        * champ_interieur(charge, rayon, distance)
        * 4
        * sp.pi
        * distance**2
    )
    enlacee = charge_dans_une_boule_uniforme(charge, rayon, distance)
    assert sp.simplify(flux - enlacee) == 0
