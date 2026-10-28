"""Syllabus d'electricite generale, chapitre 2 : forces, travail, energie.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphes 2.4 et 2.19 a 2.21, pages 2-8 et 2-64 a 2-77, lues sur les
pages rendues.

Trois ecarts, les deux premiers constates en agrandissant la page 2-69 :

* exemple 2.20.1 : la force gravitationnelle est ecrite avec
  `1 846 (0,9 . 10^-19)^2`. Le produit `m_e m_p` vaut `1 836 m_e^2` (le
  tableau 1.3.1 dit « 1 836 fois plus lourd ») avec `m_e = 0,9 . 10^-30 kg` ;
  c'est d'ailleurs ce calcul-la qui rend le `35 . 10^-48 N` imprime ;
* meme exemple : `P_p = 16,7 . 10^-27 N` suppose `g = 10 m/s^2`, alors que
  `P_e = 9,0 . 10^-30 N` suppose `g = 9,81` (avec 10, on lirait 9,1) ;
* exemple 2.21.2 : l'energie du noyau d'uranium est imprimee 813,5 MeV ;
  `(3/5) Z^2 e^2 / (4 pi eps_0 a)` donne 812,5 MeV pour `a = 9 fm`. Le
  rayon lui-meme valait 1 fm a l'exemple 1.5.2.
"""

from __future__ import annotations

import math
from itertools import pairwise

import mpmath
import pytest
import sympy as sp

from polytech.chemistry.domain.atomes import RAYON_DE_BOHR
from polytech.electromagnetisme.domain.charges import PARTICULES
from polytech.electromagnetisme.domain.constantes import (
    CHARGE_ELEMENTAIRE,
    PERMITTIVITE_DU_VIDE,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.energie import (
    MADELUNG_DE_LA_CHAINE,
    energie_de_deux_charges,
    energie_de_la_chaine,
    energie_de_la_sphere,
    energie_du_systeme,
    potentiel_sur_la_chaine,
)
from polytech.electromagnetisme.domain.electrostatique.forces import (
    couple_sur_un_dipole,
    force_de_coulomb,
    force_du_systeme,
    force_sur_un_dipole,
    force_sur_une_charge,
)
from polytech.electromagnetisme.domain.electrostatique.lois import champ_de_coulomb
from polytech.electromagnetisme.domain.electrostatique.moments import SystemeDeCharges
from polytech.electromagnetisme.domain.electrostatique.potentiel import (
    COULOMB,
    champ_du_potentiel,
    potentiel_des_charges,
)
from polytech.electromagnetisme.domain.electrostatique.sphere_uniforme import (
    champ_interieur,
    potentiel_interieur,
)
from polytech.electromagnetisme.domain.electrostatique.travail import (
    tension,
    travail_contre_le_champ,
    travail_des_forces_du_champ,
)
from polytech.electromagnetisme.domain.energie_et_forces import (
    densite_d_energie_electrique,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES
from polytech.maths.domain.comparaison import (
    egales,
    egales_arrondies,
    eloignees,
    proches,
    proches_relativement,
    rapport_au_dessus,
    suites_proches,
)
from polytech.mecanique.domain.gravitation import force_gravitationnelle

X, Y, Z = CARTESIENNES
K = 1 / (4 * math.pi * PERMITTIVITE_DU_VIDE)


IMPRIME_813_5 = 813.5  # MeV, exemple 2.21.2 du syllabus d'electricite (valeur imprimee)


def _nombres(vecteur: tuple[sp.Expr, ...]) -> list[float]:
    return [
        float(sp.sympify(c).subs(PERMITTIVITE, PERMITTIVITE_DU_VIDE)) for c in vecteur
    ]


# --- 2.4 Vecteur champ electrique ----------------------------------------------


def test_la_force_sur_une_charge_est_proportionnelle_au_champ() -> None:
    """(2.4.1) : F = q E."""
    assert force_sur_une_charge(2, (1, 0, -3)) == (2, 0, -6)
    assert force_sur_une_charge(-1, (1, 0, -3)) == (-1, 0, 3)


# --- 2.19 Travail des forces du champ ------------------------------------------


def test_la_tension_est_la_difference_des_potentiels() -> None:
    """(2.19.6) : U_12 = V_1 - V_2."""
    assert tension(12, 5) == 7
    assert tension(5, 12) == -tension(12, 5)


def test_le_travail_des_forces_du_champ() -> None:
    """(2.19.5) et (2.19.7) : T_12 = (V_1 - V_2) Q = Q U_12."""
    assert travail_des_forces_du_champ(3, 12, 5) == 21
    assert travail_des_forces_du_champ(-3, 12, 5) == -21


def test_le_travail_contre_les_forces_du_champ() -> None:
    """(2.19.8) : Q U_21 = -T_12 -- positif s'il faut remonter les potentiels."""
    assert travail_contre_le_champ(3, 12, 5) == -21
    assert travail_contre_le_champ(3, 5, 12) == 21


def test_la_tension_se_compose_par_un_potentiel_de_reference() -> None:
    """(2.19.9) et (2.19.10) : U_12 = U_1a + U_a2 = U_10 - U_20."""
    v_1, v_2, v_a, v_0 = sp.symbols("V_1 V_2 V_a V_0", real=True)
    assert tension(v_1, v_a) + tension(v_a, v_2) == tension(v_1, v_2)
    assert egales(tension(v_1, v_0) - tension(v_2, v_0), tension(v_1, v_2))


def test_le_travail_ne_depend_pas_du_chemin_suivi() -> None:
    """(2.19.1) : la circulation de q E, sur deux chemins, vaut Q (V_1 - V_2)."""
    source = SystemeDeCharges(((1e-9, (0, 0, 0)),))
    potentiel = potentiel_des_charges(source).subs(PERMITTIVITE, PERMITTIVITE_DU_VIDE)
    champ = sp.lambdify(CARTESIENNES, champ_du_potentiel(potentiel))
    valeur = sp.lambdify(CARTESIENNES, potentiel)
    depart, arrivee, charge = (1.0, 0.0, 0.0), (0.0, 3.0, 4.0), 2e-6

    def circulation(etapes: list[tuple[float, float, float]]) -> float:
        total = 0.0
        for debut, fin in pairwise(etapes):
            pas = [f - d for d, f in zip(debut, fin, strict=True)]

            def integrande(
                t: float,
                debut: tuple[float, float, float] = debut,
                pas: list[float] = pas,
            ) -> float:
                point = [d + t * p for d, p in zip(debut, pas, strict=True)]
                travail: float = sum(
                    e * p for e, p in zip(champ(*point), pas, strict=True)
                )
                return travail

            total += float(mpmath.quad(integrande, [0, 1]))
        return charge * total

    direct = circulation([depart, arrivee])
    detour = circulation([depart, (2.0, 2.0, 0.0), (0.0, 5.0, 1.0), arrivee])
    attendu = travail_des_forces_du_champ(charge, valeur(*depart), valeur(*arrivee))
    assert proches_relativement(direct, attendu, 1e-9)
    assert proches_relativement(detour, attendu, 1e-9)


# --- 2.20 Loi de Coulomb -------------------------------------------------------


def test_la_force_de_coulomb_entre_deux_charges() -> None:
    """(2.20.2) : F = q q_1 R_1 / (4 pi eps_0 R_1^3), de la source vers la charge."""
    force = _nombres(force_de_coulomb(1e-6, (0, 0, 0), 2e-6, (3, 0, 4)))
    module = K * 2e-12 / 25
    assert proches_relativement(force[0], module * 3 / 5, 1e-9)
    assert force[1] == 0
    assert proches_relativement(force[2], module * 4 / 5, 1e-9)


def test_deux_charges_de_signes_opposes_s_attirent() -> None:
    """Page 2-68 : repulsion ou attraction, selon les signes des deux charges."""
    repulsion = _nombres(force_de_coulomb(1, (0, 0, 0), 1, (1, 0, 0)))
    attraction = _nombres(force_de_coulomb(1, (0, 0, 0), -1, (1, 0, 0)))
    assert repulsion[0] > 0
    assert attraction[0] == -repulsion[0]


def test_l_action_et_la_reaction() -> None:
    """Remarque 1 de la page 2-68 : la force de q sur q_1 est egale et opposee."""
    aller = force_de_coulomb(3, (1, 2, 0), -2, (4, 6, 0))
    retour = force_de_coulomb(-2, (4, 6, 0), 3, (1, 2, 0))
    assert all(egales(a, -r) for a, r in zip(aller, retour, strict=True))


def test_deux_charges_au_meme_point_n_ont_pas_de_force_definie() -> None:
    """Page 2-14 : le champ n'est pas defini « aux points sources »."""
    with pytest.raises(ValueError, match="charges confondues"):
        force_de_coulomb(1, (1, 2, 3), 1, (1, 2, 3))


def test_la_force_d_un_systeme_est_la_somme_des_forces() -> None:
    """(2.20.3) : F = q E, et E est la somme des champs de chaque charge."""
    systeme = SystemeDeCharges(((2, (0, 0, 0)), (-1, (0, 3, 0))))
    point = (4, 0, 0)
    une_a_une = [force_de_coulomb(q, source, 5, point) for q, source in systeme.charges]
    somme = [sum(f[axe] for f in une_a_une) for axe in range(3)]
    totale = force_du_systeme(systeme, 5, point)
    assert all(egales(a, b) for a, b in zip(totale, somme, strict=True))


def test_la_force_electrique_dans_l_atome_d_hydrogene() -> None:
    """Exemple 2.20.1 : 82 . 10^-9 N a 52,9 pm."""
    force = CHARGE_ELEMENTAIRE * champ_de_coulomb(CHARGE_ELEMENTAIRE, RAYON_DE_BOHR)
    assert proches_relativement(force, 82e-9, 0.01)


def test_la_force_gravitationnelle_dans_l_atome_d_hydrogene() -> None:
    """Exemple 2.20.1 : 35 . 10^-48 N, avec 1 836 m_e^2 et m_e = 0,9 . 10^-30 kg."""
    imprime = force_gravitationnelle(0.9e-30, 1836 * 0.9e-30, RAYON_DE_BOHR)
    assert egales_arrondies(imprime * 1e48, 35.4, 1)
    tel_qu_ecrit = force_gravitationnelle(0.9e-19, 1846 * 0.9e-19, RAYON_DE_BOHR)
    assert rapport_au_dessus(tel_qu_ecrit, 35e-48, 1e20)
    reelle = force_gravitationnelle(
        PARTICULES["electron"].masse, PARTICULES["proton"].masse, RAYON_DE_BOHR
    )
    assert egales_arrondies(reelle * 1e48, 36.3, 1)


def test_la_force_electrique_ecrase_la_gravitation() -> None:
    """Exemple 2.20.1 : « environ 10^39 superieure »."""
    electrique = CHARGE_ELEMENTAIRE * champ_de_coulomb(
        CHARGE_ELEMENTAIRE, RAYON_DE_BOHR
    )
    rapport = electrique / force_gravitationnelle(
        PARTICULES["electron"].masse, PARTICULES["proton"].masse, RAYON_DE_BOHR
    )
    assert rapport_au_dessus(rapport, 1.0, 1e39)
    assert not rapport_au_dessus(rapport, 1.0, 1e40)


def test_les_poids_des_particules_de_l_exemple_2_20_1() -> None:
    """P_p = 16,7 . 10^-27 N suppose g = 10 ; P_e = 9,0 . 10^-30 N suppose g = 9,81."""
    proton, electron = PARTICULES["proton"].masse, PARTICULES["electron"].masse
    assert proches_relativement(proton * 10, 16.7e-27, 2e-3)
    assert not proches_relativement(proton * 9.81, 16.7e-27, 1e-2)
    assert proches_relativement(electron * 9.81, 9.0e-30, 1e-2)


def test_la_repulsion_de_deux_protons_dans_un_noyau() -> None:
    """Exemple 2.20.2 : 14 N a 4,0 . 10^-15 m."""
    force = CHARGE_ELEMENTAIRE * champ_de_coulomb(CHARGE_ELEMENTAIRE, 4.0e-15)
    assert proches_relativement(force, 14, 0.04)


def test_un_champ_uniforme_ne_tire_pas_sur_un_doublet() -> None:
    """Exemple 2.20.3 : la resultante (2.20.4) est nulle, le couple vaut p ^ E."""
    assert force_sur_un_dipole((0, 0, 2), (3, 0, 5)) == (0, 0, 0)
    assert couple_sur_un_dipole((0, 0, 2), (3, 0, 5)) == (0, 6, 0)


def test_le_couple_tend_a_aligner_le_dipole_sur_le_champ() -> None:
    """Page 2-70 : « un couple qui tend a l'orienter dans la direction de E »."""
    assert couple_sur_un_dipole((0, 0, 2), (0, 0, 5)) == (0, 0, 0)
    assert couple_sur_un_dipole((2, 0, 0), (0, 5, 0)) == (0, 0, 10)


def test_la_force_sur_un_dipole_dans_un_champ_non_uniforme() -> None:
    """(2.20.8) : F = (p . grad) E = grad(p . E), pour un champ irrotationnel."""
    champ = champ_du_potentiel(X**2 * Y - Z**2 * X)
    moment = (1, -2, 3)
    force = force_sur_un_dipole(moment, champ)
    scalaire = sum(p * e for p, e in zip(moment, champ, strict=True))
    gradient = [sp.diff(scalaire, axe) for axe in CARTESIENNES]
    assert all(egales(a, b) for a, b in zip(force, gradient, strict=True))


def test_la_force_sur_un_dipole_est_la_limite_de_celle_du_doublet() -> None:
    """(2.20.4) et (2.20.7) : q [E(r_1) - E(r_2)] tend vers (p . grad) E."""
    champ = champ_du_potentiel(X**2 * Y)
    ecart = sp.Symbol("d", positive=True)
    charge = 1 / ecart
    en_haut = [c.subs({X: 1, Y: 2, Z: ecart / 2}) for c in champ]
    en_bas = [c.subs({X: 1, Y: 2, Z: -ecart / 2}) for c in champ]
    doublet = [
        sp.limit(charge * (h - b), ecart, 0)
        for h, b in zip(en_haut, en_bas, strict=True)
    ]
    dipole = [c.subs({X: 1, Y: 2, Z: 0}) for c in force_sur_un_dipole((0, 0, 1), champ)]
    assert doublet == dipole


# --- 2.21 Energie potentielle electrostatique ----------------------------------


def test_l_energie_de_deux_charges() -> None:
    """(2.21.3) : W_e = q_1 q_2 / (4 pi eps_0 R_12), nulle a l'infini."""
    q_1, q_2, distance = sp.symbols("q_1 q_2 R", positive=True)
    energie = energie_de_deux_charges(q_1, q_2, distance)
    assert energie == COULOMB * q_1 * q_2 / distance
    assert sp.limit(energie, distance, sp.oo) == 0


def test_l_energie_est_mutuelle() -> None:
    """(2.21.4) : W_e = q_1 V_1 = q_2 V_2 = (q_1 V_1 + q_2 V_2) / 2."""
    paire = SystemeDeCharges(((3, (0, 0, 0)), (-2, (0, 0, 4))))
    attendu = energie_de_deux_charges(3, -2, 4)
    assert egales(energie_du_systeme(paire), attendu)
    assert attendu < 0


def test_l_energie_d_un_systeme_somme_sur_les_couples() -> None:
    """(2.21.5) : chaque couple une fois, soit la demi-somme des q_i V_i (2.21.6)."""
    trio = SystemeDeCharges(((1, (0, 0, 0)), (2, (3, 0, 0)), (-1, (0, 4, 0))))
    couples = (
        energie_de_deux_charges(1, 2, 3)
        + energie_de_deux_charges(1, -1, 4)
        + energie_de_deux_charges(2, -1, 5)
    )
    assert egales(energie_du_systeme(trio), couples)


def test_la_constante_de_madelung_de_la_chaine_lineique() -> None:
    """Exemple 2.21.1 : 2 ln 2 = 1,386."""
    assert egales(MADELUNG_DE_LA_CHAINE, sp.log(4))
    assert proches_relativement(float(MADELUNG_DE_LA_CHAINE), 1.386, 1e-3)
    serie = sp.Sum(
        (-1) ** (sp.Symbol("i") - 1) / sp.Symbol("i"), (sp.Symbol("i"), 1, sp.oo)
    )
    assert serie.doit() == sp.log(2)


def test_le_potentiel_sur_un_site_de_la_chaine() -> None:
    """Exemple 2.21.1 : V(A) = (2 / 4 pi eps_0) somme des q (-1)^i / (i d)."""
    charge, pas = sp.symbols("q d", positive=True)
    assert (
        potentiel_sur_la_chaine(charge, pas, voisins=1) == -2 * COULOMB * charge / pas
    )
    deux = potentiel_sur_la_chaine(charge, pas, voisins=2)
    assert egales(deux, -(COULOMB * charge / pas))
    oppose = potentiel_sur_la_chaine(-charge, pas, voisins=2)
    assert egales(oppose, -(deux))


def test_l_energie_par_couple_tend_vers_la_constante_de_madelung() -> None:
    """Exemple 2.21.1 : W_e / n = q V(A) = 1,386 W_c, W_c etant le couple isole."""
    isole = energie_de_deux_charges(1, -1, 1)
    rapports = [
        float(potentiel_sur_la_chaine(1, 1, voisins=n) / isole) for n in (1, 2, 3, 2000)
    ]
    assert suites_proches(rapports[:3], [2.0, 1.0, 5 / 3], 1e-6, relativement=True)
    assert proches_relativement(rapports[3], 1.386, 1e-3)


def test_l_energie_de_la_chaine_infinie() -> None:
    """Exemple 2.21.1 : W_e = -n q^2 2 ln 2 / (4 pi eps_0 d)."""
    nombre, charge, pas = sp.symbols("n q d", positive=True)
    attendu = -nombre * charge**2 * 2 * sp.log(2) * COULOMB / pas
    assert egales(energie_de_la_chaine(nombre, charge, pas), attendu)


def test_l_energie_de_la_sphere_uniformement_chargee() -> None:
    """Exemple 2.21.2 : (3/5) Q^2 / (4 pi eps_0 a) = 4 pi eta^2 a^5 / (15 eps_0)."""
    charge, rayon = sp.symbols("Q a", positive=True)
    energie = energie_de_la_sphere(charge, rayon)
    assert energie == sp.Rational(3, 5) * COULOMB * charge**2 / rayon
    eta = charge / (sp.Rational(4, 3) * sp.pi * rayon**3)
    assert egales(energie, 4 * sp.pi * eta**2 * rayon**5 / (15 * PERMITTIVITE))


def test_l_energie_de_la_sphere_par_la_charge_et_le_potentiel() -> None:
    """(2.21.8) : (1/2) integrale de eta V sur la sphere."""
    charge, rayon, r = sp.symbols("Q a r", positive=True)
    eta = charge / (sp.Rational(4, 3) * sp.pi * rayon**3)
    integrande = eta * potentiel_interieur(charge, rayon, r) * 4 * sp.pi * r**2
    par_le_potentiel = sp.integrate(integrande, (r, 0, rayon)) / 2
    assert egales(par_le_potentiel, energie_de_la_sphere(charge, rayon))


def test_l_energie_de_la_sphere_par_le_champ_dans_tout_l_espace() -> None:
    """(2.21.13) : (1/2) integrale de E . D, dedans et dehors -- le meme resultat."""
    charge, rayon, r = sp.symbols("Q a r", positive=True)
    dedans = champ_interieur(charge, rayon, r)
    dehors = COULOMB * charge / r**2
    coquille = 4 * sp.pi * r**2 * PERMITTIVITE / 2
    par_le_champ = sp.integrate(coquille * dedans**2, (r, 0, rayon)) + sp.integrate(
        coquille * dehors**2, (r, rayon, sp.oo)
    )
    assert egales(par_le_champ, energie_de_la_sphere(charge, rayon))


def test_l_energie_volumique_recoupe_celle_que_porte_deja_la_bibliotheque() -> None:
    """(2.21.14) : w_e = E . D / 2, soit eps_0 E^2 / 2 dans le vide."""
    champ = 3.0e5
    assert proches_relativement(
        densite_d_energie_electrique(PERMITTIVITE_DU_VIDE, champ),
        champ * (PERMITTIVITE_DU_VIDE * champ) / 2,
        1e-9,
    )


def test_l_energie_du_noyau_d_uranium() -> None:
    """Exemple 2.21.2 : Z = 92, a = 9 fm -- 812,5 MeV, soit 1,3 . 10^-10 J."""
    joules = float(
        energie_de_la_sphere(92 * CHARGE_ELEMENTAIRE, 9e-15).subs(
            PERMITTIVITE, PERMITTIVITE_DU_VIDE
        )
    )
    assert proches_relativement(joules, 1.3e-10, 0.01)
    mega_electronvolts = joules / CHARGE_ELEMENTAIRE / 1e6
    assert proches(mega_electronvolts, 812.5, 0.05)
    assert eloignees(mega_electronvolts, IMPRIME_813_5, 0.9)  # valeur imprimee exclue


def test_un_systeme_a_charges_confondues_n_a_pas_d_energie_finie() -> None:
    """Remarque 1 de la page 2-77 : pour des charges ponctuelles, tout diverge."""
    with pytest.raises(ValueError, match="charges confondues"):
        energie_du_systeme(SystemeDeCharges(((1, (0, 0, 0)), (1, (0, 0, 0)))))
