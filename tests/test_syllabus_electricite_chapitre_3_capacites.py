"""Syllabus d'electricite generale, chapitre 3 : coefficients et condensateurs.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphes 3.9 a 3.21, pages 3-22 a 3-35, lues sur les pages rendues.

Le syllabus etablit les trois especes d'equations sans les chiffrer. Les
oracles viennent donc d'un systeme dont le chapitre 2 donne les potentiels
exactement : des coquilles spheriques concentriques, de rayons `r_1 < r_2
< ...`. Le potentiel de la coquille `i` du a la charge de la coquille `k`
vaut `1 / (4 pi eps_0 max(r_i, r_k))` (exemple 2.6.3) : dedans, il est
constant ; dehors, c'est celui d'une charge ponctuelle.

La relation (3.12.7) y devient une egalite : la coquille interieure est
entierement entouree, toutes ses lignes de champ finissent sur la suivante
(« influence totale »), et sa capacite partielle vers l'infini est nulle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.sinusoidaux import (
    en_serie,
    impedance_capacitive,
)
from polytech.electromagnetisme.domain.constantes import PERMITTIVITE_DU_VIDE
from polytech.electromagnetisme.domain.electrostatique.coefficients import (
    SystemeDeConducteurs,
    energie_par_les_capacites_partielles,
    residu_de_l_identite_de_gauss,
)
from polytech.electromagnetisme.domain.electrostatique.condensateurs import (
    capacite_en_parallele,
    capacite_en_serie,
    capacite_equivalente,
    charge_d_un_condensateur,
    charges_en_parallele,
    tensions_en_serie,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.energie import (
    energie_de_la_sphere,
)
from polytech.electromagnetisme.domain.electrostatique.potentiel import COULOMB
from polytech.electromagnetisme.domain.energie_et_forces import (
    energie_d_un_condensateur,
)

if TYPE_CHECKING:
    from collections.abc import Callable

QUATRE_PI_EPS = 4 * sp.pi * PERMITTIVITE


def _coquilles(*rayons: object) -> SystemeDeConducteurs:
    return SystemeDeConducteurs(
        sp.Matrix(
            len(rayons),
            len(rayons),
            lambda i, k: COULOMB / sp.Max(rayons[i], rayons[k]),
        )
    )


TROIS = _coquilles(1, 2, 4)
A, B = sp.symbols("a b", positive=True)


# --- 3.11 Equations de premiere espece -----------------------------------------


def test_les_potentiels_sont_lineaires_en_les_charges() -> None:
    """(3.11.2) : U_i0 = somme des K_ik Q_k."""
    potentiels = TROIS.potentiels([4, 0, 0])
    assert list(potentiels) == [4 * COULOMB, 2 * COULOMB, COULOMB]
    somme = TROIS.potentiels([4, 2, -1])
    separes = TROIS.potentiels([4, 0, 0]) + TROIS.potentiels([0, 2, -1])
    assert sp.simplify(somme - separes) == sp.zeros(3, 1)


def test_les_coefficients_de_potentiel_sont_symetriques() -> None:
    """(3.11.6) : K_ij = K_ji."""
    assert TROIS.potentiel == TROIS.potentiel.T


def test_un_conducteur_charge_est_au_potentiel_le_plus_haut() -> None:
    """(3.11.7) : K_11 > K_12 > 0."""
    k = TROIS.potentiel / COULOMB
    assert k[0, 0] > k[0, 1] > 0
    assert k[0, 1] > k[0, 2] > 0


def test_une_matrice_non_symetrique_n_est_pas_un_systeme_de_conducteurs() -> None:
    """L'identite de Gauss impose la symetrie : sans elle, (3.9.1) tombe."""
    with pytest.raises(ValueError, match="coefficients de potentiel non symetriques"):
        SystemeDeConducteurs(sp.Matrix([[2, 1], [0, 1]]))
    with pytest.raises(ValueError, match="matrice carree attendue"):
        SystemeDeConducteurs(sp.Matrix([[2, 1, 0], [1, 1, 0]]))


# --- 3.9 Identite de Gauss -----------------------------------------------------


def test_l_identite_de_gauss_entre_deux_etats() -> None:
    """(3.9.1) : somme des Q'_k U_k0 = somme des Q_k U'_k0."""
    premier, second = [4, 2, -1], [0, -3, 5]
    residu = residu_de_l_identite_de_gauss(
        premier, TROIS.potentiels(premier), second, TROIS.potentiels(second)
    )
    assert sp.simplify(residu) == 0


def test_l_identite_de_gauss_tombe_sans_la_symetrie() -> None:
    """Des « potentiels » tires d'une matrice quelconque ne la verifient pas."""
    bancale = sp.Matrix([[2, 1], [0, 1]])
    premier, second = sp.Matrix([1, 0]), sp.Matrix([0, 1])
    residu = residu_de_l_identite_de_gauss(
        premier, bancale * premier, second, bancale * second
    )
    assert residu != 0


# --- 3.12 Equations de deuxieme espece -----------------------------------------


def test_les_capacites_inversent_les_coefficients_de_potentiel() -> None:
    """(3.12.8) : Gamma = K^-1, et Q = Gamma U (3.12.4)."""
    charges = sp.Matrix([4, 2, -1])
    retrouvees = TROIS.charges(TROIS.potentiels(charges))
    assert sp.simplify(retrouvees - charges) == sp.zeros(3, 1)
    assert sp.simplify(TROIS.capacites() * TROIS.potentiel) == sp.eye(3)


def test_la_capacite_du_condensateur_spherique() -> None:
    """Gamma_11 = 4 pi eps_0 a b / (b - a), et Gamma_12 = -Gamma_11."""
    gamma = _coquilles(A, A + B).capacites()
    spherique = QUATRE_PI_EPS * A * (A + B) / B
    assert sp.simplify(gamma[0, 0] - spherique) == 0
    assert sp.simplify(gamma[0, 1] + spherique) == 0
    assert gamma == gamma.T


def test_les_signes_des_capacites_et_des_coefficients_d_influence() -> None:
    """(3.12.7) : Gamma_11 > 0, Gamma_12 et Gamma_13 negatifs ou nuls."""
    gamma = TROIS.capacites() / QUATRE_PI_EPS
    assert gamma[0, 0] > 0
    assert gamma[0, 1] < 0
    assert gamma[0, 2] == 0
    assert gamma[0, 0] >= -(gamma[0, 1] + gamma[0, 2])


# --- 3.13 Equations de troisieme espece ----------------------------------------


def test_les_capacites_partielles_de_trois_coquilles() -> None:
    """(3.13.3) : C_i0 = somme des Gamma_ik, C_ij = -Gamma_ij."""
    partielles = {
        couple: sp.simplify(c / QUATRE_PI_EPS)
        for couple, c in TROIS.capacites_partielles().items()
    }
    assert partielles == {
        (1, 0): 0,
        (2, 0): 0,
        (3, 0): 4,
        (1, 2): 2,
        (1, 3): 0,
        (2, 3): 4,
    }


def test_la_derniere_coquille_voit_l_infini_comme_une_sphere_isolee() -> None:
    """C_20 = 4 pi eps_0 b : la capacite d'une sphere de rayon b seule dans l'espace."""
    partielles = _coquilles(A, A + B).capacites_partielles()
    assert sp.simplify(partielles[(2, 0)] - QUATRE_PI_EPS * (A + B)) == 0
    assert sp.simplify(partielles[(1, 0)]) == 0


def test_les_charges_par_les_capacites_partielles() -> None:
    """(3.13.2) : Q_i = somme des C_ij U_ij, j parcourant l'enveloppe et les autres."""
    u = TROIS.potentiels([4, 2, -1])
    c = TROIS.capacites_partielles()
    premiere = c[(1, 0)] * u[0] + c[(1, 2)] * (u[0] - u[1]) + c[(1, 3)] * (u[0] - u[2])
    seconde = c[(2, 0)] * u[1] + c[(1, 2)] * (u[1] - u[0]) + c[(2, 3)] * (u[1] - u[2])
    assert (sp.simplify(premiere), sp.simplify(seconde)) == (4, 2)


# --- 3.10 Energie potentielle --------------------------------------------------


def test_l_energie_est_la_demi_somme_des_charges_par_les_potentiels() -> None:
    """(3.10.4) : W_e = (1/2) somme des Q_k U_k0."""
    charges = sp.Matrix([4, 2, -1])
    potentiels = TROIS.potentiels(charges)
    attendu = sum(q * u for q, u in zip(charges, potentiels, strict=True)) / 2
    assert sp.simplify(TROIS.energie(charges) - attendu) == 0


def test_l_energie_par_les_capacites_partielles() -> None:
    """(3.13.7) et (3.13.8) : (1/2) (somme C_k0 U_k0^2 + somme C_ij U_ij^2)."""
    charges = sp.Matrix([4, 2, -1])
    par_les_capacites = energie_par_les_capacites_partielles(
        TROIS.capacites_partielles(), TROIS.potentiels(charges)
    )
    assert sp.simplify(par_les_capacites - TROIS.energie(charges)) == 0


def test_l_energie_du_condensateur_spherique_par_le_champ() -> None:
    """(3.10.5) : (1/2) integrale de E . D entre les armatures, pour +Q et -Q."""
    charge, r = sp.symbols("Q r", positive=True)
    systeme = _coquilles(A, A + B)
    champ = COULOMB * charge / r**2
    par_le_champ = sp.integrate(
        PERMITTIVITE * champ**2 / 2 * 4 * sp.pi * r**2, (r, A, A + B)
    )
    assert sp.simplify(systeme.energie([charge, -charge]) - par_le_champ) == 0


def test_l_energie_est_positive_meme_pour_des_charges_opposees() -> None:
    """Remarque 2 de la page 2-77 : l'energie du champ, elle, ne change pas de signe."""
    assert sp.simplify(TROIS.energie([1, -1, 0]) / COULOMB) > 0
    assert TROIS.energie([0, 0, 0]) == 0


# --- 3.16 a 3.19 Condensateur, associations ------------------------------------


def test_la_charge_d_un_condensateur() -> None:
    """(3.16.2) : Q = C U."""
    assert charge_d_un_condensateur(2e-6, 12) == 24e-6


def test_les_trois_ecritures_de_l_energie_d_un_condensateur() -> None:
    """(3.16.3) : W_e = QU/2 = CU^2/2 = Q^2/(2C)."""
    capacite, tension = sp.symbols("C U", positive=True)
    charge = charge_d_un_condensateur(capacite, tension)
    energie = energie_d_un_condensateur(capacite, tension)
    assert energie == charge * tension / 2
    assert sp.simplify(energie - charge**2 / (2 * capacite)) == 0


def test_la_sphere_isolee_a_son_autre_armature_a_l_infini() -> None:
    """C = 4 pi eps_0 a ; son energie n'est pas celle de la sphere chargee en volume."""
    charge = sp.Symbol("Q", positive=True)
    isolee = _coquilles(A)
    capacite = isolee.capacites()[0, 0]
    assert capacite == QUATRE_PI_EPS * A
    conductrice = sp.simplify(isolee.energie([charge]))
    assert sp.simplify(conductrice - charge**2 / (2 * capacite)) == 0
    en_volume = energie_de_la_sphere(charge, A)
    assert sp.simplify(en_volume / conductrice) == sp.Rational(6, 5)


def test_la_constante_electrique_en_picofarads_par_metre() -> None:
    """Page 3-32 : environ 8,854 . 10^-12 F/m, soit 8,854 pF/m."""
    assert math.isclose(PERMITTIVITE_DU_VIDE / 1e-12, 8.854, rel_tol=1e-4)


def test_trois_condensateurs_en_serie() -> None:
    """(3.18.2) : 1/C = 1/C_1 + 1/C_2 + 1/C_3."""
    assert capacite_en_serie([1, 2, 4]) == sp.Rational(4, 7)
    assert capacite_en_serie([6]) == 6


def test_la_tension_se_repartit_en_raison_inverse_des_capacites() -> None:
    """Page 3-33 : U_k = Q / C_k, la meme charge Q traversant toute la chaine."""
    tensions = tensions_en_serie(14, [1, 2, 4])
    assert tensions == [8, 4, 2]
    assert sum(tensions) == 14


def test_trois_condensateurs_en_parallele() -> None:
    """(3.19.2) : C = C_1 + C_2 + C_3."""
    assert capacite_en_parallele([1, 2, 4]) == 7


def test_la_charge_se_repartit_en_raison_directe_des_capacites() -> None:
    """Page 3-33 : Q_k = C_k U, la meme tension U sur chaque condensateur."""
    charges = charges_en_parallele(14, [1, 2, 4])
    assert charges == [2, 4, 8]
    assert sum(charges) == 14


@pytest.mark.parametrize("association", [capacite_en_serie, capacite_en_parallele])
def test_une_association_vide_ou_a_capacite_nulle_est_refusee(
    association: Callable[[list[float]], sp.Expr],
) -> None:
    """Un condensateur a une capacite strictement positive."""
    with pytest.raises(ValueError, match="aucun condensateur"):
        association([])
    with pytest.raises(ValueError, match="capacite strictement positive attendue"):
        association([1, 0])


def test_la_serie_recoupe_l_association_des_impedances() -> None:
    """Les impedances 1/(j omega C) s'ajoutent : c'est (3.18.2) vue du chapitre 16."""
    c_1, c_2 = sp.symbols("C_1 C_2", positive=True)
    somme = en_serie(impedance_capacitive(c_1), impedance_capacitive(c_2))
    equivalente = impedance_capacitive(capacite_en_serie([c_1, c_2]))
    assert sp.simplify(somme - equivalente) == 0


# --- 3.20 Autres modes d'association -------------------------------------------


def test_la_serie_et_le_parallele_par_les_noeuds() -> None:
    """Le reseau general redonne (3.18.2) et (3.19.2)."""
    chaine = [("1", "a", 1), ("a", "b", 2), ("b", "2", 4)]
    assert capacite_equivalente(chaine, "1", "2") == capacite_en_serie([1, 2, 4])
    faisceau = [("1", "2", 1), ("1", "2", 2), ("1", "2", 4)]
    assert capacite_equivalente(faisceau, "1", "2") == capacite_en_parallele([1, 2, 4])


def test_le_groupement_de_la_figure_3_20_1_a() -> None:
    """Deux condensateurs en parallele, en serie avec un troisieme."""
    reseau = [("1", "m", 1), ("1", "m", 3), ("m", "2", 4)]
    attendu = capacite_en_serie([capacite_en_parallele([1, 3]), 4])
    assert capacite_equivalente(reseau, "1", "2") == attendu == 2


def test_le_pont_de_la_figure_3_20_1_b() -> None:
    """Page 3-34 : charges nulles aux noeuds 3 et 4, maille fermee -- C = 7/5."""
    pont = [("1", "3", 1), ("3", "2", 2), ("1", "4", 2), ("4", "2", 1), ("3", "4", 1)]
    assert capacite_equivalente(pont, "1", "2") == sp.Rational(7, 5)
    sans_la_diagonale = capacite_en_parallele([capacite_en_serie([1, 2])] * 2)
    assert sans_la_diagonale == sp.Rational(4, 3)


def test_un_pont_equilibre_ignore_sa_diagonale() -> None:
    """C_13 / C_32 = C_14 / C_42 : les noeuds 3 et 4 sont au meme potentiel."""
    diagonale = sp.Symbol("C_d", positive=True)
    pont = [
        ("1", "3", 1),
        ("3", "2", 2),
        ("1", "4", 3),
        ("4", "2", 6),
        ("3", "4", diagonale),
    ]
    attendu = capacite_en_serie([1, 2]) + capacite_en_serie([3, 6])
    assert sp.simplify(capacite_equivalente(pont, "1", "2") - attendu) == 0


def test_des_bornes_confondues_ou_isolees_sont_refusees() -> None:
    """La capacite se mesure entre deux bornes distinctes et reliees."""
    with pytest.raises(ValueError, match="deux bornes distinctes attendues"):
        capacite_equivalente([("1", "2", 1)], "1", "1")
    with pytest.raises(ValueError, match="borne absente du reseau"):
        capacite_equivalente([("1", "2", 1)], "1", "9")
    with pytest.raises(ValueError, match="noeud sans liaison vers les bornes"):
        capacite_equivalente([("1", "2", 1), ("x", "y", 3)], "1", "2")
