"""Electromagnetisme, statique 12 : le travail pour ecarter deux armatures.

Sources :

- l'enonce et ses reponses, `Electromagnetisme/EXS/Exs_Notes_manuscrites.pdf`,
  section 3.1, exercice 4, page imprimee 63 ;
- la solution manuscrite, `Electromagnetisme/EXS/Exs_02_Statique_12.pdf`,
  deux pages.

Le manuscrit procede par difference d'energies. Les tests refont le
calcul par un autre chemin -- l'integrale de la force le long du
deplacement -- et exigent que les deux s'accordent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.electromagnetisme.application import macros_statique_12 as ms
from polytech.electromagnetisme.application import travail_statique_12 as tr
from polytech.electromagnetisme.domain.energie_et_forces import (
    DEPLACEMENT,
    energie_a_charge_donnee,
    energie_d_un_condensateur,
    travail_contre_le_champ,
)
from polytech.maths.domain.comparaison import egales

if TYPE_CHECKING:
    from collections.abc import Callable

EPS, S, A, B, U, Q = (
    tr.EPSILON,
    tr.SURFACE,
    tr.DEPART,
    tr.ARRIVEE,
    tr.TENSION,
    tr.CHARGE,
)


# --- Le domaine --------------------------------------------------------------


def test_l_energie_par_la_charge_est_la_meme_que_par_la_tension() -> None:
    """`Q^2/(2C)` et `C U^2/2` sont la meme energie des que `Q = C U`."""
    capacite, tension = sp.symbols("C U", positive=True)
    par_la_charge = energie_a_charge_donnee(capacite, capacite * tension)
    assert egales(par_la_charge, energie_d_un_condensateur(capacite, tension))


def test_une_capacite_nulle_n_a_pas_d_energie_a_charge_donnee() -> None:
    with pytest.raises(ValueError, match="capacite nulle"):
        energie_a_charge_donnee(sp.Integer(0), sp.Symbol("Q"))


def test_le_travail_contre_une_force_d_attraction_est_positif() -> None:
    """Une force `-k` (vers les `x` decroissants) : ecarter de 1 a 3 coute `2k`."""
    k = sp.Symbol("k", positive=True)
    assert travail_contre_le_champ(-k, 1, 3) == 2 * k
    assert travail_contre_le_champ(-k / DEPLACEMENT**2, 1, 2) == k / 2


# --- 1. A potentiels constants ----------------------------------------------


def test_la_reponse_du_recueil_a_potentiels_constants() -> None:
    """Page 63 : `T = (eps0 S / 2) (b - a)/(a b) U^2`."""
    attendue = EPS * S / 2 * (B - A) / (A * B) * U**2
    assert egales(tr.a_potentiels_constants().travail, attendue)


def test_les_energies_du_manuscrit_a_potentiels_constants() -> None:
    """Page 1 : `W_i = C_i U^2/2`, `W_f = C_f U^2/2`, et `T = -(W_f - W_i)`."""
    cas = tr.a_potentiels_constants()
    assert egales(cas.initiale, EPS * S / A * U**2 / 2)
    assert egales(cas.finale, EPS * S / B * U**2 / 2)
    assert egales(cas.travail, -(cas.finale - cas.initiale))


def test_la_force_a_potentiels_constants_decroit_avec_l_ecartement() -> None:
    """`F = -eps0 S U^2 / (2 x^2)` : attractive, en `1/x^2`."""
    force = tr.a_potentiels_constants().force
    assert egales(force, -(EPS * S * U**2 / (2 * DEPLACEMENT**2)))


# --- 2. A charges constantes ------------------------------------------------


def test_la_reponse_du_recueil_a_charges_constantes() -> None:
    """Page 63 : `T = (eps0 S / 2) (b - a)/a^2 U^2`."""
    attendue = EPS * S / 2 * (B - A) / A**2 * U**2
    assert egales(tr.a_charges_constantes().travail, attendue)


def test_les_energies_du_manuscrit_a_charges_constantes() -> None:
    """Page 2 : `W = Q^2/(2C)`, `T = +(W_f - W_i) = (b - a) Q^2 / (2 eps0 S)`."""
    cas = tr.a_charges_constantes()
    assert egales(cas.travail, cas.finale - cas.initiale)
    en_charge = tr.travail_en_fonction_de_la_charge()
    assert egales(en_charge, (B - A) * Q**2 / (2 * EPS * S))
    assert egales(tr.charge_initiale(), EPS * S * U / A)


def test_la_force_a_charges_constantes_ne_depend_pas_de_l_ecartement() -> None:
    force = tr.a_charges_constantes().force
    assert not force.has(DEPLACEMENT)
    assert egales(force, -(EPS * S * U**2 / (2 * A**2)))


# --- Ce que le corrige ajoute ------------------------------------------------


@pytest.mark.parametrize("cas", [tr.a_potentiels_constants, tr.a_charges_constantes])
def test_le_travail_est_aussi_l_integrale_de_la_force(
    cas: Callable[[], tr.Ecartement],
) -> None:
    """Second chemin : `T = -int_a^b F dx`, sans passer par les energies."""
    resolu = cas()
    par_la_force = travail_contre_le_champ(resolu.force, A, B)
    assert egales(par_la_force, resolu.travail)


def test_les_deux_forces_coincident_au_depart() -> None:
    """En `x = a` les deux theoremes donnent la meme force : seul le chemin differe."""
    constante_u = tr.a_potentiels_constants().force.subs(DEPLACEMENT, A)
    assert egales(constante_u, tr.a_charges_constantes().force)


def test_le_rapport_des_deux_travaux() -> None:
    """A charges constantes il en coute `b/a` fois plus."""
    assert tr.rapport_des_travaux() == B / A


def test_le_bilan_de_la_source_a_potentiels_constants() -> None:
    """La source recoit `2T` : le travail de l'operateur, et autant pris au champ."""
    recu, perte_du_champ = tr.bilan_de_la_source()
    travail = tr.a_potentiels_constants().travail
    assert egales(recu, 2 * travail)
    assert egales(perte_du_champ, travail)
    charge_rendue = (EPS * S / A - EPS * S / B) * U
    assert egales(recu, U * charge_rendue)


def test_la_tension_finale_a_charges_constantes() -> None:
    """Isole, le condensateur voit sa tension monter a `U b/a`."""
    assert egales(tr.tension_finale(), U * B / A)


def test_l_ecartement_a_l_infini() -> None:
    """A potentiels constants le travail total est fini : c'est l'energie initiale."""
    fini, infini = tr.a_l_infini()
    assert egales(fini, tr.a_potentiels_constants().initiale)
    assert infini == sp.oo


def test_pour_un_petit_deplacement_les_deux_travaux_se_confondent() -> None:
    """Au premier ordre en `delta = b - a` : la force au depart, fois `delta`."""
    delta = sp.Symbol("delta", positive=True)
    attendu = EPS * S * U**2 / (2 * A**2) * delta
    for cas in (tr.a_potentiels_constants(), tr.a_charges_constantes()):
        developpe = sp.series(cas.travail.subs(B, A + delta), delta, 0, 2).removeO()
        assert egales(developpe, attendu)


# --- La publication ----------------------------------------------------------


def test_un_travail_s_ecrit_avec_l_ecart_b_moins_a() -> None:
    """`sympy` ecrirait `-(a - b)` ; le recueil ecrit `b - a`, et on le suit."""
    rendu = ms.avec_l_ecart(tr.a_potentiels_constants().travail)
    assert rendu == r"\frac{S U^{2} \varepsilon_{0}}{2 a b}\,\left(b - a\right)"


def test_une_expression_sans_l_ecart_est_refusee() -> None:
    with pytest.raises(ValueError, match="b - a"):
        ms.avec_l_ecart(tr.capacite())


def test_les_macros_publient_les_deux_reponses_du_recueil() -> None:
    macros = ms.macros_statique_douze()
    assert all(nom.startswith("StatiqueDouze") for nom in macros)
    assert macros["StatiqueDouzeRapport"] == r"\frac{b}{a}"
    assert macros["StatiqueDouzeChargesTravail"].endswith(r"\left(b - a\right)")
