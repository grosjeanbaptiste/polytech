"""Syllabus d'electricite generale, chapitre 5 : calcul des forces electrostatiques.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 5-1 a 5-6, lues sur les pages rendues.

Le chapitre ne chiffre qu'une valeur (40 N/m^2 dans l'air). Les deux
theoremes du calcul indirect sont donc eprouves sur le condensateur plan,
dont la force se calcule aussi directement par la pression electrostatique
du paragraphe 5.1 : trois chemins, un resultat.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMITTIVITE_DU_VIDE
from polytech.electromagnetisme.domain.electrostatique.avancee import permittivite
from polytech.electromagnetisme.domain.electrostatique.conducteurs import (
    charge_surfacique_depuis_le_champ,
    force_surfacique,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.dielectriques import (
    CHAMP_DISRUPTIF_DE_L_AIR,
)
from polytech.electromagnetisme.domain.energie_et_forces import (
    DEPLACEMENT,
    bilan_a_potentiels_constants,
    densite_d_energie_electrique,
    energie_d_un_condensateur,
    force_a_charges_constantes,
    force_a_potentiels_constants,
    force_sur_une_armature,
)

if TYPE_CHECKING:
    from collections.abc import Callable

X = DEPLACEMENT
EPS, AIRE, CHARGE, TENSION = sp.symbols("varepsilon S Q U", positive=True)
CAPACITE_PLANE = EPS * AIRE / X
"""Deux armatures d'aire S, ecartees de x."""


# --- 5.1 Calcul direct ---------------------------------------------------------


def test_la_force_surfacique_sur_un_conducteur_dans_le_vide() -> None:
    """(5.1.5) : f_S = sigma E / 2 = eps_0 E^2 / 2, E pris le long de la surface."""
    champ = sp.Symbol("E", positive=True)
    sigma = charge_surfacique_depuis_le_champ(champ)
    assert sp.simplify(force_surfacique(sigma) - PERMITTIVITE * champ**2 / 2) == 0
    assert sp.simplify(force_surfacique(sigma) - sigma * champ / 2) == 0


def test_la_pression_electrostatique_maximale_dans_l_air() -> None:
    """Page 5-2 : au champ disruptif de 3 MV/m, « quelque 40 N/m^2 soit 4 mN/cm^2 »."""
    pression = densite_d_energie_electrique(
        PERMITTIVITE_DU_VIDE, CHAMP_DISRUPTIF_DE_L_AIR
    )
    assert math.isclose(pression, 40, rel_tol=0.01)
    assert math.isclose(pression * 1e3 / 1e4, 4, rel_tol=0.01)


def test_la_pression_electrostatique_dans_un_dielectrique() -> None:
    """(5.1.6) : eps E^2 / 2 -- eps_r fois plus qu'a vide, au meme champ."""
    a_vide = densite_d_energie_electrique(PERMITTIVITE_DU_VIDE, 1e5)
    assert math.isclose(
        densite_d_energie_electrique(permittivite(4.0), 1e5), 4 * a_vide
    )


# --- 5.2 Calcul indirect -------------------------------------------------------


def test_la_force_a_charges_constantes_sur_l_armature_d_un_condensateur() -> None:
    """Theoreme 1, (5.2.11) : F = -(dW_e/dl)_Q -- les armatures s'attirent."""
    energie = CHARGE**2 / (2 * CAPACITE_PLANE)
    force = force_a_charges_constantes(energie)
    assert sp.simplify(force + CHARGE**2 / (2 * EPS * AIRE)) == 0


def test_la_force_a_potentiels_constants_sur_la_meme_armature() -> None:
    """Theoreme 2, (5.2.11) : F = +(dW_e/dl)_U."""
    force = force_a_potentiels_constants(
        energie_d_un_condensateur(CAPACITE_PLANE, TENSION)
    )
    assert sp.simplify(force + EPS * AIRE * TENSION**2 / (2 * X**2)) == 0


def test_les_deux_theoremes_donnent_la_meme_force() -> None:
    """(5.2.11) : -(dW_e/dl)_Q = +(dW_e/dl)_U, pour le meme etat Q = C U."""
    a_charges = force_a_charges_constantes(CHARGE**2 / (2 * CAPACITE_PLANE))
    a_potentiels = force_a_potentiels_constants(
        energie_d_un_condensateur(CAPACITE_PLANE, TENSION)
    )
    meme_etat = {CHARGE: CAPACITE_PLANE * TENSION}
    assert sp.simplify(a_charges.subs(meme_etat) - a_potentiels) == 0


def test_la_force_indirecte_recoupe_la_pression_electrostatique() -> None:
    """5.1 et 5.2 : sigma^2 / (2 eps) sur toute l'armature, avec sigma = Q / S."""
    indirecte = force_a_charges_constantes(CHARGE**2 / (2 * CAPACITE_PLANE))
    directe = force_surfacique(CHARGE / AIRE).subs(PERMITTIVITE, EPS) * AIRE
    assert sp.simplify(indirecte + directe) == 0


def test_les_variations_d_energie_s_opposent() -> None:
    """(5.2.5) : (dW_e)_U = -(dW_e)_Q, pour la meme deformation du meme etat."""
    a_charges = sp.diff(CHARGE**2 / (2 * CAPACITE_PLANE), X)
    a_potentiels = sp.diff(energie_d_un_condensateur(CAPACITE_PLANE, TENSION), X)
    meme_etat = {CHARGE: CAPACITE_PLANE * TENSION}
    assert sp.simplify(a_charges.subs(meme_etat) + a_potentiels) == 0


def test_les_forces_tendent_a_augmenter_la_capacite() -> None:
    """Remarque 2 de la page 5-5 : F = (U^2 / 2) dC/dl, du signe de dC/dl."""
    capacite = sp.Function("C")(X)
    force = force_a_potentiels_constants(energie_d_un_condensateur(capacite, TENSION))
    assert sp.simplify(force - TENSION**2 * sp.diff(capacite, X) / 2) == 0
    a_charges = force_a_charges_constantes(CHARGE**2 / (2 * capacite))
    assert sp.simplify(a_charges.subs(CHARGE, capacite * TENSION) - force) == 0


def test_le_couple_sur_l_armature_mobile_d_un_condensateur_variable() -> None:
    """(5.2.13) : C_alpha = +(dW_e/d alpha)_U = k U^2 / 2, si C = k alpha."""
    pente = sp.Symbol("k", positive=True)
    couple = force_a_potentiels_constants(energie_d_un_condensateur(pente * X, TENSION))
    assert couple == pente * TENSION**2 / 2


def test_le_bilan_d_energie_a_potentiels_constants() -> None:
    """(5.2.7) a (5.2.9) : le champ gagne dT, les sources perdent 2 dT."""
    travail = sp.Symbol("dT", positive=True)
    champ, sources = bilan_a_potentiels_constants(travail)
    assert (champ, sources) == (travail, -2 * travail)
    assert travail + champ + sources == 0


def test_la_force_moyenne_du_chapitre_iv_est_ce_theoreme_en_difference_finie() -> None:
    """`force_sur_une_armature` : (1/2) U^2 |Delta C| / Delta x, exacte si C affine."""
    pente, tension, pas = 3e-9, 200.0, 1e-3
    exacte = force_a_potentiels_constants(energie_d_un_condensateur(pente * X, tension))
    assert math.isclose(
        force_sur_une_armature(tension, pente * pas, pas), float(exacte)
    )


@pytest.mark.parametrize(
    "force", [force_a_charges_constantes, force_a_potentiels_constants]
)
def test_une_energie_independante_du_deplacement_ne_donne_aucune_force(
    force: Callable[[sp.Expr], sp.Expr],
) -> None:
    """Un degre de liberte qui ne change pas l'energie n'est le siege d'aucune force."""
    assert force(CHARGE * TENSION / 2) == 0
