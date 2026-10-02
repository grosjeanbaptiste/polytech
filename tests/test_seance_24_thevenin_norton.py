"""La seance 2.4 : cinq exercices de Thevenin et Norton, recoupes par les noeuds.

Le manuscrit resout chaque exercice par une loi des mailles. Ce fichier
resout les memes reseaux par la loi des noeuds -- une route qui ne choisit
aucune maille -- et compare aux **reponses imprimees du recueil**
(`Electromagnetisme/EXS/Exs_Notes_manuscrites.pdf`, section 2.3.3,
exercices 2, 3, 4, 6 et 9). L'oracle est donc exterieur a la fois au
manuscrit et a la bibliotheque.

Les reseaux sont ceux des figures 2.3.16 a 2.3.24 du recueil, decrits
dans `enonces_seance_24`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.electromagnetisme.application.enonces_seance_24 import (
    reseau_charge_du_six,
    reseau_du_deux,
    reseau_du_neuf,
    reseau_du_quatre,
    reseau_du_six,
    reseau_du_trois,
)
from polytech.electromagnetisme.application.symboles_seance_24 import (
    CHARGE,
    COURANT,
    COURANT_3,
    DONNEES_DU_SIX,
    R1,
    R2,
    R3,
    R4,
    TENSION,
    TENSION_1,
    TENSION_2,
)
from polytech.electromagnetisme.domain.circuits.equivalents import (
    courant_de_court_circuit,
    resistance_vue_des_bornes,
    tension_a_vide,
)

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.circuits.reseau import Reseau


def _equivalents(reseau: Reseau) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """La tension a vide, la resistance du reseau eteint, le court-circuit."""
    return (
        tension_a_vide(reseau, "A", "B"),
        resistance_vue_des_bornes(reseau, "A", "B"),
        courant_de_court_circuit(reseau, "A", "B"),
    )


def test_exercice_2_le_courant_dans_la_charge_est_celui_du_recueil() -> None:
    """`I = (R_1 E_2 + R_2 E_1) / (R R_1 + R R_2 + R_1 R_2)`.

    Calcule de deux facons : par le generateur de Thevenin debitant sur
    `R`, et par le reseau complet ou `R` est branchee.
    """
    a_vide, resistance, _ = _equivalents(reseau_du_deux())
    par_thevenin = a_vide / (resistance + CHARGE)
    complet = reseau_du_deux(charge=True).potentiels()["A"] / CHARGE
    attendu = (R1 * TENSION_2 + R2 * TENSION_1) / (CHARGE * R1 + CHARGE * R2 + R1 * R2)
    assert sp.simplify(par_thevenin - attendu) == 0
    assert sp.simplify(complet - attendu) == 0


def test_exercice_2_la_tension_de_thevenin_est_celle_du_manuscrit() -> None:
    """`E_TH = E_2 + R_2 (E_1 - E_2) / (R_1 + R_2)`, et `R_TH = R_1 // R_2`."""
    a_vide, resistance, _ = _equivalents(reseau_du_deux())
    attendue = TENSION_2 + R2 * (TENSION_1 - TENSION_2) / (R1 + R2)
    assert sp.simplify(a_vide - attendue) == 0
    assert sp.simplify(resistance - R1 * R2 / (R1 + R2)) == 0


def test_exercice_3_les_trois_grandeurs_sont_celles_du_recueil() -> None:
    """`E_0`, `R_0` et `I_0` de la figure 2.3.17."""
    a_vide, resistance, court_circuit = _equivalents(reseau_du_trois())
    somme = R1 + R2 + R3
    assert (
        sp.simplify(a_vide - ((R2 + R3) * TENSION_1 + R1 * R3 * COURANT_3) / somme) == 0
    )
    assert sp.simplify(resistance - (R1 * R2 + R1 * R3) / somme) == 0
    attendu = (R2 + R3) / (R1 * R2 + R1 * R3) * TENSION_1 + R3 / (R2 + R3) * COURANT_3
    assert sp.simplify(court_circuit - attendu) == 0


def test_exercice_4_les_trois_grandeurs_sont_celles_du_recueil() -> None:
    """`E_0`, `R_0` et `I_0` de la figure 2.3.18, en regime."""
    a_vide, resistance, court_circuit = _equivalents(reseau_du_quatre())
    attendue = (R1 * TENSION_2 + R3 * TENSION_1 - R1 * R3 * COURANT) / (R1 + R3)
    assert sp.simplify(a_vide - attendue) == 0
    assert sp.simplify(resistance - R1 * R3 / (R1 + R3)) == 0
    assert sp.simplify(court_circuit - (TENSION_2 / R3 + TENSION_1 / R1 - COURANT)) == 0


def test_exercice_4_ni_r2_ni_r4_n_entrent_dans_le_resultat() -> None:
    """`R_2` est en serie avec un condensateur, `R_4` doublee par une inductance.

    En regime la premiere ne porte aucun courant et la seconde aucune
    tension : elles disparaissent du resultat, et le manuscrit les retire
    du schema sans le dire.
    """
    for grandeur in _equivalents(reseau_du_quatre()):
        assert grandeur.free_symbols.isdisjoint({R2, R4})


def test_exercice_6_le_generateur_est_celui_du_recueil() -> None:
    """`E_0 = -12 V` et `R_0 = 2,4 ohms`."""
    a_vide, resistance, _ = _equivalents(reseau_du_six())
    assert float(a_vide.subs(DONNEES_DU_SIX)) == pytest.approx(-12.0)
    assert float(resistance.subs(DONNEES_DU_SIX)) == pytest.approx(2.4)


def test_exercice_6_la_puissance_dans_la_charge_est_celle_du_recueil() -> None:
    """`P = 28,224 W`, par le reseau complet et non par le generateur.

    Route independante de celle du manuscrit, qui passe par Thevenin :
    ici la charge et sa source sont branchees, et la tension aux bornes
    se lit directement.
    """
    potentiels = reseau_charge_du_six().potentiels()
    tension = (potentiels["A"] - potentiels["B"]).subs(DONNEES_DU_SIX)
    puissance = tension**2 / sp.Rational(16, 10)
    assert float(puissance) == pytest.approx(28.224)


def test_exercice_6_la_branche_e5_r5_ne_change_rien() -> None:
    """Elle double une source de tension ideale, qui impose deja sa tension.

    Le manuscrit le montre d'un trait : il barre `R_5` dans le schema du
    reseau eteint. Le recueil donne pourtant `E_5` et `R_5` parmi les
    donnees.
    """
    tension_5, resistance_5 = sp.symbols("E_5 R_5", positive=True)
    for grandeur in _equivalents(reseau_du_six()):
        assert grandeur.free_symbols.isdisjoint({tension_5, resistance_5})


def test_exercice_9_diminuer_r1_ne_change_ni_i0_ni_r0() -> None:
    """Reponse du recueil : « `R_0` et `I_0` restent inchanges ».

    En regime l'inductance double la branche `R_2`-`R_1` d'un fil : aucun
    courant n'y passe, et le generateur de Norton ne contient ni l'une ni
    l'autre.
    """
    _, resistance, court_circuit = _equivalents(reseau_du_neuf())
    assert resistance.free_symbols.isdisjoint({R1, R2})
    assert court_circuit.free_symbols.isdisjoint({R1, R2})


def test_exercice_9_le_generateur_de_norton_se_lit_sur_le_schema() -> None:
    """`I_0 = I + E / R_3` et `R_0 = R_3 // R_4`."""
    _, resistance, court_circuit = _equivalents(reseau_du_neuf())
    assert sp.simplify(court_circuit - (COURANT + TENSION / R3)) == 0
    assert sp.simplify(resistance - R3 * R4 / (R3 + R4)) == 0
