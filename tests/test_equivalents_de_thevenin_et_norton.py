"""Les generateurs equivalents d'un reseau, par trois mesures independantes.

Vu de deux bornes, un reseau lineaire se reduit a deux nombres. On peut
les obtenir de trois facons, et aucune n'emploie les deux autres :

- la **tension a vide**, bornes ouvertes ;
- le **courant de court-circuit**, bornes reliees par un fil ;
- la **resistance du reseau eteint**, mesuree en y injectant un courant.

Le theoreme dit que la premiere est le produit des deux autres. Ce n'est
pas une definition mais une propriete : ce fichier la verifie sur des
reseaux quelconques, ce qui eprouve le resolveur et les trois mesures a
la fois.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.equivalents import (
    courant_de_court_circuit,
    eteindre,
    resistance_vue_des_bornes,
    tension_a_vide,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Reseau,
    Resistance,
    SourceDeCourant,
    SourceDeTension,
)

TENSION = sp.Symbol("E", positive=True)
COURANT = sp.Symbol("I", positive=True)
PREMIERE, SECONDE, TROISIEME = sp.symbols("R_1 R_2 R_3", positive=True)

SOURCE_REELLE = Reseau(
    elements=(
        SourceDeTension("interne", "B", TENSION),
        Resistance("interne", "A", PREMIERE),
    ),
    reference="B",
)
"""Une source de tension en serie avec sa resistance : le cas d'ecole."""

PONT = Reseau(
    elements=(
        SourceDeTension("haut", "B", TENSION),
        Resistance("haut", "A", PREMIERE),
        Resistance("A", "B", SECONDE),
        SourceDeCourant("B", "A", COURANT),
        Resistance("A", "milieu", TROISIEME),
        Resistance("milieu", "B", TROISIEME),
    ),
    reference="B",
)
"""Un reseau a deux sources de natures differentes, sans symetrie."""


def test_la_tension_a_vide_d_une_source_reelle_est_sa_force_electromotrice() -> None:
    """Aucun courant ne sort, donc la resistance interne ne chute rien."""
    assert sp.simplify(tension_a_vide(SOURCE_REELLE, "A", "B") - TENSION) == 0


def test_le_court_circuit_d_une_source_reelle_debite_e_sur_r() -> None:
    """Toute la force electromotrice tombe dans la resistance interne."""
    attendu = TENSION / PREMIERE
    obtenu = courant_de_court_circuit(SOURCE_REELLE, "A", "B")
    assert sp.simplify(obtenu - attendu) == 0


def test_la_resistance_du_reseau_eteint_est_la_resistance_interne() -> None:
    """La source de tension eteinte est un fil : il ne reste que `R_1`."""
    obtenue = resistance_vue_des_bornes(SOURCE_REELLE, "A", "B")
    assert sp.simplify(obtenue - PREMIERE) == 0


def test_eteindre_annule_les_sources_sans_toucher_aux_resistances() -> None:
    """Tension nulle pour l'une, courant nul pour l'autre : un fil, un vide."""
    eteint = eteindre(PONT)
    assert tension_a_vide(eteint, "A", "B") == 0
    resistances = [e for e in eteint.elements if isinstance(e, Resistance)]
    assert len(resistances) == len(
        [e for e in PONT.elements if isinstance(e, Resistance)],
    )


def test_la_tension_a_vide_est_le_produit_des_deux_autres_mesures() -> None:
    """`E_TH = R_TH . I_cc` sur un reseau quelconque : le theoreme lui-meme.

    Trois calculs distincts -- trois resolutions de trois reseaux
    differents -- et une seule relation entre eux.
    """
    a_vide = tension_a_vide(PONT, "A", "B")
    court_circuit = courant_de_court_circuit(PONT, "A", "B")
    resistance = resistance_vue_des_bornes(PONT, "A", "B")
    assert sp.simplify(a_vide - resistance * court_circuit) == 0


def test_la_resistance_du_pont_est_celle_des_trois_branches_en_parallele() -> None:
    """Oracle independant : `R_1 // R_2 // 2 R_3`, ecrit a la main."""
    inverse = 1 / PREMIERE + 1 / SECONDE + 1 / (2 * TROISIEME)
    obtenue = resistance_vue_des_bornes(PONT, "A", "B")
    assert sp.simplify(obtenue - 1 / inverse) == 0


def test_inverser_les_bornes_inverse_la_tension_et_le_courant() -> None:
    """Et laisse la resistance inchangee : elle n'a pas de sens."""
    assert (
        sp.simplify(tension_a_vide(PONT, "A", "B") + tension_a_vide(PONT, "B", "A"))
        == 0
    )
    directe = resistance_vue_des_bornes(PONT, "A", "B")
    inverse = resistance_vue_des_bornes(PONT, "B", "A")
    assert sp.simplify(directe - inverse) == 0


def test_des_bornes_confondues_sont_refusees() -> None:
    """Entre un noeud et lui-meme il n'y a ni tension ni dipole."""
    with pytest.raises(ValueError, match="bornes confondues"):
        tension_a_vide(PONT, "A", "A")
