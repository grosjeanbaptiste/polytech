"""L'analyse nodale modifiee assemblee a partir d'une liste d'elements.

Oracles independants de l'assemblage : le diviseur de tension, la loi
d'Ohm sous une source de courant, l'amplificateur a transconductance --
trois circuits dont la reponse se connait sans aucune matrice.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceCommandee,
    SourceDeCourant,
    SourceDeTension,
    resoudre,
    systeme_nodal_modifie,
)

R1, R2, G, VS, IS, I1 = sp.symbols("R_1 R_2 g V_S I_S I_1", positive=True)


def _diviseur() -> list[Element]:
    return [
        SourceDeTension(1, 0, VS, I1),
        Admittance(1, 2, 1 / R1),
        Admittance(2, 0, 1 / R2),
    ]


def test_le_diviseur_de_tension_est_retrouve() -> None:
    solution = resoudre(_diviseur(), 2)
    assert sp.simplify(solution[1] - VS * R2 / (R1 + R2)) == 0


def test_le_courant_de_source_sort_par_la_borne_plus() -> None:
    """La source debite `V_S / (R_1 + R_2)` dans le circuit."""
    solution = resoudre(_diviseur(), 2)
    assert sp.simplify(solution[2] - VS / (R1 + R2)) == 0


def test_les_courants_entrants_comptent_positivement() -> None:
    """La convention du cours : une admittance pese en negatif sur sa diagonale."""
    matrice, _ = systeme_nodal_modifie(_diviseur(), 2)
    assert matrice[0, 0] == -1 / R1
    assert matrice[0, 1] == 1 / R1


def test_une_source_de_courant_impose_la_loi_d_ohm() -> None:
    circuit: list[Element] = [SourceDeCourant(0, 1, IS), Admittance(1, 0, 1 / R1)]
    assert sp.simplify(resoudre(circuit, 1)[0] - IS * R1) == 0


def test_une_source_commandee_amplifie() -> None:
    """`g V_1` tire du noeud 2 vers la masse : `V_2 = -g R_2 V_S`."""
    circuit: list[Element] = [
        SourceDeTension(1, 0, VS, I1),
        SourceCommandee(2, 0, G, 1, 0),
        Admittance(2, 0, 1 / R2),
    ]
    assert sp.simplify(resoudre(circuit, 2)[1] + G * R2 * VS) == 0


def test_un_noeud_hors_du_circuit_est_refuse() -> None:
    with pytest.raises(ValueError, match="hors du circuit"):
        systeme_nodal_modifie([Admittance(1, 3, 1 / R1)], 2)


def test_un_circuit_singulier_est_refuse() -> None:
    """Un noeud flottant n'a pas de tension definie."""
    with pytest.raises(ValueError, match="systeme nodal singulier"):
        resoudre([SourceDeCourant(0, 1, IS)], 1)


def test_une_source_de_tension_flottante_s_ajoute() -> None:
    """`V_x` empilee sur `V_S` : le noeud 2 porte leur somme."""
    vx, ix = sp.symbols("V_x I_x")
    circuit: list[Element] = [
        SourceDeTension(1, 0, VS, I1),
        SourceDeTension(2, 1, vx, ix),
        Admittance(2, 0, 1 / R2),
    ]
    assert sp.simplify(resoudre(circuit, 2)[1] - (VS + vx)) == 0


def test_une_source_commandee_lit_une_difference() -> None:
    """`g (V_1 - V_2)` : seule la chute dans `R_1` est amplifiee."""
    circuit: list[Element] = [
        SourceDeTension(1, 0, VS, I1),
        Admittance(1, 2, 1 / R1),
        Admittance(2, 0, 1 / R2),
        SourceCommandee(3, 0, G, 1, 2),
        Admittance(3, 0, 1 / R1),
    ]
    chute = VS * R1 / (R1 + R2)
    assert sp.simplify(resoudre(circuit, 3)[2] + G * R1 * chute) == 0
