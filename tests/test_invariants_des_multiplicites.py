"""Les invariants des multiplicites d'une valeur propre.

Formulaire, 1.2 : pour chaque valeur propre `1 <= mg <= ma`, la somme des
multiplicites algebriques (sur C) vaut `n`, et les tailles des blocs de
Jordan d'une valeur propre somment a `ma`. Verifie sur des matrices de
formes differentes : diagonalisable, symetrique, defective, bloc de Jordan
pur, nilpotente, a valeurs propres complexes.
"""

import pytest
import sympy as sp

from polytech.maths.domain.algebre.jordan import multiplicites, tailles_des_blocs

MATRICES = {
    "diagonalisable": sp.Matrix([[2, 0, 0], [0, -1, 0], [0, 0, 5]]),
    "symetrique": sp.Matrix([[8, -2, -2], [-2, 5, 4], [-2, 4, 5]]),
    "defective": sp.Matrix([[-1, 1, 0], [0, -1, 0], [0, 0, 3]]),
    "jordan_pur": sp.Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]]),
    "nilpotente": sp.Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
    "complexe": sp.Matrix([[0, -1], [1, 0]]),
}


@pytest.mark.parametrize("nom", MATRICES)
def test_la_geometrique_est_entre_un_et_l_algebrique(nom: str) -> None:
    for algebrique, geometrique in multiplicites(MATRICES[nom]).values():
        assert 1 <= geometrique <= algebrique


@pytest.mark.parametrize("nom", MATRICES)
def test_les_multiplicites_algebriques_somment_a_n(nom: str) -> None:
    matrice = MATRICES[nom]
    total = sum(a for a, _ in multiplicites(matrice).values())
    assert total == matrice.rows


@pytest.mark.parametrize("nom", [n for n in MATRICES if n != "complexe"])
def test_les_blocs_de_jordan_somment_a_l_algebrique(nom: str) -> None:
    matrice = MATRICES[nom]
    for valeur, (algebrique, geometrique) in multiplicites(matrice).items():
        tailles = tailles_des_blocs(matrice, valeur)
        assert sum(tailles) == algebrique
        assert len(tailles) == geometrique
