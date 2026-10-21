"""Les espaces propres generalises et la taille des blocs de Jordan.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
7.4.4, pages 132 et 133 (imprimees) : `E_1 = ker(A - lambda I)`,
`E_2 = ker(A - lambda I)^2`, ... « jusqu'a ce que `E_n` ne s'accroisse
plus », quand sa dimension egale la multiplicite algebrique ; « il existe
`g_i` blocs [...] la somme des tailles de bloc est egale a `m_i` ».
L'exemple de la page 132 : `A = [[1, 0, 1], [2, -2, -1], [0, 0, 1]]`, de
polynome caracteristique `-(lambda - 1)^2 (lambda + 2)` ; page 133,
`J = diag(-2, [[1, 1], [0, 1]])`. Le formulaire (1.2, page 1) :
`1 <= mg(lambda) <= ma(lambda)`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.jordan import (
    dimensions_des_espaces_generalises,
    forme_de_jordan,
    multiplicites,
    ordre_de_la_reduction,
    tailles_des_blocs,
)

A = sp.Matrix([[1, 0, 1], [2, -2, -1], [0, 0, 1]])
"""L'exemple de la page 132."""
J = sp.Matrix([[-2, 0, 0], [0, 1, 1], [0, 0, 1]])


def test_les_espaces_generalises_de_la_valeur_double() -> None:
    assert dimensions_des_espaces_generalises(A, 1) == (1, 2)


def test_la_valeur_simple_n_a_qu_un_espace() -> None:
    assert dimensions_des_espaces_generalises(A, -2) == (1,)


def test_un_seul_bloc_de_taille_deux_pour_la_valeur_double() -> None:
    assert tailles_des_blocs(A, 1) == (2,)


def test_un_bloc_de_jordan_complet_trois() -> None:
    bloc = sp.Matrix([[5, 1, 0], [0, 5, 1], [0, 0, 5]])
    assert dimensions_des_espaces_generalises(bloc, 5) == (1, 2, 3)
    assert tailles_des_blocs(bloc, 5) == (3,)


def test_deux_blocs_de_tailles_deux_et_un() -> None:
    melange = sp.Matrix([[4, 1, 0], [0, 4, 0], [0, 0, 4]])
    assert dimensions_des_espaces_generalises(melange, 4) == (2, 3)
    assert tailles_des_blocs(melange, 4) == (2, 1)


def test_les_tailles_concordent_avec_la_forme_de_sympy() -> None:
    """Second chemin : la forme de Jordan que sympy calcule."""
    _, blocs = forme_de_jordan(A)
    assert sorted(tailles_des_blocs(A, 1) + tailles_des_blocs(A, -2)) == [1, 2]
    assert blocs.trace() == J.trace()


def test_une_valeur_qui_n_est_pas_propre_est_refusee() -> None:
    with pytest.raises(ValueError, match="pas valeur propre"):
        dimensions_des_espaces_generalises(A, 3)


def test_les_multiplicites_algebriques_font_l_ordre() -> None:
    """La somme des `ma` est l'ordre (degre du polynome), et `1 <= mg <= ma`."""
    m = multiplicites(A)
    assert sum(a for a, _ in m.values()) == A.rows
    assert all(1 <= g <= a for a, g in m.values())


def test_l_ordre_de_la_reduction_compte_les_racines_reelles() -> None:
    """Formulaire 1.2 : « sinon A est reductible jusqu'a l'ordre somme ma »."""
    rotation = sp.Matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert ordre_de_la_reduction(rotation) == 1
    assert ordre_de_la_reduction(A) == 3


def test_une_matrice_sans_valeur_propre_reelle_n_est_pas_reduite() -> None:
    assert ordre_de_la_reduction(sp.Matrix([[0, -1], [1, 0]])) == 0
