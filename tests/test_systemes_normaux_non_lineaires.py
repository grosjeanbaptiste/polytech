"""Syllabus de mathematiques appliquees 6.1 et 6.5 : le cas non lineaire.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`
(pages imprimees) :
- 6.1, theoreme 6.1 (page 102) : `y' = f(t, y)`, `y(t0) = y0`, pour une
  `f` **quelconque**, continue et lipschitzienne ; la transformation
  `T(y)(t) = y0 + int_t0^t f(tau, y(tau)) dtau`, iteree depuis `y0(t) = 0`
  comme dans l'exemple 6.1 (page 103) ;
- 6.5 (page 117) : `y^(n) = phi(t, y, ..., y^(n-1))` devient, par
  `y_1 = y, ..., y_n = y^(n-1)`, le systeme
  `y_1' = y_2, ..., y_(n-1)' = y_n, y_n' = phi(t, y_1, ..., y_n)` -- « mise
  sous forme normale ».

L'exemple 6.1 est lineaire : `systemes_normaux.iterees_vectorielles` le
porte, et sert ici de second chemin.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.systemes_non_lineaires import (
    INCONNUES,
    iterees_de_picard,
    sous_forme_normale,
)
from polytech.maths.domain.differentielles.systemes_normaux import (
    iterees_vectorielles,
)
from polytech.maths.domain.symboles import TEMPS as T

Y1, Y2, Y3 = INCONNUES[:3]


def test_la_forme_normale_d_une_equation_du_second_ordre() -> None:
    """`y'' = -y` : `(y_2, -y_1)`."""
    assert sous_forme_normale(-Y1, 2) == sp.Matrix([Y2, -Y1])


def test_la_forme_normale_d_ordre_trois_non_lineaire() -> None:
    phi = T * Y1 * Y3 - Y2**2
    assert sous_forme_normale(phi, 3) == sp.Matrix([Y2, Y3, phi])


def test_une_forme_normale_d_ordre_nul_est_refusee() -> None:
    with pytest.raises(ValueError, match="hors de"):
        sous_forme_normale(Y1, 0)


def test_un_ordre_au_dela_des_inconnues_est_refuse() -> None:
    with pytest.raises(ValueError, match="hors de"):
        sous_forme_normale(Y1, len(INCONNUES) + 1)


def test_le_plus_grand_ordre_est_accepte() -> None:
    assert sous_forme_normale(Y1, len(INCONNUES))[-1] == Y1


def test_phi_ne_peut_citer_une_derivee_d_ordre_n() -> None:
    """`y'' = phi(t, y, y')` : `y_3` n'existe pas a l'ordre deux."""
    with pytest.raises(ValueError, match="y3"):
        sous_forme_normale(Y3, 2)


def test_les_iterees_de_l_exemple_6_1_par_le_cas_general() -> None:
    """Second chemin : les iterees du cas lineaire."""
    generales = iterees_de_picard(sp.Matrix([Y2, Y1]), sp.Matrix([1, 0]), nombre=4)
    lineaires = iterees_vectorielles(
        sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([1, 0]), nombre=4
    )
    assert generales == lineaires


def test_les_iterees_de_y_prime_egal_y_carre() -> None:
    """`y' = y^2`, `y(0) = 1` : `0`, `1`, `1 + t`, `1 + t + t^2 + t^3/3`."""
    iterees = iterees_de_picard(sp.Matrix([Y1**2]), sp.Matrix([1]), nombre=3)
    assert iterees == (
        sp.Matrix([0]),
        sp.Matrix([1]),
        sp.Matrix([1 + T]),
        sp.Matrix([1 + T + T**2 + T**3 / 3]),
    )


def test_les_iterees_approchent_la_solution_exacte() -> None:
    """`1 / (1 - t)` : la quatrieme iteree a son developpement jusqu'a `t^3`."""
    derniere = iterees_de_picard(sp.Matrix([Y1**2]), sp.Matrix([1]), nombre=4)[-1]
    exacte = sp.series(1 / (1 - T), T, 0, 4).removeO()
    assert sp.expand(derniere[0] - exacte).as_poly(T).degree() >= 4


def test_un_second_membre_qui_depend_du_temps_et_un_instant_initial() -> None:
    """`y' = t y`, `y(1) = 2` : la deuxieme iteree vaut `2 + (t^2 - 1)`."""
    iterees = iterees_de_picard(sp.Matrix([T * Y1]), sp.Matrix([2]), nombre=2, t0=1)
    assert sp.expand(iterees[2][0] - (2 + T**2 - 1)) == 0


def test_des_conditions_de_mauvaise_taille_sont_refusees() -> None:
    with pytest.raises(ValueError, match="conditions"):
        iterees_de_picard(sp.Matrix([Y2, Y1]), sp.Matrix([1]), nombre=1)
