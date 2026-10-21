"""Formulaire 9, page 13 : `y(t0) = c` donne `y = e^(A(t - t0)) c + int_t0^t ...`.

Source : `Mathematique/FORMULAIRES/Formulaire_Aide_memoire.pdf`, page 13 :
le systeme homogene `y' = A y`, `y(t0) = c`, a pour solution
`e^(A(t - t0)) c` -- le formulaire ecrit `c e^(A(t - t0))`, dans l'ordre
qu'un produit matriciel n'accepte pas -- et le systeme non homogene y
ajoute `int_t0^t e^(A(t - tau)) x(tau) dtau`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.differentielles.systemes import SystemeDifferentiel
from polytech.maths.domain.symboles import TEMPS as T

ROTATION = SystemeDifferentiel(sp.Matrix([[0, 1], [-1, 0]]))
C = sp.Matrix([1, 0])
T0 = sp.pi / 2


def test_la_solution_vaut_c_en_t0() -> None:
    y = ROTATION.par_exponentielle(C, t0=T0)
    assert sp.simplify(y.subs(T, T0) - C) == sp.zeros(2, 1)


def test_la_solution_homogene_est_une_rotation_decalee() -> None:
    y = ROTATION.par_exponentielle(C, t0=T0)
    attendue = sp.Matrix([sp.cos(T - T0), -sp.sin(T - T0)])
    assert sp.simplify(y - attendue) == sp.zeros(2, 1)


def test_la_solution_forcee_verifie_l_equation_et_la_condition() -> None:
    force = sp.Matrix([0, 1])
    y = ROTATION.par_exponentielle(C, force, t0=T0)
    assert sp.simplify(y.subs(T, T0) - C) == sp.zeros(2, 1)
    residu = sp.diff(y, T) - ROTATION.matrice * y - force
    assert sp.simplify(residu) == sp.zeros(2, 1)


def test_t0_nul_par_defaut_garde_la_formule_d_origine() -> None:
    assert ROTATION.par_exponentielle(C) == ROTATION.par_exponentielle(C, t0=0)
