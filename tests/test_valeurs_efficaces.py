"""La valeur efficace d'un signal periodique, par sa definition integrale.

`X = racine((1/T) int_0^T x^2 dt)`. L'enonce du labo I la donne, puis la
specialise au sinus : `x_m / racine 2`. Les deux autres formes du labo,
le carre et le triangle, ne s'en deduisent pas -- c'est le piege, et la
raison d'integrer plutot que de diviser par racine de deux.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.valeurs_efficaces import (
    carre,
    sinus,
    triangle,
    valeur_efficace,
)

A, T = sp.symbols("A T", positive=True)


def test_un_sinus_vaut_son_amplitude_sur_racine_de_deux() -> None:
    assert sp.simplify(valeur_efficace(sinus(A, T), T) - A / sp.sqrt(2)) == 0


def test_un_carre_symetrique_vaut_son_amplitude() -> None:
    assert sp.simplify(valeur_efficace(carre(A, T), T) - A) == 0


def test_un_triangle_symetrique_vaut_son_amplitude_sur_racine_de_trois() -> None:
    assert sp.simplify(valeur_efficace(triangle(A, T), T) - A / sp.sqrt(3)) == 0


def test_les_trois_formes_ont_la_meme_crete() -> None:
    """La comparaison n'a de sens qu'a crete egale : un volt pour les trois."""
    instants = (0, T / 4, T / 2, 3 * T / 4)
    for forme in (sinus(A, T), carre(A, T), triangle(A, T)):
        cretes = [abs(forme.subs(sp.Symbol("t", real=True), t)) for t in instants]
        assert sp.simplify(max(cretes, key=sp.default_sort_key) - A) == 0


def test_une_periode_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="periode"):
        valeur_efficace(sinus(A, T), 0)
