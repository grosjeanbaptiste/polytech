"""La regle de L'Hopital, telle que la planche 40 la mene.

Source : `Mathematique/SLIDES/Slides_01_Algebre_lineaire_et_integrales.pdf`,
planche 40/42 (page 42 du PDF) : `int_0^1 ln x dx` est impropre en 0 ;
`lim (delta ln delta)` s'ecrit `lim ln delta / (1/delta)`, forme `oo/oo`,
puis `lim (1/delta) / (-1/delta^2)`, et l'integrale vaut `-1`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.hopital import par_l_hopital

D = sp.Symbol("delta", positive=True)


def test_la_planche_40_en_une_application() -> None:
    etapes, limite = par_l_hopital(sp.log(D), 1 / D, D, 0)
    assert limite == 0
    assert sp.simplify(etapes[1] - (1 / D) / (-1 / D**2)) == 0


def test_la_limite_concorde_avec_sympy() -> None:
    """Second chemin : `sp.limit` du produit de depart."""
    _, limite = par_l_hopital(sp.log(D), 1 / D, D, 0)
    assert limite == sp.limit(D * sp.log(D), D, 0, "+")


def test_deux_applications_pour_x_carre_sur_e_x() -> None:
    x = sp.Symbol("x", positive=True)
    etapes, limite = par_l_hopital(x**2, sp.exp(x), x, sp.oo)
    assert limite == 0
    assert len(etapes) == 3


def test_zero_sur_zero() -> None:
    x = sp.Symbol("x", real=True)
    _, limite = par_l_hopital(sp.sin(x), x, x, 0)
    assert limite == 1


def test_une_forme_determinee_est_refusee() -> None:
    x = sp.Symbol("x", real=True)
    with pytest.raises(ValueError, match="pas indeterminee"):
        par_l_hopital(x + 1, x + 2, x, 0)


def test_une_regle_qui_ne_conclut_pas_est_refusee() -> None:
    """`sqrt(x^2 + 1) / x` en `oo` redonne son inverse, puis lui-meme."""
    x = sp.Symbol("x", positive=True)
    with pytest.raises(ValueError, match="ne conclut pas"):
        par_l_hopital(sp.sqrt(x**2 + 1), x, x, sp.oo, maximum=3)
