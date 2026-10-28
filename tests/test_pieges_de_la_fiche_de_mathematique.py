"""Les conseils et pieges qualitatifs de la fiche de « Mathematique », calcules."""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.laplace.signaux import rendre_causale
from polytech.maths.domain.symboles import TEMPS as T
from polytech.maths.domain.symboles import VARIABLE as X


def test_on_prend_pour_u_le_logarithme_car_la_derivation_le_simplifie() -> None:
    bon = IntegrationParParties(sp.log(X), X**2)
    mauvais = IntegrationParParties(X**2, sp.log(X))
    assert not bon.integrale_restante().function.has(sp.log)
    assert mauvais.integrale_restante().function.has(sp.log)


def test_une_puissance_de_x_baisse_d_un_cran_par_parties() -> None:
    parties = IntegrationParParties(X**3, sp.exp(X))
    assert sp.degree(parties.derivee_de_u(), X) == 2


def test_les_fonctions_de_laplace_sont_causales() -> None:
    f = rendre_causale(sp.cos(T))
    assert f.subs(T, -1) == 0
    assert f.subs(T, 1) == sp.cos(1)
