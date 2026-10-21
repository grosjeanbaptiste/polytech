"""L'integration par parties repetee : jusqu'a epuisement, ou en boucle.

Sources :
- `Mathematique/SLIDES/Slides_01_Algebre_lineaire_et_integrales.pdf`,
  planche 42/42 (page 44 du PDF) :
  `int x^3 e^(2x) = 1/2 x^3 e^(2x) - 3/4 x^2 e^(2x) + 3/4 x e^(2x) - 3/8 e^(2x)` ;
- `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, page 74 :
  `int (x + 1)^3 ch x = (x + 1)^3 sh x - 3(x + 1)^2 ch x + 6(x + 1) sh x
  - 6 ch x` en trois parties, la derniere integrale `6 int ch x` se
  primitivant a vue ; pages 74 et 75 : `int 2^x sin 3x` « retombe
  sur la primitive demandee » apres deux parties, et vaut
  `9 / (9 + ln^2 2) (-1/3 2^x cos 3x + ln 2 / 9 2^x sin 3x)`.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
import sympy as sp

from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.integration.parties_repetees import (
    PartiesRepetees,
    jusqu_au_bout,
)
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
E2X = sp.exp(2 * X)


def _egales(a: sp.Expr, b: sp.Expr) -> bool:
    return bool(sp.simplify(sp.expand(a - b)) == 0)


def test_la_planche_42_en_trois_parties() -> None:
    parties = jusqu_au_bout(X**3, E2X)
    attendue = (
        R(1, 2) * X**3 * E2X - R(3, 4) * X**2 * E2X + R(3, 4) * X * E2X - R(3, 8) * E2X
    )
    assert len(parties.etapes) == 3
    assert _egales(parties.primitive, attendue)


def test_chaque_etape_derive_de_la_precedente() -> None:
    etapes = jusqu_au_bout(X**3, E2X).etapes
    assert etapes[0] == IntegrationParParties(X**3, E2X)
    for avant, apres in pairwise(etapes):
        assert apres.u == avant.derivee_de_u()
        assert apres.derivee_de_v == avant.v()


def test_l_exemple_42_des_notes() -> None:
    parties = jusqu_au_bout((X + 1) ** 3, sp.cosh(X))
    assert len(parties.etapes) == 3
    attendue = (
        (X + 1) ** 3 * sp.sinh(X)
        - 3 * (X + 1) ** 2 * sp.cosh(X)
        + 6 * (X + 1) * sp.sinh(X)
        - 6 * sp.cosh(X)
    )
    assert _egales(parties.primitive, attendue)


def test_l_exemple_55_des_notes_retombe_sur_lui_meme() -> None:
    parties = jusqu_au_bout(2**X, sp.sin(3 * X))
    ln2 = sp.log(2)
    attendue = (
        9
        / (9 + ln2**2)
        * (-R(1, 3) * 2**X * sp.cos(3 * X) + ln2 / 9 * 2**X * sp.sin(3 * X))
    )
    assert len(parties.etapes) == 2
    assert parties.en_boucle
    assert _egales(parties.primitive, attendue)


def test_un_polynome_ne_boucle_pas() -> None:
    parties = jusqu_au_bout(X**3, E2X)
    assert isinstance(parties, PartiesRepetees)
    assert not parties.en_boucle


def test_la_primitive_se_derive_en_l_integrande() -> None:
    """Second chemin : deriver."""
    for u, dv in ((X**3, E2X), (2**X, sp.sin(3 * X)), (X**2, sp.cos(X))):
        primitive = jusqu_au_bout(u, dv).primitive
        assert _egales(sp.diff(primitive, X), u * dv)


def test_une_iteration_sans_fin_est_refusee() -> None:
    """`u = ln x`, `dv = dx` : `1/x`, puis `-1/x^2`... jamais nul."""
    with pytest.raises(ValueError, match="ne s'acheve pas"):
        jusqu_au_bout(sp.log(X), sp.Integer(1), maximum=5)


def test_une_seule_partie_quand_u_est_du_premier_degre() -> None:
    parties = jusqu_au_bout(X, sp.exp(X))
    assert len(parties.etapes) == 1
    assert _egales(parties.primitive, (X - 1) * sp.exp(X))


def test_cinq_parties_tiennent_dans_un_maximum_de_cinq() -> None:
    parties = jusqu_au_bout(X**5, sp.exp(X), maximum=5)
    assert len(parties.etapes) == 5
    assert _egales(sp.diff(parties.primitive, X), X**5 * sp.exp(X))


def test_cinq_parties_depassent_un_maximum_de_quatre() -> None:
    with pytest.raises(ValueError, match="ne s'acheve pas en 4"):
        jusqu_au_bout(X**5, sp.exp(X), maximum=4)
