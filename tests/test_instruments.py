"""La precision d'un multimetre, et sa propagation a un rapport.

La notice donne `+-(% de la lecture + % de la gamme)` : une part
proportionnelle a ce qu'on lit, une part fixe qui depend du calibre.
Une resistance tiree de `U / I` cumule, dans le pire des cas, les deux
incertitudes **relatives** -- c'est la regle d'un quotient.
"""

from __future__ import annotations

import pytest

from polytech.electromagnetisme.domain.instruments import (
    precision_d_un_quotient,
    precision_d_une_lecture,
)


def test_la_precision_somme_une_part_de_la_lecture_et_une_de_la_gamme() -> None:
    """2,2 kohms sur le calibre de 10 kohms : 0,044 + 0,05 ohm."""
    assert precision_d_une_lecture(2200.0, 10_000.0, 0.0020, 0.0005) == pytest.approx(
        0.094
    )


def test_une_lecture_hors_de_la_gamme_est_refusee() -> None:
    with pytest.raises(ValueError, match="gamme"):
        precision_d_une_lecture(12.0, 10.0, 0.0015, 0.0004)


def test_un_quotient_cumule_les_incertitudes_relatives() -> None:
    """`R = U / I` : `dR/R = dU/U + dI/I`."""
    resistance, incertitude = precision_d_un_quotient(4.0, 0.04, 2.0, 0.01)
    assert resistance == pytest.approx(2.0)
    assert incertitude == pytest.approx(2.0 * (0.01 + 0.005))


def test_un_diviseur_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="nul"):
        precision_d_un_quotient(4.0, 0.04, 0.0, 0.01)
