"""La mise sous forme de Bode d'une transmittance : gain, classe, cassures.

Planches « Analyse harmonique », « Courbes de Bode - trace asymptotique » :
on part de la forme factorisee. Les oracles sont les cinq transmittances
de la feuille `SignauxSystemes/EXS/Exs_06_Analyse_harmonique.pdf`,
exercice 1, factorisees a la main.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.trace_asymptotique import trace_d_une_transmittance


def test_un_systeme_sans_integrateur() -> None:
    """`10/((1 + p/3)(1 + p/2))` : gain 10, classe 0, deux cassures descendantes."""
    trace = trace_d_une_transmittance(10 / ((1 + P / 3) * (1 + P / 2)))
    assert (trace.gain, trace.classe) == (10, 0)
    assert trace.cassures == ((2.0, -1), (3.0, -1))


def test_un_integrateur_et_un_zero() -> None:
    """`25 (p + 1)/(p (p + 5)) = 5 (1 + p)/(p (1 + p/5))`."""
    trace = trace_d_une_transmittance(25 * (P + 1) / (P * (P + 5)))
    assert (trace.gain, trace.classe) == (5, 1)
    assert trace.cassures == ((1.0, 1), (5.0, -1))
    assert trace.pentes() == (-20, 0, -20)


def test_un_derivateur_a_une_classe_negative() -> None:
    """`16 p/((p + 1)(p + 2)) = 8 p/((1 + p)(1 + p/2))` : la pente part a +20."""
    trace = trace_d_une_transmittance(16 * P / ((P + 1) * (P + 2)))
    assert (trace.gain, trace.classe) == (8, -1)
    assert trace.pentes() == (20, 0, -20)


def test_une_paire_complexe_casse_deux_fois_a_sa_pulsation_propre() -> None:
    """`10/(p^2 + 0,8 p + 1)` : deux poles de module 1, donc -40 dB/decade."""
    trace = trace_d_une_transmittance(10 / (P**2 + sp.Rational(4, 5) * P + 1))
    assert trace.gain == 10
    assert [pulsation for pulsation, _ in trace.cassures] == pytest.approx([1.0, 1.0])
    assert [saut for _, saut in trace.cassures] == [-1, -1]
    assert trace.pentes()[-1] == -40


def test_un_gain_negatif_est_refuse() -> None:
    """Le trace porte un module : un gain negatif n'a pas de logarithme."""
    with pytest.raises(ValueError, match="gain de Bode"):
        trace_d_une_transmittance(-3 / (P + 1))
