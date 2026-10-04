"""Macros du TP 3 d'electronique : transferts, reponses, quadripoles.

Des expressions composees en LaTeX, plus les deux nombres du filtre :
son gain en continu et sa pulsation de coupure.

Le prefixe est `TpTrois`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.application import laplace_tp_03 as lp
from polytech.electronique.application import quadripoles_tp_03 as qp
from polytech.electronique.application.formatage import nombre
from polytech.electronique.application.symboles_tp_03 import C, L, R, T


def _latex(expression: sp.Expr) -> str:
    return str(sp.latex(expression, imaginary_unit="j"))


def _macros_de_laplace() -> dict[str, str]:
    reponse = lp.reponse_au_cosinus_du_2_1()
    regime = lp.regime_du_2_1()
    transitoire = sp.simplify((reponse - regime) * sp.exp(T / (C * R)))
    alpha, beta = lp.alpha_beta()
    return {
        "TpTroisHUn": _latex(lp.transfert_du_2_1()),
        "TpTroisIndicielleUn": _latex(lp.indicielle_du_2_1()),
        "TpTroisTransitoireUn": _latex(sp.simplify(transitoire)),
        "TpTroisRegimeUn": _latex(
            sp.simplify(reponse - transitoire * sp.exp(-T / (C * R)))
        ),
        "TpTroisHDeux": _latex(lp.transfert_du_2_2()),
        "TpTroisAlpha": _latex(alpha),
        "TpTroisBeta": _latex(beta),
        "TpTroisBetaComplexe": _latex(sp.sqrt(4 * L / (C * R**2) - 1)),
        "TpTroisResonance": _latex(lp.resonance_sans_resistance()),
    }


def _macros_des_quadripoles() -> dict[str, str]:
    _, origine, coupure = qp.filtre_numerique()
    return {
        "TpTroisMatrice": _latex(qp.matrice_de_la_cascade()),
        "TpTroisHCascade": _latex(sp.factor(qp.transfert_de_la_cascade())),
        "TpTroisOrigine": _latex(origine),
        "TpTroisCoupure": nombre(float(coupure), 0),
    }


def macros_du_tp_trois() -> dict[str, str]:
    """Les deux parties du TP."""
    return {**_macros_de_laplace(), **_macros_des_quadripoles()}
