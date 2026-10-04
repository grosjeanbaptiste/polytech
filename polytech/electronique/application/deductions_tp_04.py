"""Ce que le TP 4 tire des systemes nodaux : gains, pulsations, Miller.

Chaque sortie est obtenue en *resolvant* le systeme assemble, puis en
passant a la limite que l'enonce demande (`R_L`, `r_o` ou `r_pi` infinis)
-- jamais en recopiant la formule du « logiciel de calcul symbolique »
de l'enonce, contre laquelle les tests la comparent.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.application import netlistes_tp_04 as nl
from polytech.electronique.application.symboles_tp_04 import (
    COL,
    RL,
    RO,
    RPI1,
    RPI2,
    VS,
    R,
    S,
    W,
)
from polytech.electronique.domain.netliste import resoudre

_EPSILON = sp.Symbol("epsilon", positive=True)


def limite_a_l_infini(expression: sp.Expr, symbole: sp.Symbol) -> sp.Expr:
    """La limite d'une fraction rationnelle quand `symbole -> oo`.

    Avec `symbole = 1/epsilon`, la limite se lit sur les termes de plus bas
    degre en `epsilon` du numerateur et du denominateur.
    """
    numerateur, denominateur = sp.fraction(
        sp.cancel(expression.subs(symbole, 1 / _EPSILON))
    )
    haut = sp.Poly(numerateur, _EPSILON)
    bas = sp.Poly(denominateur, _EPSILON)
    ordre_haut = min(m[0] for m in haut.monoms())
    ordre_bas = min(m[0] for m in bas.monoms())
    if ordre_haut < ordre_bas:
        msg = f"limite infinie quand {symbole} -> oo : {expression}"
        raise ValueError(msg)
    if ordre_haut > ordre_bas:
        return sp.Integer(0)
    return sp.cancel(
        haut.coeff_monomial(_EPSILON**ordre_haut)
        / bas.coeff_monomial(_EPSILON**ordre_bas)
    )


def _a_la_limite(expression: sp.Expr, *infinis: sp.Symbol) -> sp.Expr:
    for symbole in infinis:
        expression = limite_a_l_infini(expression, symbole)
    return expression


def transfert_du_dephaseur() -> sp.Expr:
    """`V_4 / V_S` pour `R_L -> oo`."""
    v4 = resoudre(nl.DEPHASEUR, 4)[3]
    return _a_la_limite(v4 / VS, RL)


def pulsation_d_inversion() -> sp.Expr:
    """La pulsation `> 0` ou `H(j omega)` est reel et strictement negatif."""
    reponse = transfert_du_dephaseur().subs(S, sp.I * W)
    candidates = sp.solve(sp.im(sp.expand_complex(sp.cancel(reponse))), W)
    retenues = [
        w
        for w in candidates
        if w.is_positive and sp.simplify(reponse.subs(W, w)).is_negative
    ]
    if len(retenues) != 1:
        msg = f"pulsation d'inversion introuvable ou multiple : {candidates}"
        raise ValueError(msg)
    return sp.simplify(retenues[0])


def gain_a_l_inversion() -> sp.Expr:
    """`H(j omega_r)`, reel et negatif par construction."""
    reponse = transfert_du_dephaseur().subs(S, sp.I * W)
    return sp.simplify(reponse.subs(W, pulsation_d_inversion()))


def transfert_de_la_source_commune() -> sp.Expr:
    """`V_4 / V_S` pour `r_o -> oo`."""
    v4 = resoudre(nl.SOURCE_COMMUNE, 4)[3]
    return _a_la_limite(v4 / VS, RO)


def gain_sans_capacite() -> sp.Expr:
    """`C_ol = 0` : plus aucun terme en `s`."""
    return sp.simplify(transfert_de_la_source_commune().subs(COL, 0))


def gain_sans_degenerescence() -> sp.Expr:
    """`C_ol = 0` et `R = 0` : le gain vaut `-A_v`."""
    return sp.simplify(gain_sans_capacite().subs(R, 0))


def tensions_de_la_cascode() -> tuple[sp.Expr, sp.Expr]:
    """`V_3` et `V_4` pour `r_pi1, r_pi2 -> oo`."""
    solution = resoudre(nl.CASCODE, 4)
    return (
        _a_la_limite(solution[2], RPI1, RPI2),
        _a_la_limite(solution[3], RPI1, RPI2),
    )


def courant_dans_la_charge() -> sp.Expr:
    """`V_4 / R_L`, sans hypothese sur `R_L`."""
    return sp.simplify(tensions_de_la_cascode()[1] / RL)


def gain_de_la_cascode_a_vide() -> sp.Expr:
    """`V_4 / V_S` pour `R_L -> oo`."""
    return _a_la_limite(tensions_de_la_cascode()[1] / VS, RL)
