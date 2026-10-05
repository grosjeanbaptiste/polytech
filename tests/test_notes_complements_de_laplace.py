"""Notes de cours de 1993, complements manuscrits au chapitre 3 (CDI 2002-2003).

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 101 a
105 du fichier (un scan, lu en image). Les theoremes de la valeur initiale
et de la valeur finale, appliques au circuit RC, a la stabilite d'une
fonction de transfert et au gain en regime indiciel.

Un ecart du manuscrit est etabli -- le scan a ete relu a plus haute
resolution avant de conclure : page 105, les residus de la reponse
indicielle sont ecrits 1 et -1 ; ce sont -1/2 et 1/3.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.circuits.enclenchement import enclenchement
from polytech.electromagnetisme.domain.symboles import TEMPS as TEMPS_DES_CIRCUITS
from polytech.maths.domain.laplace.heaviside import (
    fractions_simples,
    par_fractions_simples,
)
from polytech.maths.domain.laplace.theoremes import (
    limite_en_zero,
    poles_genants,
    valeur_finale,
    valeur_initiale,
)
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

R = sp.Rational
TRANSFERT = 1 / ((P + 2) * (P + 3))


def test_la_reponse_indicielle_du_circuit_rc() -> None:
    """`R i + (1/C) int i = H(t)` : `I = (1/R) / (p + 1/(CR))`, `i = e^(-t/CR) / R`."""
    resistance, capacite = sp.symbols("R C", positive=True)
    image = (1 / P) / (resistance + 1 / (capacite * P))
    assert (
        sp.simplify(image - (1 / resistance) / (P + 1 / (capacite * resistance))) == 0
    )
    courant = par_fractions_simples(image)
    attendu = sp.exp(-T / (capacite * resistance)) / resistance
    assert sp.simplify(courant - attendu) == 0


def test_la_valeur_initiale_du_courant_sans_inverser() -> None:
    """Page 102 : `lim p I(p) = 1/R = i(0+)`."""
    resistance, capacite = sp.symbols("R C", positive=True)
    image = (1 / resistance) / (P + 1 / (capacite * resistance))
    assert valeur_initiale(image) == 1 / resistance
    assert par_fractions_simples(image).subs(T, 0) == 1 / resistance


def test_le_condensateur_se_charge_comme_le_dit_l_enclenchement_des_circuits() -> None:
    """Lien : `u_C = 1 - e^(-t/RC)` par l'enclenchement, et `i = C du_C/dt`."""
    resistance, capacite = sp.symbols("R C", positive=True)
    tension = enclenchement(sp.Integer(1), sp.Integer(0), resistance * capacite)
    courant = capacite * sp.diff(tension, TEMPS_DES_CIRCUITS)
    attendu = sp.exp(-TEMPS_DES_CIRCUITS / (capacite * resistance)) / resistance
    assert sp.simplify(courant - attendu) == 0


def test_la_stabilite_se_lit_sur_les_poles_de_p_f() -> None:
    """Page 104 : un pole en `p = 3` fait diverger `f` ; celui en `p = -2`, non."""
    instable = 1 / ((P + 2) * (P - 3))
    assert limite_en_zero(instable) == 0
    assert poles_genants(instable) == (3,)
    assert poles_genants(TRANSFERT / P) == ()


def test_le_gain_en_regime_indiciel() -> None:
    """Page 105 : `lim y(t) = lim p Y(p) = lim G(p)` ; pour `1/((p+2)(p+3))`, `1/6`."""
    assert valeur_finale(TRANSFERT / P) == R(1, 6)
    assert TRANSFERT.subs(P, 0) == R(1, 6)
    assert sp.limit(par_fractions_simples(TRANSFERT / P), T, sp.oo) == R(1, 6)


def test_les_residus_ecrits_page_105_sont_faux() -> None:
    """Le manuscrit ecrit `Y = (1/6)/p + 1/(p+2) - 1/(p+3)` et l'original qui va avec.

    Les residus sont `-1/2` et `1/3`. Second chemin : un systeme au repos
    repond a l'echelon en partant de 0 ; la reponse ecrite vaudrait `1/6` en
    `t = 0`. La limite `1/6`, elle, est juste.
    """
    justes = {R(1, 6) / P, -R(1, 2) / (P + 2), R(1, 3) / (P + 3)}
    assert set(fractions_simples(TRANSFERT / P)) == justes
    reponse = par_fractions_simples(TRANSFERT / P)
    attendue = R(1, 6) - sp.exp(-2 * T) / 2 + sp.exp(-3 * T) / 3
    assert sp.simplify(reponse - attendue) == 0
    assert reponse.subs(T, 0) == 0
    ecrite = R(1, 6) + sp.exp(-2 * T) - sp.exp(-3 * T)
    assert ecrite.subs(T, 0) == R(1, 6)
    imprimee = R(1, 6) / P + 1 / (P + 2) - 1 / (P + 3)
    assert sp.simplify(imprimee - TRANSFERT / P) != 0
