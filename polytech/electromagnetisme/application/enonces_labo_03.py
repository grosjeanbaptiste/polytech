"""Les circuits du laboratoire III d'electromagnetisme (simulateur QUCS).

Source : `Electromagnetisme/LABOS/Lab_03_Simulateur_de_circuit_Enonce.pdf`,
le protocole auquel renvoie le devoir « Rapport du Labo III » du cours de
Charleroi.

**Figure 1** : `E1`-`R1` et `E2`-`R2` en parallele sur les bornes `A`-`B`.
**Figure 2**, lue sur l'enonce agrandi : les deux piles ont leur borne
`+` en haut (trait long) ; la source `I` fait passer son courant du noeud
bas de `R4` vers le noeud bas de `R5` ; `R7` et `R10` aboutissent a `B` ;
`R8` puis `R9` relient le bas des piles a `B`. Noeuds : `s` (haut de
`E1`), `t1` et `t2` (le long du fil du haut), `n4` (bas de `R4`, haut de
`E2`), `n6` (bas de `R5`), `p` (entre `R8` et `R9`), `m` (bas des piles).

La resistance de Thevenin de la figure 2 se calcule aussi **a la main**,
par associations et une transformation triangle-etoile. Ce second chemin
ne partage rien avec le resolveur, et c'est pourquoi il est ecrit ici.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.electromagnetisme.domain.circuits.elements_operationnels import Bobine
from polytech.electromagnetisme.domain.circuits.etoile import triangle_vers_etoile
from polytech.electromagnetisme.domain.circuits.operationnel import ReseauOperationnel
from polytech.electromagnetisme.domain.circuits.phaseurs import en_regime_sinusoidal
from polytech.electromagnetisme.domain.circuits.reseau import (
    Reseau,
    Resistance,
    SourceDeCourant,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE, original

if TYPE_CHECKING:
    from collections.abc import Sequence

RESISTANCES = (1, 1, 1, 2, 2, 3, 3, 4, 4, 5)
"""`R1` a `R10` de la figure 2, en ohms."""


def circuit_simple() -> Reseau:
    """Figure 1 : `R1 = 5`, `R2 = 10` ohms, `E1 = 5`, `E2 = 2` volts."""
    return Reseau(
        elements=(
            SourceDeTension("s1", "m", 5),
            Resistance("s1", "a", 5),
            SourceDeTension("s2", "m", 2),
            Resistance("s2", "a", 10),
        ),
        reference="m",
    )


def circuit_a_dix_resistances(e1: int = 10, e2: int = 5, i: int = 1) -> Reseau:
    """Figure 2 ; les sources sont des parametres pour la superposition."""
    r = dict(enumerate(RESISTANCES, start=1))
    return Reseau(
        elements=(
            SourceDeTension("s", "m", e1),
            Resistance("s", "t1", r[1]),
            Resistance("t1", "t2", r[2]),
            Resistance("t2", "a", r[3]),
            Resistance("t1", "n4", r[4]),
            Resistance("t2", "n6", r[5]),
            SourceDeTension("n4", "m", e2),
            SourceDeCourant("n4", "n6", i),
            Resistance("n4", "n6", r[6]),
            Resistance("n6", "b", r[7]),
            Resistance("m", "p", r[8]),
            Resistance("p", "b", r[9]),
            Resistance("a", "b", r[10]),
        ),
        reference="m",
    )


def _parallele(*valeurs: sp.Expr) -> sp.Expr:
    return 1 / sum(sp.Integer(1) / v for v in valeurs)


def etapes_de_kennelly() -> Sequence[sp.Expr]:
    """Les sept resistances intermediaires du calcul a la main, dans l'ordre."""
    r = {k: sp.Integer(v) for k, v in enumerate(RESISTANCES, start=1)}
    gauche = _parallele(r[1], r[4])
    vers_t2 = gauche + r[2]
    etoile = triangle_vers_etoile(vers_t2, r[6], r[5])
    par_m, par_n6 = etoile[1] + r[8] + r[9], etoile[2] + r[7]
    vers_b = etoile[0] + _parallele(par_m, par_n6)
    return (gauche, vers_t2, *etoile, par_m, par_n6, vers_b + r[3])


def resistance_par_kennelly() -> sp.Expr:
    """`R_TH` de la figure 2, sans resolveur : la derniere branche avec `R10`."""
    return _parallele(etapes_de_kennelly()[-1], sp.Integer(RESISTANCES[9]))


def tension_de_bobine(v0: sp.Expr, resistance: sp.Expr, inductance: sp.Expr) -> sp.Expr:
    """`v_L(t)` sous un echelon `v0` : la figure 3, resolue par Laplace."""
    reseau = ReseauOperationnel(
        elements=(
            SourceDeTension("s", "m", v0 / VARIABLE),
            Resistance("s", "l", resistance),
            Bobine("l", "m", inductance),
        ),
        reference="m",
    )
    return sp.simplify(original(reseau.potentiels()["l"]))


def frequence_de_coupure(resistance: sp.Expr, inductance: sp.Expr) -> sp.Expr:
    """`f_c = R / (2 pi L)` : la ou les deux impedances s'egalent."""
    return resistance / (2 * sp.pi * inductance)


def module_du_filtre(
    resistance: sp.Expr, inductance: sp.Expr, frequence: sp.Expr
) -> sp.Expr:
    """`|V_L / V_s|` a cette frequence, depuis la transmittance du circuit."""
    reseau = ReseauOperationnel(
        elements=(
            SourceDeTension("s", "m", 1),
            Resistance("s", "l", resistance),
            Bobine("l", "m", inductance),
        ),
        reference="m",
    )
    f = sp.Symbol("f", positive=True)
    phaseur = en_regime_sinusoidal(reseau.potentiels()["l"], 2 * sp.pi * f)
    return sp.simplify(sp.Abs(phaseur)).subs(f, frequence)
