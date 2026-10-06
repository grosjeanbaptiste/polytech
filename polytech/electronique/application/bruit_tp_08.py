"""TP 8 d'electronique : le bruit thermique des resistances d'un montage a AOP.

Chaque resistance bruyante est une resistance sans bruit en serie avec
une source de tension de densite `4 k_B T R`. Le montage est resolu par
ses noeuds avec ces sources, ce qui rend le transfert de chacune vers la
sortie ; les densites s'ajoutent ensuite, ponderees par `|H|^2`, parce
que les sources sont independantes (`polytech.signaux`).

Les deux montages de l'enonce -- non-inverseur et inverseur -- ne
different que par l'endroit ou le signal entre : signal eteint, ils sont
le meme circuit, donc le meme bruit de sortie.
"""

from __future__ import annotations

import enum
import functools
import math
from typing import Final

import sympy as sp

from polytech.electronique.domain.aop import aop, sortie_ideale
from polytech.electronique.domain.bruit_thermique import densite_thermique
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeTension,
)
from polytech.signaux.domain import processus_aleatoires as pa

TEMPERATURE: Final = 298.0
"""25 degres Celsius, arrondis comme le corrige : `25 + 273`."""
R1: Final = 1e3
R2: Final = 100e3
_E1, _E2, _VIN = sp.symbols("e_1 e_2 V_in")
_COURANTS: Final = sp.symbols("I_1 I_2 I_3")


class Montage(enum.Enum):
    """Les deux montages de l'exercice 2.2."""

    NON_INVERSEUR = "signal sur l'entree +"
    INVERSEUR = "signal a travers R_1"


def densite_efficace(resistance: float) -> float:
    """`racine(4 k_B T R)`, en volts par racine de hertz."""
    return math.sqrt(densite_thermique(resistance, TEMPERATURE))


def depuis_la_valeur_de_l_enonce(resistance: float) -> float:
    """Exercice 2.1 : le bruit croit comme `racine(R)`, depuis 4 nV a 1 kohm."""
    if resistance <= 0:
        msg = f"resistance nulle ou negative : {resistance} ohm"
        raise ValueError(msg)
    return 4e-9 * math.sqrt(resistance / R1)


@functools.cache
def _sortie(montage: Montage) -> sp.Expr:
    """`V_out` en fonction du signal et des deux tensions de bruit.

    Noeuds : 1 le pied de `R_1`, 2 apres sa source de bruit, 3 l'entree
    `-`, 4 entre `R_2` et sa source de bruit, 5 la sortie, 6 l'entree `+`.
    """
    inverseur = montage is Montage.INVERSEUR
    elements: list[Element] = [
        SourceDeTension(1, 0, _VIN if inverseur else sp.Integer(0), _COURANTS[0]),
        SourceDeTension(6, 0, sp.Integer(0) if inverseur else _VIN, _COURANTS[1]),
        SourceDeTension(2, 1, _E1, _COURANTS[2]),
        SourceDeTension(5, 4, _E2, sp.Symbol("I_4")),
        Admittance(2, 3, sp.Integer(1) / sp.nsimplify(R1)),
        Admittance(3, 4, sp.Integer(1) / sp.nsimplify(R2)),
        *aop(6, 3, 5),
    ]
    return sp.expand(sortie_ideale(elements, 6, 5))


def transferts_des_bruits() -> tuple[sp.Expr, sp.Expr]:
    """Les gains de `e_1` et de `e_2` vers la sortie : les memes pour les deux."""
    par_montage = {
        (sp.diff(_sortie(m), _E1), sp.diff(_sortie(m), _E2)) for m in Montage
    }
    (commun,) = par_montage
    return commun


def gain_du_signal(montage: Montage) -> sp.Expr:
    """Le gain de `V_in` vers la sortie : il depend, lui, du montage."""
    return sp.diff(_sortie(montage), _VIN)


def densite_en_sortie(
    *, par_kilohm: float | None = None, echelle: float = 1.0
) -> float:
    """`|H_1|^2 S_1 + |H_2|^2 S_2`, en volts carres par hertz.

    `par_kilohm` impose la densite efficace d'un kilohm -- les 4 nV de
    l'enonce -- au lieu de la calculer. `echelle` multiplie les deux
    resistances : les gains n'en dependent pas, le bruit si.
    """
    if echelle <= 0:
        msg = f"facteur d'echelle nul ou negatif : {echelle}"
        raise ValueError(msg)
    if par_kilohm is not None and par_kilohm <= 0:
        msg = f"densite nulle ou negative : {par_kilohm}"
        raise ValueError(msg)
    unite = densite_efficace(1e3) if par_kilohm is None else par_kilohm
    sources = (unite**2 * echelle * r / 1e3 for r in (R1, R2))
    filtrees = (
        pa.densite_en_sortie(gain, sp.Float(s))
        for gain, s in zip(transferts_des_bruits(), sources, strict=True)
    )
    return float(pa.densite_de_la_somme(*filtrees))


def parts_du_bruit_de_sortie() -> tuple[float, float]:
    """La fraction de la densite de sortie due a `R_1`, puis a `R_2`."""
    de_r1, de_r2 = (abs(complex(h)) ** 2 for h in transferts_des_bruits())
    total = de_r1 * R1 + de_r2 * R2
    return de_r1 * R1 / total, de_r2 * R2 / total


def rapportee_a_l_entree(montage: Montage, *, par_kilohm: float | None = None) -> float:
    """La densite efficace d'une source, en serie avec `V_in`, de meme effet."""
    sortie = math.sqrt(densite_en_sortie(par_kilohm=par_kilohm))
    return sortie / abs(complex(gain_du_signal(montage)))


def resistance_equivalente(montage: Montage) -> float:
    """La resistance dont le bruit thermique egalerait le bruit rapporte."""
    return rapportee_a_l_entree(montage) ** 2 / densite_thermique(1.0, TEMPERATURE)
