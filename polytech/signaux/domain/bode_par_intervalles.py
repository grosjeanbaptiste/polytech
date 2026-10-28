"""Le trace asymptotique d'un plan de Bode, intervalle par intervalle.

Une fonction de transfert a poles et zeros reels se factorise en

    H(p) = A_0 prod (1 + p / z_i) / prod (1 + p / p_k).

Les `z_i` et les `p_k`, ranges par ordre croissant, decoupent l'axe des
pulsations. Loin d'une cassure `c`, le facteur `1 + j w / c` vaut 1 en
dessous et `j w / c` au-dessus : sur chaque intervalle la fonction se
reduit a `k (j w)^n`. Sur des axes logarithmiques c'est une droite de
pente `n`, et une phase de `n` quarts de tour.

Chaque zero franchi releve donc la pente d'une unite, chaque pole
l'abaisse d'autant ; une paire de poles complexes conjugues de module
`p_0` agit, loin d'elle, comme un pole double en `p_0`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.lecture_d_expressions import module_de

if TYPE_CHECKING:
    from collections.abc import Sequence


_W = sp.Symbol("omega", positive=True)


def distance_logarithmique(depart: float, arrivee: float) -> float:
    """`log10(arrivee / depart)` : ce qu'un axe logarithmique mesure, en decades."""
    if depart <= 0 or arrivee <= 0:
        msg = f"un axe logarithmique n'a pas de zero : {depart} et {arrivee}"
        raise ValueError(msg)
    return math.log10(arrivee / depart)


def intervalles(zeros: Sequence[float], poles: Sequence[float]) -> tuple[float, ...]:
    """Les cassures, zeros et poles confondus, par ordre croissant."""
    return tuple(sorted((*zeros, *poles)))


@dataclass(frozen=True, slots=True)
class Factorisation:
    """`A_0 prod (1 + p / z_i) / prod (1 + p / p_k)`, par son gain et ses cassures."""

    gain: sp.Expr
    zeros: tuple[sp.Expr, ...]
    poles: tuple[sp.Expr, ...]

    def exacte(self, variable: sp.Expr) -> sp.Expr:
        """La fonction de transfert elle-meme."""
        numerateur = sp.prod([1 + variable / zero for zero in self.zeros])
        denominateur = sp.prod([1 + variable / pole for pole in self.poles])
        return self.gain * numerateur / denominateur

    def approchee(
        self, ordre: Sequence[sp.Expr], rang: int, variable: sp.Expr
    ) -> sp.Expr:
        """`H` sur l'intervalle ou les `rang` premieres cassures sont franchies.

        `ordre` range toutes les cassures par ordre croissant : il est
        donne, et non deduit, pour que des cassures symboliques le soient.
        """
        if sorted(map(str, ordre)) != sorted(map(str, (*self.zeros, *self.poles))):
            msg = "l'ordre doit nommer chaque zero et chaque pole, une fois chacun"
            raise ValueError(msg)
        if not 0 <= rang <= len(ordre):
            msg = f"rang hors de [0, {len(ordre)}] : {rang}"
            raise ValueError(msg)
        zeros, expression = list(self.zeros), sp.sympify(self.gain)
        for cassure in ordre[:rang]:
            if cassure in zeros:
                zeros.remove(cassure)
                expression *= variable / cassure
            else:
                expression /= variable / cassure
        return sp.simplify(expression)

    def module_exact(self, pulsation: float) -> float:
        """`|H(j w)|` de la fonction elle-meme, en un point."""
        return module_de(self.exacte(sp.I * _W), {_W: pulsation})

    def gain_du_palier(self, ordre: Sequence[sp.Expr], rang: int) -> sp.Expr:
        """Le niveau constant de l'intervalle ; leve si la pente n'y est pas nulle."""
        if self.pente(ordre, rang) != 0:
            msg = f"l'intervalle de rang {rang} n'est pas un palier"
            raise ValueError(msg)
        return self.approchee(ordre, rang, sp.I * _W)

    def pente(self, ordre: Sequence[sp.Expr], rang: int) -> int:
        """`n` de `k (j w)^n` : zeros franchis moins poles franchis."""
        franchis = list(ordre[:rang])
        zeros = sum(1 for zero in self.zeros if zero in franchis)
        return 2 * zeros - len(franchis)


def parametres_du_rlc(
    resistance: sp.Expr, inductance: sp.Expr, capacite: sp.Expr
) -> tuple[sp.Expr, sp.Expr]:
    """`(p_0, zeta) = (1 / racine(L C), (R / 2) racine(C / L))` du RLC passe-bas."""
    propre = 1 / sp.sqrt(inductance * capacite)
    return propre, resistance / 2 * sp.sqrt(capacite / inductance)
