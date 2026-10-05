"""La suspension active : un systeme lineaire a deux entrees.

Planches « Systemes » du cours de signaux et systemes, exemple 4. La
masse suspendue obeit a

`M x_v'' = -k (x_v - x_r - L_0) - f d(x_v - x_r)/dt + F - M g`,

lineaire, mais avec des termes constants. Au repos, sous `x_r^0` et
`F^0`, elle se tient en `x_v^0 = x_r^0 + L_0 + (F^0 - M g) / k`. En
variations autour de ce point les constantes disparaissent, et

`X_v = (p f + k) / (p^2 M + p f + k) X_r + 1 / (p^2 M + p f + k) F`.

Un systeme lineaire a plusieurs entrees a une transmittance par entree,
et sa sortie est la somme de leurs effets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.transmittance import Transmittance

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class Suspension:
    """La masse, la raideur, le frottement, la longueur a vide, la pesanteur."""

    masse: sp.Expr
    raideur: sp.Expr
    frottement: sp.Expr
    longueur: sp.Expr
    pesanteur: sp.Expr

    def acceleration(
        self, position: sp.Expr, route: sp.Expr, force: sp.Expr
    ) -> sp.Expr:
        """`x_v''` a vitesses nulles : ce que les forces laissent au repos."""
        rappel = -self.raideur * (position - route - self.longueur)
        return (rappel + force - self.masse * self.pesanteur) / self.masse

    def position_de_repos(self, route: sp.Expr, force: sp.Expr) -> sp.Expr:
        """`x_v^0 = x_r^0 + L_0 + (F^0 - M g) / k`."""
        poids = self.masse * self.pesanteur
        return route + self.longueur + (force - poids) / self.raideur

    def transmittances(self) -> tuple[Transmittance, Transmittance]:
        """Celle de la route, puis celle de la force, en variations."""
        passage = P * self.frottement + self.raideur
        denominateur = P**2 * self.masse + passage
        return (
            Transmittance(expression=passage / denominateur),
            Transmittance(expression=1 / denominateur),
        )


def sortie_par_superposition(effets: Mapping[Transmittance, sp.Expr]) -> sp.Expr:
    """`Y = somme G_i U_i` : chaque entree agit par sa propre transmittance."""
    if not effets:
        msg = "aucune entree : la sortie d'un systeme sans entree n'est pas definie"
        raise ValueError(msg)
    termes = (modele.expression * entree for modele, entree in effets.items())
    return sp.cancel(sum(termes, sp.S.Zero))
