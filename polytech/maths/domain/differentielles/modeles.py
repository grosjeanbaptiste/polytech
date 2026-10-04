"""Les modeles que le cours donne en application des EDO du premier ordre.

* la loi de Newton, `F = d(m v)/dt` (planche 2) ;
* le modele **malthusien**, `y' = k y - r` : reproduction au taux `k`,
  pertes `r` (planche 41) -- les mulots et les hiboux de la planche 42 ;
* le modele **logistique** de Verhulst, `y' = lambda y (K - y)`, qui
  borne la population par ce que le milieu supporte (planche 45) ;
* un modele de **prix** : offre `Q_O = -alpha + beta P`, demande
  `Q_D = gamma - delta P`, et un prix qui monte quand la demande depasse
  l'offre, `P' = k (Q_D - Q_O)` (planches 46-47).

Aucune discipline du depot ne les revendique : ils restent avec le cours
qui les enseigne. La desintegration, la vidange et le circuit RLC, eux,
vivent dans `chemistry`, `fluides` et `electromagnetisme`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.differentielles.autonomes import EquationAutonome
from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.differentielles.premier_ordre import INCONNUE


def force(masse: object, vitesse: object, temps: sp.Symbol) -> sp.Expr:
    """`F = d(m v)/dt`."""
    return sp.diff(sp.sympify(masse) * sp.sympify(vitesse), temps)


def malthus(reproduction: object, pertes: object) -> EquationLineaire:
    """`y' = k y - r`, ecrite `y' - k y = -r`."""
    return EquationLineaire(-sp.sympify(reproduction), -sp.sympify(pertes))


def logistique(taux: object, capacite: object) -> EquationAutonome:
    """`y' = lambda y (K - y)`."""
    croissance = sp.sympify(taux) * INCONNUE * (sp.sympify(capacite) - INCONNUE)
    return EquationAutonome(croissance)


@dataclass(frozen=True, slots=True)
class PrixDeMarche:
    """L'offre `-alpha + beta P`, la demande `gamma - delta P`, la reactivite `k`."""

    alpha: sp.Expr
    beta: sp.Expr
    gamma: sp.Expr
    delta: sp.Expr
    k: sp.Expr

    def offre(self, prix: object) -> sp.Expr:
        """`Q_O(P) = -alpha + beta P`."""
        return -self.alpha + self.beta * sp.sympify(prix)

    def demande(self, prix: object) -> sp.Expr:
        """`Q_D(P) = gamma - delta P`."""
        return self.gamma - self.delta * sp.sympify(prix)

    def equation(self) -> EquationLineaire:
        """`P' + k (beta + delta) P = k (alpha + gamma)`."""
        return EquationLineaire(
            self.k * (self.beta + self.delta), self.k * (self.alpha + self.gamma)
        )

    def prix_d_equilibre(self) -> sp.Expr:
        """`(alpha + gamma) / (beta + delta)` : la ou l'offre egale la demande."""
        return (self.alpha + self.gamma) / (self.beta + self.delta)


def prix_de_marche(*parametres: object) -> PrixDeMarche:
    """Le modele, pour `alpha`, `beta`, `gamma`, `delta` et `k` dans cet ordre."""
    return PrixDeMarche(*(sp.sympify(p) for p in parametres))
