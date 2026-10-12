"""Une charge repartie, et le torseur qui la remplace.

Une charge repartie est une fonction `p(x)` de la position, de la
dimension d'une force par unite de longueur. Entre `x_min` et `x_max`
elle se reduit a son torseur resultant :

* sa **charge totale** `P = integrale de p(x) dx` ;
* son **moment** au pole origine, `integrale de p(x) x dx = P x_G`, ou
  `x_G` est l'abscisse du centre de gravite de la surface de la charge.

La resultante passe donc par `x_G` : c'est sa ligne d'action. La
mecanique des fluides n'ecrit pas autre chose quand elle integre la
pression sur une paroi.

Source : `MecaniqueRationnelle1/SYLLABUS/Syllabus_Partie_1_Statique.pdf`,
section 2.2.6, pages 96 et 97, relation (2.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.mecanique.domain.statique.torseurs import Torseur


@dataclass(frozen=True, slots=True)
class ChargeRepartie:
    """Une densite `p(x)`, positive vers le bas, et l'intervalle qui la porte."""

    densite: sp.Expr
    variable: sp.Symbol
    debut: sp.Expr
    fin: sp.Expr

    def charge_totale(self) -> sp.Expr:
        """`P = integrale de p(x) dx`, relation (2.5)."""
        bornes = (self.variable, self.debut, self.fin)
        return sp.simplify(sp.integrate(self.densite, bornes))

    def moment_a_l_origine(self) -> sp.Expr:
        """`integrale de p(x) x dx`, la grandeur du moment au pole origine."""
        bornes = (self.variable, self.debut, self.fin)
        return sp.integrate(self.densite * self.variable, bornes)

    def abscisse_du_centre(self) -> sp.Expr:
        """`x_G`, ou passe la resultante ; leve si la charge totale est nulle."""
        totale = self.charge_totale()
        if totale == 0:
            msg = "charge totale nulle : la resultante n'a pas de ligne d'action"
            raise ValueError(msg)
        return sp.simplify(self.moment_a_l_origine() / totale)

    def torseur_a_l_origine(self) -> Torseur:
        """`R = -P u_y` et `M_O = -P x_G u_z` : une charge qui agit vers le bas."""
        zero = sp.Integer(0)
        return Torseur(
            resultante=(zero, -self.charge_totale(), zero),
            moment=(zero, zero, -sp.simplify(self.moment_a_l_origine())),
            pole=(zero, zero, zero),
        )
