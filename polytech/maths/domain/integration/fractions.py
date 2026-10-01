"""La decomposition en fractions simples, troisieme geste du fascicule.

Il la nomme explicitement : « 3e methode : decomposition en fractions
simples ». Ce que le corrige doit montrer n'est pas la primitive finale
mais la **somme de fractions**, terme a terme -- c'est elle qui rend
chaque morceau integrable a vue.

La garde refuse une fraction dont le numerateur n'est pas de degre
strictement inferieur au denominateur : il faut d'abord diviser, et
sauter cette division donnerait une decomposition fausse sans rien dire.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


@dataclass(frozen=True, slots=True)
class FractionRationnelle:
    """Un quotient de polynomes, et sa decomposition."""

    numerateur: sp.Expr
    denominateur: sp.Expr

    def _degres(self) -> tuple[int, int]:
        haut = sp.degree(sp.Poly(self.numerateur, VARIABLE), VARIABLE)
        bas = sp.degree(sp.Poly(self.denominateur, VARIABLE), VARIABLE)
        return int(haut), int(bas)

    def est_propre(self) -> bool:
        """Vrai si le numerateur est de degre strictement moindre."""
        haut, bas = self._degres()
        return haut < bas

    def partie_entiere(self) -> sp.Expr:
        """Le quotient de la division euclidienne, nul si la fraction est propre."""
        quotient, _ = sp.div(self.numerateur, self.denominateur, VARIABLE)
        return sp.expand(quotient)

    def termes(self) -> tuple[sp.Expr, ...]:
        """Les fractions simples, une par une ; leve si la fraction est impropre."""
        if not self.est_propre():
            haut, bas = self._degres()
            msg = (
                f"fraction impropre ({haut} >= {bas}) : diviser d'abord, "
                f"la partie entiere vaut {self.partie_entiere()}"
            )
            raise ValueError(msg)
        decomposee = sp.apart(self.numerateur / self.denominateur, VARIABLE)
        return tuple(decomposee.as_ordered_terms())

    def primitive(self) -> sp.Expr:
        """La primitive, obtenue en integrant chaque fraction simple."""
        return sp.simplify(sum(sp.integrate(t, VARIABLE) for t in self.termes()))
