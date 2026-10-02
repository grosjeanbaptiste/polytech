"""Un reseau en regime transitoire, resolu dans le domaine de Laplace.

C'est le reseau stationnaire, auquel s'ajoutent les deux elements qui ont
de la memoire. Leurs conditions initiales entrent dans les equations
elles-memes :

- une **bobine** parcourue par `I` impose `V = L (p I - i(0))`, plus
  `M (p I' - i'(0))` pour chaque bobine qui lui est couplee ;
- une **capacite** sous la tension `U` fait passer `C (p U - u(0))`.

La loi des noeuds est ecrite a chaque noeud ; chaque source de tension
et chaque bobine ajoute une inconnue de courant. Aucune maille n'est
choisie, ce qui rend la methode independante des corriges manuscrits.

**Le courant d'une bobine** va de sa borne `de` a sa borne `vers` ; celui
d'une source de tension **sort** par sa borne plus. Les deux conventions
different, et `courant` rend chacun dans la sienne.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.electromagnetisme.domain.circuits.elements_operationnels import (
    Bobine,
    Capacite,
    Couplage,
    Element,
    bornes,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Resistance,
    SourceDeCourant,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE


@dataclass(frozen=True, slots=True)
class ReseauOperationnel:
    """Des elements, leurs couplages, et le noeud pris pour reference."""

    elements: tuple[Element, ...]
    reference: str
    couplages: tuple[Couplage, ...] = ()

    def _inconnue(self, element: Element) -> sp.Symbol:
        """Le courant inconnu d'une source de tension ou d'une bobine."""
        if element not in self.elements:
            msg = f"bobine absente du reseau : {element}"
            raise ValueError(msg)
        return sp.Symbol(f"j_{self.elements.index(element)}")

    def _flux_impose(self, bobine: Bobine) -> sp.Expr:
        """`L (p I - i_0)`, et un terme mutuel par couplage qui la nomme."""
        propre = self._inconnue(bobine)
        total = bobine.inductance * (VARIABLE * propre - bobine.courant_initial)
        for couplage in self.couplages:
            if bobine not in {couplage.premiere, couplage.seconde}:
                continue
            autre = (
                couplage.seconde if bobine == couplage.premiere else couplage.premiere
            )
            courant = self._inconnue(autre)
            total += couplage.mutuelle * (VARIABLE * courant - autre.courant_initial)
        return total

    def _branche(
        self, element: Element, chute: sp.Expr
    ) -> tuple[sp.Expr, sp.Expr | None]:
        """Le courant qui quitte la premiere borne, et la contrainte eventuelle."""
        if isinstance(element, Resistance):
            return chute / element.valeur, None
        if isinstance(element, SourceDeCourant):
            return element.valeur, None
        if isinstance(element, Capacite):
            return element.capacite * (
                VARIABLE * chute - element.tension_initiale
            ), None
        if isinstance(element, SourceDeTension):
            return -self._inconnue(element), chute - element.valeur
        return self._inconnue(element), chute - self._flux_impose(element)

    def _resoudre(self) -> dict[sp.Symbol, sp.Expr]:
        """Les potentiels et les courants inconnus, par la loi des noeuds."""
        noeuds = sorted({borne for e in self.elements for borne in bornes(e)})
        if self.reference not in noeuds:
            msg = f"noeud de reference absent du reseau : {self.reference!r}"
            raise ValueError(msg)
        potentiel = {n: sp.Symbol(f"V_{n}") for n in noeuds if n != self.reference}
        potentiel[self.reference] = sp.Integer(0)
        sortant = dict.fromkeys(noeuds, sp.Integer(0))
        contraintes = []
        for element in self.elements:
            premier, second = bornes(element)
            courant, contrainte = self._branche(
                element, potentiel[premier] - potentiel[second]
            )
            sortant[premier] += courant
            sortant[second] -= courant
            if contrainte is not None:
                contraintes.append(contrainte)
        libres = [n for n in noeuds if n != self.reference]
        imposes = [e for e in self.elements if isinstance(e, SourceDeTension | Bobine)]
        inconnues = [potentiel[n] for n in libres] + [
            self._inconnue(e) for e in imposes
        ]
        solutions = sp.linsolve([sortant[n] for n in libres] + contraintes, inconnues)
        if not solutions or next(iter(solutions)).free_symbols & set(inconnues):
            msg = "le reseau n'a pas de solution unique"
            raise ValueError(msg)
        etat = dict(zip(inconnues, next(iter(solutions)), strict=True))
        etat[sp.Symbol(f"V_{self.reference}")] = sp.Integer(0)
        return etat

    def potentiels(self) -> dict[str, sp.Expr]:
        """La transformee du potentiel de chaque noeud."""
        etat = self._resoudre()
        noeuds = {borne for e in self.elements for borne in bornes(e)}
        return {n: sp.simplify(etat[sp.Symbol(f"V_{n}")]) for n in sorted(noeuds)}

    def courant(self, element: Bobine | SourceDeTension) -> sp.Expr:
        """La transformee du courant de cet element, dans sa convention."""
        return sp.simplify(self._resoudre()[self._inconnue(element)])
