"""Un reseau lineaire en regime stationnaire, resolu par ses noeuds.

La loi des noeuds est ecrite a chaque noeud ; chaque element qui impose
une tension -- une source, ou une inductance, qui en regime est un fil --
ajoute une inconnue, le courant qu'il debite, et une equation, la tension
qu'il impose. C'est l'analyse nodale modifiee.

Elle ne choisit **aucune maille** : c'est ce qui la rend independante des
corriges manuscrits, qui en choisissent une a chaque exercice. Les deux
methodes ne partagent que les lois de Kirchhoff.

**En regime stationnaire**, une inductance impose une tension nulle et un
condensateur ne laisse passer aucun courant. Les deux elements sont donc
portes ici sans valeur : seule leur place dans le reseau compte.

**Un reseau mal pose est refuse.** Un noeud sans chemin vers la reference
ou deux tensions contradictoires ne donnent pas une solution unique ; le
dire vaut mieux que rendre un potentiel arbitraire.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True)
class Resistance:
    """Une resistance entre deux noeuds."""

    de: str
    vers: str
    valeur: sp.Expr


@dataclass(frozen=True, slots=True)
class SourceDeTension:
    """Impose `V(plus) - V(moins) = valeur`."""

    plus: str
    moins: str
    valeur: sp.Expr


@dataclass(frozen=True, slots=True)
class SourceDeCourant:
    """Fait passer `valeur` du noeud `de` au noeud `vers`, a travers elle."""

    de: str
    vers: str
    valeur: sp.Expr


@dataclass(frozen=True, slots=True)
class Inductance:
    """En regime : un fil. Elle impose une tension nulle entre ses bornes."""

    de: str
    vers: str


@dataclass(frozen=True, slots=True)
class Condensateur:
    """En regime : un circuit ouvert. Il ne contribue a aucune equation."""

    de: str
    vers: str


Element = Resistance | SourceDeTension | SourceDeCourant | Inductance | Condensateur


def _bornes(element: Element) -> tuple[str, str]:
    """Les deux noeuds de l'element, le premier etant sa borne « plus »."""
    if isinstance(element, SourceDeTension):
        return element.plus, element.moins
    return element.de, element.vers


def _tension_imposee(element: SourceDeTension | Inductance) -> sp.Expr:
    """La tension que l'element impose entre ses bornes : nulle pour un fil."""
    if isinstance(element, SourceDeTension):
        return element.valeur
    return sp.Integer(0)


@dataclass(frozen=True, slots=True)
class Reseau:
    """Des elements, et le noeud dont le potentiel est pris pour zero."""

    elements: tuple[Element, ...]
    reference: str

    def _resoudre(self) -> dict[sp.Symbol, sp.Expr]:
        """Les potentiels et les courants debites, par la loi des noeuds."""
        noeuds = sorted({borne for e in self.elements for borne in _bornes(e)})
        if self.reference not in noeuds:
            msg = f"noeud de reference absent du reseau : {self.reference!r}"
            raise ValueError(msg)
        potentiel = {n: sp.Symbol(f"V_{n}") for n in noeuds if n != self.reference}
        potentiel[self.reference] = sp.Integer(0)
        sortant = dict.fromkeys(noeuds, sp.Integer(0))
        contraintes = []
        debits = []
        for rang, element in enumerate(self.elements):
            premier, second = _bornes(element)
            chute = potentiel[premier] - potentiel[second]
            if isinstance(element, Condensateur):
                continue
            if isinstance(element, Resistance):
                courant = chute / element.valeur
            elif isinstance(element, SourceDeCourant):
                courant = element.valeur
            else:
                contraintes.append(chute - _tension_imposee(element))
                debits.append(sp.Symbol(f"j_{rang}"))
                courant = -debits[-1]
            sortant[premier] += courant
            sortant[second] -= courant
        inconnues = [potentiel[n] for n in noeuds if n != self.reference] + debits
        equations = [sortant[n] for n in noeuds if n != self.reference] + contraintes
        solutions = sp.linsolve(equations, inconnues)
        if not solutions or next(iter(solutions)).free_symbols & set(inconnues):
            msg = (
                "le reseau n'a pas de solution unique : "
                "noeud flottant ou sources contradictoires"
            )
            raise ValueError(msg)
        etat = dict(zip(inconnues, next(iter(solutions)), strict=True))
        etat[sp.Symbol(f"V_{self.reference}")] = sp.Integer(0)
        return etat

    def potentiels(self) -> dict[str, sp.Expr]:
        """Le potentiel de chaque noeud, la reference etant a zero."""
        etat = self._resoudre()
        noeuds = {borne for e in self.elements for borne in _bornes(e)}
        return {n: sp.simplify(etat[sp.Symbol(f"V_{n}")]) for n in sorted(noeuds)}

    def courant_debite(self, source: SourceDeTension | Inductance) -> sp.Expr:
        """Le courant qui sort de la borne « plus » de cet element."""
        rang = self.elements.index(source)
        return sp.simplify(self._resoudre()[sp.Symbol(f"j_{rang}")])
