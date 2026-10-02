"""Les generateurs equivalents d'un reseau, par trois mesures independantes.

Vu de deux bornes, un reseau lineaire se reduit a une source et une
resistance. Trois mesures les donnent, et chacune est une resolution a
part entiere d'un reseau different :

- `tension_a_vide` resout le reseau tel quel ;
- `courant_de_court_circuit` y ajoute un fil entre les deux bornes ;
- `resistance_vue_des_bornes` eteint les sources et injecte un ampere.

Que la premiere soit le produit des deux autres est le theoreme de
Thevenin. Ici ce n'est pas une definition : les trois se calculent
separement, de sorte que leur accord est une verification.

**Eteindre une source depend de sa nature.** Une source de tension
eteinte impose zero volt : c'est un fil. Une source de courant eteinte
impose zero ampere : c'est un circuit ouvert. Les confondre est l'erreur
classique du calcul de la resistance equivalente.
"""

from __future__ import annotations

from dataclasses import replace

import sympy as sp

from polytech.electromagnetisme.domain.circuits.reseau import (
    Element,
    Inductance,
    Reseau,
    SourceDeCourant,
    SourceDeTension,
)


def _exiger_deux_bornes(plus: str, moins: str) -> None:
    """Refuse de mesurer entre un noeud et lui-meme."""
    if plus == moins:
        msg = f"bornes confondues : {plus!r}"
        raise ValueError(msg)


def tension_a_vide(reseau: Reseau, plus: str, moins: str) -> sp.Expr:
    """`V(plus) - V(moins)`, aucune charge n'etant branchee."""
    _exiger_deux_bornes(plus, moins)
    potentiels = reseau.potentiels()
    return sp.simplify(potentiels[plus] - potentiels[moins])


def courant_de_court_circuit(reseau: Reseau, plus: str, moins: str) -> sp.Expr:
    """Le courant qui va de `plus` a `moins` dans un fil reliant les bornes.

    Le fil est une inductance en regime : il impose une tension nulle,
    et son courant devient une inconnue du reseau. Le courant qu'il
    **debite** a sa premiere borne est l'oppose de celui qui le traverse
    de cette borne vers l'autre.
    """
    _exiger_deux_bornes(plus, moins)
    fil = Inductance(plus, moins)
    court_circuite = replace(reseau, elements=(*reseau.elements, fil))
    return sp.simplify(-court_circuite.courant_debite(fil))


def _eteint(element: Element) -> Element:
    """La meme place dans le reseau, la valeur d'une source ramenee a zero."""
    if isinstance(element, SourceDeTension | SourceDeCourant):
        return replace(element, valeur=sp.Integer(0))
    return element


def eteindre(reseau: Reseau) -> Reseau:
    """Le reseau, toutes sources annulees : un fil ou un vide a leur place."""
    return replace(reseau, elements=tuple(_eteint(e) for e in reseau.elements))


def resistance_vue_des_bornes(reseau: Reseau, plus: str, moins: str) -> sp.Expr:
    """La resistance du reseau eteint, mesuree en y injectant un ampere.

    Le courant d'essai entre par `plus` et ressort par `moins` ; la
    tension qui apparait entre les bornes est alors, en ohms, la
    resistance cherchee.
    """
    _exiger_deux_bornes(plus, moins)
    essai = SourceDeCourant(moins, plus, sp.Integer(1))
    eteint = eteindre(reseau)
    sonde = replace(eteint, elements=(*eteint.elements, essai))
    return tension_a_vide(sonde, plus, moins)
