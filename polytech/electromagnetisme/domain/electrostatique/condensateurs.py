"""Le condensateur, et ses associations.

Le syllabus d'electricite generale (3.16 a 3.20) :

* un condensateur parfait porte sur ses armatures des charges opposees,
  proportionnelles a la tension : `Q = C U` (3.16.2) ;
* en serie, la meme charge traverse la chaine : `1/C = somme 1/C_i`
  (3.18.2), et la tension se repartit en raison inverse des capacites ;
* en parallele, la meme tension s'applique a tous : `C = somme C_i`
  (3.19.2), et la charge se repartit en raison directe des capacites ;
* un groupement qui n'est ni l'un ni l'autre (le pont de la figure 3.20.1b)
  se resout en ecrivant que la charge totale des armatures reliees a un
  noeud interieur, initialement neutre, reste nulle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

Branche = tuple[str, str, Any]
"""Un condensateur du reseau : ses deux noeuds et sa capacite."""


def _capacites(valeurs: Sequence[Any]) -> list[sp.Expr]:
    if not valeurs:
        msg = "aucun condensateur a associer"
        raise ValueError(msg)
    capacites = [sp.sympify(c) for c in valeurs]
    for capacite in capacites:
        if capacite.is_positive is not True:
            msg = f"capacite strictement positive attendue, et non {capacite}"
            raise ValueError(msg)
    return capacites


def charge_d_un_condensateur(capacite: Any, tension: Any) -> Any:  # noqa: ANN401 -- reels ou symboles
    """`Q = C U`."""
    return capacite * tension


def capacite_en_serie(capacites: Sequence[Any]) -> sp.Expr:
    """`1/C = somme 1/C_i`."""
    return 1 / sp.Add(*(1 / c for c in _capacites(capacites)))


def capacite_en_parallele(capacites: Sequence[Any]) -> sp.Expr:
    """`C = somme C_i`."""
    return sp.Add(*_capacites(capacites))


def tensions_en_serie(tension: Any, capacites: Sequence[Any]) -> list[sp.Expr]:  # noqa: ANN401 -- reelle ou symbolique
    """`U_k = Q / C_k` : la tension de chaque condensateur de la chaine."""
    charge = capacite_en_serie(capacites) * sp.sympify(tension)
    return [charge / c for c in _capacites(capacites)]


def charges_en_parallele(charge: Any, capacites: Sequence[Any]) -> list[sp.Expr]:  # noqa: ANN401 -- reelle ou symbolique
    """`Q_k = C_k U` : la charge de chaque condensateur du faisceau."""
    tension = sp.sympify(charge) / capacite_en_parallele(capacites)
    return [c * tension for c in _capacites(capacites)]


def capacite_equivalente(
    branches: Sequence[Branche], entree: str, sortie: str
) -> sp.Expr:
    """La capacite vue entre deux bornes d'un reseau quelconque de condensateurs."""
    if entree == sortie:
        msg = f"deux bornes distinctes attendues, et non deux fois {entree}"
        raise ValueError(msg)
    noeuds = sorted({n for a, b, _ in branches for n in (a, b)})
    for borne in (entree, sortie):
        if borne not in noeuds:
            msg = f"borne absente du reseau : {borne}"
            raise ValueError(msg)
    potentiels: dict[str, sp.Expr] = {n: sp.Symbol(f"V_{n}") for n in noeuds}
    potentiels |= {entree: sp.Integer(1), sortie: sp.Integer(0)}

    def charge_du_noeud(noeud: str) -> sp.Expr:
        sortante = [
            c * (potentiels[a] - potentiels[b]) for a, b, c in branches if a == noeud
        ]
        entrante = [
            c * (potentiels[b] - potentiels[a]) for a, b, c in branches if b == noeud
        ]
        return sp.Add(*sortante, *entrante)

    interieurs = [n for n in noeuds if n not in (entree, sortie)]
    inconnues = [potentiels[n] for n in interieurs]
    bilans = [charge_du_noeud(n) for n in interieurs]
    (solution,) = sp.linsolve(bilans, inconnues) if interieurs else [()]
    if any(valeur.free_symbols & set(inconnues) for valeur in solution):
        msg = "noeud sans liaison vers les bornes : son potentiel reste indetermine"
        raise ValueError(msg)
    resolus = dict(zip(inconnues, solution, strict=True))
    return sp.simplify(charge_du_noeud(entree).subs(resolus))
