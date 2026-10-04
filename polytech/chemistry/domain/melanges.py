"""Le bilan de matiere d'une cuve parfaitement agitee.

Une cuve de volume `V` recoit un debit `q` de solution a la concentration
`C_in` ; le melange, homogeneise instantanement, sort au meme debit. La
quantite de solute dans la cuve varie de ce qui entre moins ce qui sort :

    V C'(t) = q C_in - q C(t),

une equation differentielle lineaire du premier ordre. La concentration
rejoint celle de l'entree avec la constante de temps `V / q` :

    C(t) = C_in + (C(0) - C_in) e^(-q t / V).

C'est l'exemple 1.5 du syllabus de mathematiques appliquees (de l'alcool
pur dans une cuve d'eau) ; le bilan, lui, est de genie chimique.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.differentielles.lineaire import EquationLineaire


def equation_de_la_cuve(
    debit: object, volume: object, entree: object
) -> EquationLineaire:
    """`C' + (q / V) C = (q / V) C_in`."""
    renouvellement = sp.sympify(debit) / sp.sympify(volume)
    return EquationLineaire(renouvellement, renouvellement * sp.sympify(entree))


def concentration_dans_la_cuve(
    debit: object,
    volume: object,
    *,
    entree: object,
    initiale: object,
    instant: object,
) -> sp.Expr:
    """`C_in + (C(0) - C_in) e^(-q t / V)`."""
    renouvellement = sp.sympify(debit) / sp.sympify(volume)
    ecart = sp.sympify(initiale) - sp.sympify(entree)
    return sp.sympify(entree) + ecart * sp.exp(-renouvellement * sp.sympify(instant))
