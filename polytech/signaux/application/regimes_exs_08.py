"""Feuille « Performances », exercices 1 et 2 : les valeurs de regime.

En regime tout est constant. Un integrateur ne peut alors recevoir que
zero -- sinon sa sortie deriverait -- et c'est cette seule remarque qui
donne toutes les valeurs, sans calculer une seule transmittance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.signaux.domain.boucle_fermee import Boucle
from polytech.signaux.domain.marges import marges_de_stabilite
from polytech.signaux.domain.symboles import IMAGE as P

ALPHA, BETA = sp.symbols("alpha beta", positive=True)
CAPTEUR: Final = sp.Integer(2)
CORRECTEUR_DEUX: Final = (
    sp.Rational(42, 100)
    * (1 + sp.Rational(821, 100) * P)
    * (1 + sp.Rational(274, 100) * P)
    / (sp.Rational(821, 100) * P * (1 + sp.Rational(274, 1000) * P))
)
"""Le correcteur de l'exercice 2 : un PID, donc une action integrale."""
PROCEDE_DEUX: Final = 1 / ((1 + 2 * P) * (1 + 10 * P))
"""Le procede de l'exercice 2, sans son retard d'une seconde."""


@dataclass(frozen=True, slots=True)
class Regime:
    """La sortie, la commande, son image apres l'actionneur, et l'ecart."""

    y: sp.Expr
    v: sp.Expr
    u: sp.Expr
    ecart: sp.Expr


def boucle_un(alpha: object = ALPHA, beta: object = BETA) -> Boucle:
    """L'exercice 1 : avance de phase, actionneur, procede integrateur."""
    return Boucle(
        correcteur=(1 + alpha * P) / (1 + beta * P),
        actionneur=2 / (1 + 3 * P),
        procede=5 / (P * (1 + 10 * P)),
        capteur=CAPTEUR,
    )


def regime_un(*, consigne: object, perturbation: object) -> Regime:
    """L'integrateur du procede ne recoit rien : `v = p`, et l'on remonte."""
    v = sp.sympify(perturbation)
    u = v / 2
    ecart = u
    return Regime(y=(sp.sympify(consigne) - ecart) / CAPTEUR, v=v, u=u, ecart=ecart)


def sortie_finale_un(*, consigne: object, perturbation: object) -> sp.Expr:
    """`lim p Y(p)` quand `p` tend vers zero, par les deux transmittances."""
    image = boucle_un().sortie(sp.sympify(consigne) / P, sp.sympify(perturbation) / P)
    return sp.limit(sp.cancel(P * image), P, 0)


def est_stable_un(*, alpha: object, beta: object) -> bool:
    """Vrai si la boucle de l'exercice 1 est stable pour ce correcteur."""
    return boucle_un(alpha, beta).est_stable()


def regime_deux(*, consigne: object, perturbation: object) -> Regime:
    """L'integrateur du correcteur ne recoit rien : `eps = 0`, et l'on descend."""
    y = sp.sympify(consigne) / CAPTEUR
    v = y / PROCEDE_DEUX.subs(P, 0) + sp.sympify(perturbation)
    return Regime(y=y, v=v, u=v / 5, ecart=sp.Integer(0))


def marge_de_phase_deux() -> float:
    """La marge de phase de la boucle de l'exercice 2, retard compris."""
    ouverte = CAPTEUR * CORRECTEUR_DEUX * 5 * PROCEDE_DEUX * sp.exp(-P)
    return marges_de_stabilite(ouverte).de_phase
