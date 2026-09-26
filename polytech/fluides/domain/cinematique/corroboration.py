"""Recoupement numerique des resultats symboliques (cours III).

La partie I du corrige se compare a la reponse publiee par le cours. La
partie II n'avait aucun recoupement : lignes de courant, trajectoires et
accelerations sortaient toutes du meme moteur symbolique -- donc d'une seule
facon de se tromper.

Ce module integre les memes equations par Runge-Kutta d'ordre 4, methode
sans aucun rapport avec la resolution formelle, et mesure l'ecart. Deux
chemins independants qui se rejoignent valent mieux qu'un seul verifie deux
fois.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T, X, Y

if TYPE_CHECKING:
    from collections.abc import Callable

    from polytech.fluides.domain.cinematique.champ import ChampDeVitesse

_Etat = tuple[float, float]


def _exiger_pas_positif(pas: float) -> None:
    if pas <= 0:
        msg = f"pas d'integration non strictement positif : {pas}"
        raise ValueError(msg)


def _pas_de_runge_kutta(
    derivee: Callable[[float, _Etat], _Etat],
    variable: float,
    etat: _Etat,
    pas: float,
) -> _Etat:
    """Un pas de RK4 sur un systeme a deux inconnues."""
    k1 = derivee(variable, etat)
    k2 = derivee(
        variable + pas / 2, (etat[0] + pas * k1[0] / 2, etat[1] + pas * k1[1] / 2)
    )
    k3 = derivee(
        variable + pas / 2, (etat[0] + pas * k2[0] / 2, etat[1] + pas * k2[1] / 2)
    )
    k4 = derivee(variable + pas, (etat[0] + pas * k3[0], etat[1] + pas * k3[1]))
    return (
        etat[0] + pas * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
        etat[1] + pas * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
    )


def _vitesse_numerique(champ: ChampDeVitesse) -> Callable[[float, _Etat], _Etat]:
    u = sp.lambdify((X, Y, T), champ.u, "math")
    v = sp.lambdify((X, Y, T), champ.v, "math")
    return lambda t, etat: (u(etat[0], etat[1], t), v(etat[0], etat[1], t))


def trajectoire_numerique(
    champ: ChampDeVitesse,
    depart: _Etat,
    duree: float,
    pas: float,
) -> tuple[tuple[float, _Etat], ...]:
    """Integre ``dx/dt = u``, ``dy/dt = v`` par RK4, depuis ``t = 0``."""
    _exiger_pas_positif(pas)
    derivee = _vitesse_numerique(champ)
    instant, etat, releves = 0.0, depart, [(0.0, depart)]
    while instant < duree:
        etat = _pas_de_runge_kutta(derivee, instant, etat, pas)
        instant += pas
        releves.append((instant, etat))
    return tuple(releves)


def ecart_a_la_trajectoire_symbolique(
    champ: ChampDeVitesse,
    depart: _Etat,
    duree: float,
    pas: float,
) -> float:
    """Plus grand ecart entre l'integration numerique et la solution formelle."""
    releves = trajectoire_numerique(champ, depart, duree, pas)
    x_formel, y_formel = champ.trajectoire(*depart)
    x_numerique = sp.lambdify(T, x_formel, "math")
    y_numerique = sp.lambdify(T, y_formel, "math")
    ecarts = [
        max(abs(etat[0] - float(x_numerique(t))), abs(etat[1] - float(y_numerique(t))))
        for t, etat in releves
    ]
    return max(ecarts)


def variation_de_l_invariant(
    champ: ChampDeVitesse,
    depart: _Etat,
    duree: float,
    pas: float,
) -> float:
    """Variation relative de ``x y`` le long de la trajectoire numerique.

    Si elle reste nulle, la particule ne quitte pas l'hyperbole ``xy = C`` --
    ce que l'equation des lignes de courant affirme, obtenue tout autrement.
    """
    releves = trajectoire_numerique(champ, depart, duree, pas)
    initial = depart[0] * depart[1]
    return max(abs(etat[0] * etat[1] - initial) / abs(initial) for _, etat in releves)


def ecart_aux_lignes_de_courant(
    champ: ChampDeVitesse,
    instant: float,
    depart: _Etat,
    arrivee: float,
    pas: float,
) -> float:
    """Integre ``dy/dx = v/u`` a instant fige et compare a la forme close."""
    _exiger_pas_positif(pas)
    pente = sp.lambdify((X, Y), (champ.v / champ.u).subs(T, instant), "math")
    formelle = champ.lignes_de_courant_explicites().subs(T, instant)
    constante = sp.solve(
        sp.Eq(formelle.subs(X, depart[0]), depart[1]),
        sp.Symbol("C"),
    )[0]
    close = sp.lambdify(X, formelle.subs(sp.Symbol("C"), constante), "math")

    abscisse, ordonnee, ecart = depart[0], depart[1], 0.0
    while abscisse < arrivee:
        k1 = pente(abscisse, ordonnee)
        k2 = pente(abscisse + pas / 2, ordonnee + pas * k1 / 2)
        k3 = pente(abscisse + pas / 2, ordonnee + pas * k2 / 2)
        k4 = pente(abscisse + pas, ordonnee + pas * k3)
        ordonnee += pas * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        abscisse += pas
        ecart = max(ecart, abs(ordonnee - close(abscisse)))
    return ecart
