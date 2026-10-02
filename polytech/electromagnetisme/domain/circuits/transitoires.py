"""Les etats quasi-stationnaires : regimes transitoires du premier ordre.

Les exercices de ce chapitre suivent tous le meme chemin, et c'est lui
qu'il faut exposer : etablir l'**etat initial** avant la commutation,
poser les equations apres, et resoudre. La transcription de l'exercice 6
le montre bien -- avant la fermeture, `i_1 = i_2 = 0` et
`u_C1 = E_2 - E_1`, et ces trois valeurs sont les conditions initiales de
tout ce qui suit.

Deux continuites gouvernent l'etat initial, et les confondre est l'erreur
du chapitre : le **courant** dans une inductance est continu, la
**tension** aux bornes d'un condensateur l'est aussi. Ce ne sont pas les
memes grandeurs, et chacune porte sa propre inertie.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS


def constante_de_temps_rl(resistance: float, inductance: float) -> float:
    """`tau = L / R`."""
    if resistance <= 0 or inductance <= 0:
        msg = f"R = {resistance} ohm et L = {inductance} H doivent etre positifs"
        raise ValueError(msg)
    return inductance / resistance


def constante_de_temps_rc(resistance: float, capacite: float) -> float:
    """`tau = R C`."""
    if resistance <= 0 or capacite <= 0:
        msg = f"R = {resistance} ohm et C = {capacite} F doivent etre positifs"
        raise ValueError(msg)
    return resistance * capacite


def reponse_du_premier_ordre(
    valeur_initiale: sp.Expr,
    valeur_finale: sp.Expr,
    constante: sp.Expr,
) -> sp.Expr:
    """`y(t) = y_inf + (y_0 - y_inf) exp(-t / tau)`.

    Une seule formule pour tous les premiers ordres du chapitre, quelle
    que soit la grandeur : c'est la forme generale, et elle rend inutile
    de la redemontrer a chaque exercice.
    """
    return valeur_finale + (valeur_initiale - valeur_finale) * sp.exp(
        -TEMPS / constante
    )


def satisfait_son_equation(
    reponse: sp.Expr,
    constante: sp.Expr,
    valeur_finale: sp.Expr,
) -> bool:
    """Vrai si la reponse verifie `tau y' + y = y_inf`.

    Le controle intrinseque : la formule generale n'est pas crue, elle est
    reinjectee dans l'equation differentielle dont elle pretend etre la
    solution.
    """
    reste = constante * sp.diff(reponse, TEMPS) + reponse - valeur_finale
    return bool(sp.simplify(reste) == 0)


def amortissement_rlc(
    resistance: float,
    inductance: float,
    capacite: float,
) -> sp.Expr:
    """`zeta = (R/2) racine(C/L)` : le facteur d'amortissement du second ordre.

    Le type de retour est symbolique et non flottant : la racine d'un
    rationnel reste exacte, et un corrige qui l'afficherait en flottant
    perdrait la forme que le lecteur reconnait.
    """
    if inductance <= 0 or capacite <= 0:
        msg = f"L = {inductance} H et C = {capacite} F doivent etre positifs"
        raise ValueError(msg)
    return resistance / 2 * sp.sqrt(capacite / inductance)


def pulsation_propre(inductance: float, capacite: float) -> sp.Expr:
    """`omega_0 = 1 / racine(L C)`, sous forme exacte."""
    if inductance <= 0 or capacite <= 0:
        msg = f"L = {inductance} H et C = {capacite} F doivent etre positifs"
        raise ValueError(msg)
    return 1 / sp.sqrt(inductance * capacite)


def instant_du_seuil(
    valeur_finale: sp.Expr,
    constante: sp.Expr,
    seuil: sp.Expr,
) -> sp.Expr:
    """L'instant ou une montee du premier ordre, partie de zero, vaut `seuil`.

    `t = -tau ln(1 - s / y_inf)`. La forme fermee evite de resoudre une
    equation transcendante, et elle dit ce que l'equation cache : la
    montee n'atteint jamais sa valeur finale, donc un seuil qui l'egale
    ou la depasse n'a pas d'instant.
    """
    if not (seuil > 0):
        msg = f"seuil nul ou negatif : {seuil}"
        raise ValueError(msg)
    if not (seuil < valeur_finale):
        msg = f"seuil jamais atteint : {seuil} pour une valeur finale {valeur_finale}"
        raise ValueError(msg)
    return -constante * sp.log(1 - seuil / valeur_finale)
