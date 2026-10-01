"""Les circuits a courants sinusoidaux : impedances et resonance.

Le chapitre remplace les equations differentielles par des **nombres
complexes**, et c'est tout son interet : une derivation devient une
multiplication par `j omega`. Les impedances s'ecrivent donc une fois ici,
et les lois de Kirchhoff s'appliquent telles quelles.

La resonance serie et la resonance parallele n'ont pas la meme pulsation
des que la resistance n'est plus negligeable, et les confondre est
l'erreur du chapitre. La resonance serie, elle, est exacte : le module de
l'impedance y est minimal et son argument nul, et c'est cette double
caracterisation qui sert de verification.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.symboles import PULSATION


def impedance_resistive(resistance: sp.Expr) -> sp.Expr:
    """`Z = R` : reelle, independante de la pulsation."""
    return sp.sympify(resistance)


def impedance_inductive(inductance: sp.Expr) -> sp.Expr:
    """`Z = j omega L` : la derivation devenue multiplication."""
    return sp.I * PULSATION * inductance


def impedance_capacitive(capacite: sp.Expr) -> sp.Expr:
    """`Z = 1 / (j omega C)` ; leve pour une capacite nulle."""
    if sp.simplify(capacite) == 0:
        msg = "capacite nulle : son impedance serait infinie a toute pulsation"
        raise ValueError(msg)
    return 1 / (sp.I * PULSATION * capacite)


def en_serie(*impedances: sp.Expr) -> sp.Expr:
    """Les impedances en serie s'additionnent."""
    if not impedances:
        msg = "aucune impedance a mettre en serie"
        raise ValueError(msg)
    return sp.simplify(sum(impedances))


def en_parallele(*impedances: sp.Expr) -> sp.Expr:
    """Les admittances s'additionnent : `1/Z = somme 1/Z_i`."""
    if not impedances:
        msg = "aucune impedance a mettre en parallele"
        raise ValueError(msg)
    return sp.simplify(1 / sum(1 / impedance for impedance in impedances))


def pulsation_de_resonance_serie(inductance: sp.Expr, capacite: sp.Expr) -> sp.Expr:
    """`omega = 1 / racine(L C)` : la pulsation ou l'impedance devient reelle."""
    return 1 / sp.sqrt(inductance * capacite)


def est_resonant(impedance: sp.Expr, pulsation: sp.Expr) -> bool:
    """Vrai si l'impedance est reelle a cette pulsation.

    C'est la definition de la resonance, et elle se verifie : la partie
    imaginaire doit s'annuler. Chercher un minimum de module serait plus
    fragile -- et moins fidele a ce que le cours enonce.
    """
    valeur = sp.simplify(impedance.subs(PULSATION, pulsation))
    return bool(sp.simplify(sp.im(valeur)) == 0)


def facteur_de_qualite(
    resistance: sp.Expr,
    inductance: sp.Expr,
    capacite: sp.Expr,
) -> sp.Expr:
    """`Q = (1/R) racine(L/C)` pour un circuit serie."""
    if sp.simplify(resistance) == 0:
        msg = "resistance nulle : le facteur de qualite serait infini"
        raise ValueError(msg)
    return sp.sqrt(inductance / capacite) / resistance
