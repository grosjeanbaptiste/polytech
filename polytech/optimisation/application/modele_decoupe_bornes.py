"""Modelisation, probleme II : pourquoi 28 rouleaux et 56 decoupes sont optimaux.

La planche 11 donne deux solutions et les dit optimales. Deux preuves :
une borne tiree de la relaxation, arrondie, et un decompte direct.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.application.modele_decoupe import (
    COMMANDES,
    LARGEURS,
    MOTIFS,
    ROULEAU,
    decoupes,
    perte,
    relaxation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from polytech.optimisation.application.modele_decoupe import Motif

Bilan = tuple[int, tuple[int, ...], sp.Expr, int]


def bilan(solution: Mapping[int, int]) -> Bilan:
    """Rouleaux, pieces livrees par largeur, perte et decoupes d'une solution."""
    lignes = [(MOTIFS[rang - 1], nombre) for rang, nombre in solution.items()]
    livrees = tuple(sum(m[k] * n for m, n in lignes) for k in range(len(LARGEURS)))
    chute = sum((perte(m) * n for m, n in lignes), sp.Integer(0))
    return (
        sum(solution.values()),
        livrees,
        chute,
        sum(decoupes(m) * n for m, n in lignes),
    )


def largeur_commandee() -> sp.Expr:
    """La largeur totale des pieces a livrer."""
    return sum(
        (n * cote for n, cote in zip(COMMANDES, LARGEURS, strict=True)), sp.Integer(0)
    )


def perte_de(rouleaux: int) -> sp.Expr:
    """La perte quand la commande est livree exactement avec ce nombre de rouleaux."""
    return rouleaux * ROULEAU - largeur_commandee()


def rouleaux_au_moins() -> int:
    """La relaxation, arrondie vers le haut : on ne coupe pas un demi-rouleau."""
    return int(sp.ceiling(relaxation("rouleaux")))


def perte_au_moins() -> sp.Expr:
    """La perte du plus petit nombre de rouleaux possible."""
    return perte_de(rouleaux_au_moins())


def decoupes_au_moins() -> int:
    """La relaxation des decoupes, arrondie vers le haut."""
    return int(sp.ceiling(relaxation("decoupes")))


def grandes_pieces() -> int:
    """Les pieces commandees de plus du tiers d'un rouleau."""
    return sum(
        n for n, cote in zip(COMMANDES, LARGEURS, strict=True) if 3 * cote > ROULEAU
    )


def grandes_pieces_par_rouleau() -> int:
    """Le plus de grandes pieces qu'un meme rouleau puisse porter."""
    grandes = [k for k, cote in enumerate(LARGEURS) if 3 * cote > ROULEAU]
    return max(sum(m[k] for k in grandes) for m in MOTIFS)


def motifs_sans_chute() -> tuple[Motif, ...]:
    """Les motifs qui consomment le rouleau entier."""
    return tuple(m for m in MOTIFS if perte(m) == 0)


def rouleaux_sans_chute_au_plus() -> int:
    """Le plus de rouleaux sans chute que la commande permette de couper."""
    motifs = motifs_sans_chute()
    plafonds = [
        min(c // n for c, n in zip(COMMANDES, m, strict=True) if n) for m in motifs
    ]
    meilleur = 0
    for nombres in itertools.product(*(range(p + 1) for p in plafonds)):
        pieces = [
            sum(n * m[k] for n, m in zip(nombres, motifs, strict=True))
            for k in range(len(LARGEURS))
        ]
        if all(p <= c for p, c in zip(pieces, COMMANDES, strict=True)):
            meilleur = max(meilleur, sum(nombres))
    return meilleur
