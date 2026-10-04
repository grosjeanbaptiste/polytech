"""Chapitre II.5.2 : le probleme augmente `P^a`, et sa base de depart.

La phase II suppose une base admissible initiale (hypothese H2). Quand
toutes les contraintes sont des inegalites `<=` a second membre positif,
les variables d'ecart la fournissent : leurs colonnes forment une matrice
unite. Sinon, on ajoute une **variable artificielle** non negative au
premier membre de chaque contrainte qui n'a pas sa colonne unite -- une
seule suffit parfois (page 63) -- et on la penalise par un cout `M`, de
sorte que l'algorithme cherche d'abord a l'annuler.

Le cours suppose `d >= 0`, « ce qui n'est pas restrictif » : une
equation a second membre negatif se multiplie par `-1`.
"""

from __future__ import annotations

import sympy as sp

from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.formes import FormeStandard
from polytech.optimisation.domain.grand_m import GRAND_M


def a_seconds_membres_non_negatifs(forme: FormeStandard) -> FormeStandard:
    """Le meme systeme, chaque equation de second membre negatif retournee."""
    matrice, seconds = forme.matrice.copy(), forme.seconds_membres.copy()
    for ligne in range(matrice.rows):
        if seconds[ligne, 0] < 0:
            matrice[ligne, :] = -matrice[ligne, :]
            seconds[ligne, 0] = -seconds[ligne, 0]
    return FormeStandard(forme.couts, matrice, seconds)


def base_evidente(forme: FormeStandard) -> dict[int, int]:
    """Pour chaque contrainte qui en a une, la variable de sa colonne unite.

    Contraintes et variables sont numerotees a partir de 1, comme dans le
    cours.
    """
    trouvees: dict[int, int] = {}
    for colonne in range(forme.matrice.cols):
        non_nuls = [
            i for i in range(forme.matrice.rows) if forme.matrice[i, colonne] != 0
        ]
        if len(non_nuls) == 1 and forme.matrice[non_nuls[0], colonne] == 1:
            trouvees.setdefault(non_nuls[0] + 1, colonne + 1)
    return trouvees


def probleme_augmente(
    forme: FormeStandard,
    sens: Optimisation,
) -> tuple[FormeStandard, tuple[int, ...]]:
    """`P^a` et sa base admissible initiale, une variable artificielle par manque."""
    positive = a_seconds_membres_non_negatifs(forme)
    evidente = base_evidente(positive)
    penalite = GRAND_M if sens is Optimisation.MIN else -GRAND_M
    matrice, couts = positive.matrice, positive.couts
    base = []
    for contrainte in range(1, matrice.rows + 1):
        if contrainte in evidente:
            base.append(evidente[contrainte])
            continue
        unite = sp.zeros(matrice.rows, 1)
        unite[contrainte - 1, 0] = 1
        matrice, couts = (
            matrice.row_join(unite),
            couts.row_join(sp.Matrix([[penalite]])),
        )
        base.append(matrice.cols)
    return FormeStandard(couts, matrice, positive.seconds_membres), tuple(base)


def reduire_a_une_artificielle(forme: FormeStandard) -> FormeStandard:
    """Remarque 3 : le systeme equivalent ou une seule ligne demande une artificielle.

    Quand les contraintes sont des inegalites, leurs ecarts forment une
    base, inadmissible la ou `d_i < 0`. Soit `d_p` le plus negatif : on
    soustrait la ligne `p` des autres lignes negatives, dont le second
    membre `d_i - d_p` devient positif, puis on retourne la ligne `p`.
    Seule celle-ci a perdu sa colonne unite.
    """
    negatives = [
        i for i in range(forme.matrice.rows) if forme.seconds_membres[i, 0] < 0
    ]
    if not negatives:
        return forme
    pire = min(negatives, key=lambda i: forme.seconds_membres[i, 0])
    matrice, seconds = forme.matrice.copy(), forme.seconds_membres.copy()
    for ligne in negatives:
        if ligne != pire:
            matrice[ligne, :] = forme.matrice[ligne, :] - forme.matrice[pire, :]
            seconds[ligne, 0] = (
                forme.seconds_membres[ligne, 0] - forme.seconds_membres[pire, 0]
            )
    matrice[pire, :] = -forme.matrice[pire, :]
    seconds[pire, 0] = -forme.seconds_membres[pire, 0]
    return FormeStandard(forme.couts, matrice, seconds)
