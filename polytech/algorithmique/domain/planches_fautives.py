"""Deux fonctions des planches, recopiees instruction pour instruction.

Leur modele sert a montrer ce qu'elles font vraiment : la planche 37 du S3
(`Doublon`) laisse passer des doublons, la boucle de la planche 62
(insertion dans un tableau) ouvre la place un rang trop loin.
"""

from __future__ import annotations


def doublon_de_la_planche_37[T](elements: list[T]) -> list[T]:
    """`Doublon` : efface l'element `j` puis passe au suivant sans le reexaminer."""
    t, n = list(elements), len(elements)
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            if t[i] == t[j]:
                for k in range(j, n - 1):
                    t[k] = t[k + 1]
                n -= 1
            j += 1
        i += 1
    return t[:n]


def insertion_de_la_planche_62[T](memoire: list[T], n: int, p: int) -> list[T]:
    """`for (i = N; i > P; i--) Tab[i+1] = Tab[i];` sur une copie de la memoire.

    Leve `IndexError` si `Tab[N+1]` sort de la memoire : C++ ecrirait hors du
    tableau, le modele refuse de le taire.
    """
    if n + 1 >= len(memoire):
        msg = f"Tab[{n + 1}] hors de la memoire de {len(memoire)} cases"
        raise IndexError(msg)
    copie = list(memoire)
    for i in range(n, p, -1):
        copie[i + 1] = copie[i]
    return copie
