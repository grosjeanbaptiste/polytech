"""Trois calculs que les boucles du syllabus servent a mener.

Les exercices du chapitre 3 (« Les structures de controle ») demandent
d'ecrire une boucle dont chaque tour s'appuie sur les precedents :

* les **nombres premiers** (3.5) : un entier est premier si aucun nombre
  premier deja trouve, jusqu'a sa racine, ne le divise ;
* la suite de **Fibonacci** (3.12), `U1 = U2 = 1`, `Un = Un-1 + Un-2`, et
  « le rang N et la valeur UN du terme maximal que l'on peut calculer »
  dans un type donne : au-dela, l'entier deborde ;
* la **serie** (3.13) `1 - X.3!/(1.2) + X^2.5!/(2.4) - X^3.7!/(3.8) ...`,
  ou « le calcul du Nieme terme se fera a partir du N-1 » : on passe d'un
  terme au suivant en le multipliant par `-X (2n+1)(n-1)` -- le rapport
  de `X^n (2n+1)! / (n 2^n)` a son predecesseur, une fois simplifie.
"""

from __future__ import annotations

from typing import Any

from polytech.algorithmique.domain.types_cpp import TYPES_ENTIERS


def nombres_premiers(nombre: int) -> tuple[int, ...]:
    """Les `nombre` premiers nombres premiers, dans l'ordre."""
    if nombre < 0:
        msg = f"nombre negatif de nombres premiers : {nombre}"
        raise ValueError(msg)
    premiers: list[int] = []
    candidat = 2
    while len(premiers) < nombre:
        if all(candidat % p for p in premiers if p * p <= candidat):
            premiers.append(candidat)
        candidat += 1
    return tuple(premiers)


def dernier_fibonacci_au_plus(maximum: int) -> tuple[int, int]:
    """Le rang et la valeur du plus grand terme qui ne depasse pas la borne."""
    if maximum < 1:
        msg = f"borne inferieure a 1 : {maximum}"
        raise ValueError(msg)
    rang, terme, suivant = 1, 1, 1
    while suivant <= maximum:
        rang, terme, suivant = rang + 1, suivant, terme + suivant
    return rang, terme


def dernier_fibonacci_representable(nom_du_type: str) -> tuple[int, int]:
    """Le rang et la valeur du plus grand terme qui tient dans ce type entier."""
    if nom_du_type not in TYPES_ENTIERS:
        msg = f"type entier inconnu : {nom_du_type}"
        raise ValueError(msg)
    return dernier_fibonacci_au_plus(TYPES_ENTIERS[nom_du_type].maximum)


def sommes_partielles(x: Any, termes: int) -> tuple[Any, ...]:  # noqa: ANN401 -- entier, reel ou fraction
    """Les sommes de la serie apres un terme, deux termes, ..., `termes` termes."""
    if termes < 1:
        msg = f"au moins un terme : {termes} demande"
        raise ValueError(msg)
    sommes, terme = [1], -3 * x
    for n in range(2, termes + 1):
        sommes.append(sommes[-1] + terme)
        terme = -terme * x * (2 * n + 1) * (n - 1)
    return tuple(sommes)
