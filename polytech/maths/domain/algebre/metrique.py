"""Produit scalaire, norme, distance, orthogonalite -- chapitre IX du syllabus.

Rien ne portait ce chapitre. Le syllabus pose le produit scalaire par
**trois axiomes** -- positivite et definition, linearite en le premier
argument, symetrie -- puis en deduit tout le reste : « une fois le
produit scalaire choisi, on utilisera la norme qui en decoule », et « la
distance entre un vecteur `X` et un vecteur `Y` est la norme de
`X - Y` ». L'ordre compte : rien ici n'est pose independamment.

Le point le plus instructif est la **projection orthogonale**, que le
syllabus definit deux fois :

* « le seul vecteur `Y` de `W` tel que `X - Y` soit orthogonal a tout
  `Z` de `W` » ;
* « parmi tous les vecteurs `Z` de `W`, celui qui minimise la distance
  `||X - Z||` ».

Les deux coincident, et c'est un theoreme -- « on dit que `proj_W X` est
la meilleure approximation de `X` dans `W` ». Les tests exigent les deux
proprietes, la seconde en echantillonnant d'autres vecteurs du
sous-espace.

Deux refus plutot que deux resultats faux. Gram-Schmidt exige une famille
**libre** -- le syllabus le precise -- parce que sur une famille liee le
procede produit un vecteur nul. Et la formule de projection exige une
base **orthogonale** : l'appliquer a une base quelconque rendrait un
vecteur qui n'est pas la projection, sans rien signaler.
"""

from __future__ import annotations

import math

TOLERANCE = 1e-10
"""En dessous, un produit scalaire est tenu pour nul."""


def produit_scalaire(x: tuple[float, ...], y: tuple[float, ...]) -> float:
    """`<X|Y>` : le produit scalaire canonique de l'espace reel."""
    if len(x) != len(y):
        msg = f"dimensions differentes : {len(x)} et {len(y)}"
        raise ValueError(msg)
    return sum(a * b for a, b in zip(x, y, strict=True))


def norme(x: tuple[float, ...]) -> float:
    """`||X|| = racine(<X|X>)` : la norme qui decoule du produit scalaire."""
    return math.sqrt(produit_scalaire(x, x))


def distance(x: tuple[float, ...], y: tuple[float, ...]) -> float:
    """`d(X, Y) = ||X - Y||`, la definition du syllabus."""
    return norme(tuple(a - b for a, b in zip(x, y, strict=True)))


def est_orthogonale(famille: tuple[tuple[float, ...], ...]) -> bool:
    """Vrai si les vecteurs sont orthogonaux deux a deux."""
    return all(
        abs(produit_scalaire(famille[i], famille[j])) < TOLERANCE
        for i in range(len(famille))
        for j in range(i + 1, len(famille))
    )


def est_orthonormale(famille: tuple[tuple[float, ...], ...]) -> bool:
    """Orthogonale, et tous ses vecteurs unitaires."""
    return est_orthogonale(famille) and all(
        abs(norme(v) - 1) < TOLERANCE for v in famille
    )


def gram_schmidt(
    famille: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """La famille orthogonale construite depuis une famille libre.

    Le premier vecteur est conserve tel quel : c'est ce qui rend le
    procede reproductible, l'ordre de depart determinant le resultat.
    """
    obtenus: list[tuple[float, ...]] = []
    for vecteur in famille:
        restant = list(vecteur)
        for deja in obtenus:
            facteur = produit_scalaire(vecteur, deja) / produit_scalaire(deja, deja)
            restant = [a - facteur * b for a, b in zip(restant, deja, strict=True)]
        if norme(tuple(restant)) < TOLERANCE:
            msg = "famille liee : le procede y produit un vecteur nul"
            raise ValueError(msg)
        obtenus.append(tuple(restant))
    return tuple(obtenus)


def projection_orthogonale(
    x: tuple[float, ...],
    base: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    """La projection de `X` sur l'espace engendre par une base orthogonale."""
    if not est_orthogonale(base):
        msg = "base non orthogonale : la formule de projection ne s'y applique pas"
        raise ValueError(msg)
    projection = [0.0] * len(x)
    for vecteur in base:
        facteur = produit_scalaire(x, vecteur) / produit_scalaire(vecteur, vecteur)
        projection = [p + facteur * c for p, c in zip(projection, vecteur, strict=True)]
    return tuple(projection)


def distance_au_sous_espace(
    x: tuple[float, ...],
    base: tuple[tuple[float, ...], ...],
) -> float:
    """`d(X, W) = ||X - proj_W X||` : la meilleure approximation possible."""
    return distance(x, projection_orthogonale(x, base))
