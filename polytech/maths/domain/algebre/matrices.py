"""Les operations sur les matrices -- chapitre II du syllabus d'algebre.

II.1 Definitions, II.2 Operations -- addition, multiplication par un
scalaire, produit de deux matrices -- et II.3 Transposee. Rien ne les
portait : la bibliotheque avait les systemes, les determinants et le
spectre, mais pas les operations elles-memes.

Deux pieges, et les deux viennent de ce que ces operations paraissent
trop simples pour en avoir.

**La multiplication par un scalaire et le determinant.** Multiplier une
matrice `n x n` par `k` multiplie son determinant par `k^n`, non par
`k` : chaque ligne est multipliee, et le determinant est multilineaire.

**La transposee d'un produit inverse l'ordre** -- `(A B)^T = B^T A^T`.
C'est ce qu'on oublie, et sur des matrices carrees l'erreur passe
inapercue parce que le format reste bon.
"""

from __future__ import annotations

Matrice = tuple[tuple[float, ...], ...]


def _format(matrice: Matrice) -> tuple[int, int]:
    """Les dimensions, apres verification de la regularite."""
    if not matrice or not matrice[0]:
        msg = "matrice vide : ce n'est pas un cas du cours"
        raise ValueError(msg)
    colonnes = len(matrice[0])
    if any(len(ligne) != colonnes for ligne in matrice):
        msg = "lignes de longueurs differentes : ce n'est pas une matrice"
        raise ValueError(msg)
    return len(matrice), colonnes


def addition(gauche: Matrice, droite: Matrice) -> Matrice:
    """La somme terme a terme, definie seulement a format egal."""
    if _format(gauche) != _format(droite):
        msg = f"formats differents : {_format(gauche)} et {_format(droite)}"
        raise ValueError(msg)
    return tuple(
        tuple(a + b for a, b in zip(lg, ld, strict=True))
        for lg, ld in zip(gauche, droite, strict=True)
    )


def multiplication_par_un_scalaire(scalaire: float, matrice: Matrice) -> Matrice:
    """Chaque terme multiplie -- et le determinant par `k^n`, non par `k`."""
    _format(matrice)
    return tuple(tuple(scalaire * a for a in ligne) for ligne in matrice)


def produit(gauche: Matrice, droite: Matrice) -> Matrice:
    """`(A B)_ij = somme_k A_ik B_kj`. Non commutatif."""
    _, colonnes_gauche = _format(gauche)
    lignes_droite, colonnes_droite = _format(droite)
    if colonnes_gauche != lignes_droite:
        msg = (
            f"dimensions incompatibles : {colonnes_gauche} colonnes a gauche "
            f"pour {lignes_droite} lignes a droite"
        )
        raise ValueError(msg)
    return tuple(
        tuple(
            sum(ligne[k] * droite[k][j] for k in range(colonnes_gauche))
            for j in range(colonnes_droite)
        )
        for ligne in gauche
    )


def transposee(matrice: Matrice) -> Matrice:
    """Lignes et colonnes echangees. Involution, et `(A B)^T = B^T A^T`."""
    _format(matrice)
    return tuple(zip(*matrice, strict=True))


def identite(ordre: int) -> Matrice:
    """La matrice unite d'ordre `n`, neutre pour le produit des deux cotes."""
    if ordre <= 0:
        msg = f"ordre nul ou negatif : {ordre}"
        raise ValueError(msg)
    return tuple(
        tuple(1.0 if i == j else 0.0 for j in range(ordre)) for i in range(ordre)
    )
