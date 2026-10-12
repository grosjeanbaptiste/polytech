"""Les exercices du chapitre X qui ne sont pas une reduction : 2 a 4, 12 a 18.

Chacun se resout par une propriete plutot que par le polynome
caracteristique : une matrice triangulaire porte ses valeurs propres sur sa
diagonale, la trace est leur somme et le determinant leur produit, une
matrice diagonalisable s'eleve a une puissance par sa diagonale.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import recueil_algebre_ch10_enonces as enonces
from polytech.maths.domain.algebre.diagonalisation import (
    diagonaliser_orthogonalement,
    puissance,
    sous_espace_propre,
)

K = sp.Symbol("k", integer=True, positive=True)
N = sp.Symbol("n", integer=True, positive=True)
TEMPS = sp.Symbol("t", real=True)
Base = tuple[tuple[sp.Expr, ...], ...]


def valeur_associee(matrice: sp.Matrix, vecteur: sp.Matrix) -> sp.Expr:
    """Le `lambda` tel que `A v = lambda v` ; leve si `v` n'est pas propre."""
    image = matrice * vecteur
    rapports = [image[i] / terme for i, terme in enumerate(vecteur) if terme != 0]
    if not rapports or image != rapports[0] * vecteur:
        msg = f"{tuple(vecteur)} n'est pas un vecteur propre de la matrice"
        raise ValueError(msg)
    return rapports[0]


def exercice_2() -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    """La diagonale d'une matrice triangulaire, et le premier vecteur canonique."""
    triangulaire = enonces.MATRICES[2]
    return tuple(triangulaire.diagonal()), tuple(sp.eye(triangulaire.rows)[:, 0])


def exercice_3() -> sp.Expr:
    """La valeur propre a laquelle `(1, 1, 1)` est associe."""
    return valeur_associee(enonces.MATRICES[3], enonces.VECTEUR_3)


def exercice_4() -> tuple[int, bool]:
    """`m_g(0) = n - rg A`, et si `m_a(0)` peut valoir 2."""
    geometrique = enonces.ORDRE_4 - enonces.RANG_4
    return geometrique, geometrique <= enonces.MULTIPLICITE_4


def exercice_12() -> tuple[sp.Matrix, sp.Matrix]:
    """`(P, D)` avec `P` orthogonale et `P^t A P = D`."""
    return diagonaliser_orthogonalement(enonces.MATRICES[12])


def exercice_13() -> sp.Matrix:
    """`A^k = P D^k P^-1`."""
    return sp.simplify(puissance(enonces.MATRICES[13], K))


def exercice_14() -> tuple[sp.Expr, tuple[sp.Expr, ...], sp.Matrix]:
    """La valeur propre du vecteur donne, les deux autres, la diagonale."""
    carree = enonces.MATRICES[14]
    connue, du_vecteur = enonces.VALEUR_14, valeur_associee(carree, enonces.VECTEUR_14)
    x, y = sp.symbols("x y")
    somme = connue + du_vecteur + x + y - carree.trace()
    produit = connue * du_vecteur * x * y - enonces.DETERMINANT_14
    autres = tuple(sorted(sp.solve([somme, produit], [x, y], dict=True)[0].values()))
    return du_vecteur, autres, sp.diag(connue, autres[0], du_vecteur, autres[1])


def exercice_15() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """`P`, `D` de `A^2`, et la racine a valeurs propres positives `P sqrt(D) P^-1`."""
    passage, diagonale = enonces.CARRE_15.diagonalize()
    racine = passage * diagonale.applyfunc(sp.sqrt) * passage.inv()
    return passage, diagonale, racine


def exercice_16() -> tuple[sp.Matrix, sp.Expr, Base, Base]:
    """`A^2`, la trace, puis des bases de `S(1)` et de `S(-1)`."""
    carree = enonces.MATRICES[16]
    plus, moins = sous_espace_propre(carree, 1), sous_espace_propre(carree, -1)
    return carree**2, carree.trace(), plus, moins


def exercice_17() -> tuple[sp.Matrix, tuple[tuple[sp.Expr, sp.Expr], ...]]:
    """La matrice de la derivation dans `{sin t, cos t}`, et ses fonctions propres."""
    base = (sp.sin(TEMPS), sp.cos(TEMPS))
    derivees = [sp.diff(fonction, TEMPS) for fonction in base]
    matrice = sp.Matrix(2, 2, lambda i, j: derivees[j].coeff(base[i]))
    couples = []
    for valeur, _, (vecteur,) in matrice.eigenvects():
        fonction = vecteur[0] * base[0] + vecteur[1] * base[1]
        couples.append((valeur, sp.expand(fonction / fonction.subs(TEMPS, 0))))
    return matrice, tuple(couples)


def exercice_18() -> tuple[sp.Matrix, Base, Base]:
    """La projection sur le plan parallelement a la droite, `S(0)` et `S(1)`."""
    normale, direction = enonces.NORMALE_18, enonces.DIRECTION_18
    projection = sp.eye(3) - direction * normale.T / normale.dot(direction)
    return (
        projection,
        sous_espace_propre(projection, 0),
        sous_espace_propre(projection, 1),
    )


def exercice_23() -> sp.Matrix:
    """`A^n` pour `a = 2`, ou la matrice est diagonalisable."""
    return sp.simplify(puissance(enonces.MATRICES[23].subs(enonces.A, 2), N))
