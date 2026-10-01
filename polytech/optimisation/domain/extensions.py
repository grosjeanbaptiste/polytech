"""Les extensions du simplexe : inverse de base, et variables bornees.

Deux sujets que les planches traitent ensemble.

La **forme revisee** ne recalcule pas le tableau entier a chaque
iteration : elle maintient `B^-1` et en deduit les colonnes au besoin. La
forme **produit** de cet inverse est ce qui rend la mise a jour peu
couteuse -- un pivot ne change qu'une colonne, donc `B^-1` se met a jour
par un seul facteur elementaire. L'identite `B_nouveau^-1 = E B^-1` est
exposee, et un test la verifie plutot que de la croire.

Les **variables bornees** `l <= x <= u` ne passent pas par des contraintes
supplementaires : ce serait doubler la taille du tableau. Elles se
traitent comme des bornes, en translatant la variable. La translation est
ici, une fois.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def inverse_de_base(etendue: sp.Matrix, base: tuple[int, ...]) -> sp.Matrix:
    """`B^-1` pour cette base ; leve si les colonnes sont dependantes."""
    sous_matrice = etendue[:, list(base)]
    if sous_matrice.det() == 0:
        msg = f"base singuliere : les colonnes {base} sont lineairement dependantes"
        raise ValueError(msg)
    return sp.Matrix(sous_matrice.inv())


def facteur_elementaire(colonne_transformee: sp.Matrix, rang: int) -> sp.Matrix:
    """La matrice `E` telle que `B_nouveau^-1 = E B^-1` apres un pivot.

    C'est la forme produit des planches : `E` est l'identite dont une
    seule colonne est remplacee, et c'est pourquoi la mise a jour coute
    une colonne et non une inversion.
    """
    pivot = colonne_transformee[rang, 0]
    if pivot == 0:
        msg = f"pivot nul a la ligne {rang} : le facteur elementaire n'existe pas"
        raise ValueError(msg)
    taille = colonne_transformee.rows
    facteur = sp.eye(taille)
    for ligne in range(taille):
        if ligne == rang:
            facteur[ligne, rang] = 1 / pivot
        else:
            facteur[ligne, rang] = -colonne_transformee[ligne, 0] / pivot
    return facteur


@dataclass(frozen=True, slots=True)
class VariableBornee:
    """Une variable `l <= x <= u`, ramenee a une variable positive bornee."""

    borne_basse: sp.Expr
    borne_haute: sp.Expr

    def __post_init__(self) -> None:
        """Refuse un intervalle vide : il n'y aurait aucune valeur admissible."""
        if sp.simplify(self.borne_haute - self.borne_basse) < 0:
            msg = (
                f"intervalle vide : borne haute {self.borne_haute} sous la "
                f"borne basse {self.borne_basse}"
            )
            raise ValueError(msg)

    @property
    def amplitude(self) -> sp.Expr:
        """`u - l`, la borne de la variable translatee."""
        return sp.simplify(self.borne_haute - self.borne_basse)

    def translater(self, valeur: sp.Expr) -> sp.Expr:
        """`x - l` : la variable ramenee a un domaine commencant a zero."""
        return sp.simplify(valeur - self.borne_basse)

    def retablir(self, translatee: sp.Expr) -> sp.Expr:
        """L'inverse exact de la translation, pour rendre la solution lisible."""
        return sp.simplify(translatee + self.borne_basse)

    def contient(self, valeur: sp.Expr) -> bool:
        """Vrai si la valeur respecte les deux bornes."""
        return bool(
            sp.simplify(valeur - self.borne_basse) >= 0
            and sp.simplify(self.borne_haute - valeur) >= 0,
        )
