"""Les quatre jeux de parametres d'un quadripole, et le passage de l'un a l'autre.

Planches du chapitre 1, partie IV (planches 4 a 30). Un quadripole
lineaire lie quatre grandeurs, `V_1`, `V_2`, `I_1`, `I_2`, par deux
equations. Selon les deux que l'on choisit d'exprimer par les deux
autres, on obtient :

* `Z` : `(V_1, V_2)` par `(I_1, I_2)` -- s'ajoute en **serie** ;
* `Y` : `(I_1, I_2)` par `(V_1, V_2)` -- s'ajoute en **parallele** ;
* `H` : `(V_1, I_2)` par `(I_1, V_2)` -- serie a l'entree, parallele en sortie ;
* `G` : `(I_1, V_2)` par `(V_1, I_2)` -- parallele a l'entree, serie en sortie.

Ce sont les memes deux equations, lues autrement : la conversion est de
l'algebre lineaire, et le tableau de la planche 30 en est le resultat.
La matrice `ABCD`, qui range l'entree a gauche et la sortie a droite,
se multiplie en cascade.
"""

from __future__ import annotations

from enum import Enum

import sympy as sp

V_1, V_2, I_1, I_2 = range(4)


class Jeu(Enum):
    """Un jeu de parametres : les deux grandeurs exprimees, puis les deux libres."""

    Z = ((V_1, V_2), (I_1, I_2))
    Y = ((I_1, I_2), (V_1, V_2))
    H = ((V_1, I_2), (I_1, V_2))
    G = ((I_1, V_2), (V_1, I_2))


def _contraintes(matrice: sp.Matrix, jeu: Jeu) -> sp.Matrix:
    """Les deux equations, en `K (V_1, V_2, I_1, I_2)^T = 0`."""
    exprimees, libres = jeu.value
    lignes = sp.zeros(2, 4)
    for rang, exprimee in enumerate(exprimees):
        lignes[rang, exprimee] = 1
        for colonne, libre in enumerate(libres):
            lignes[rang, libre] -= matrice[rang, colonne]
    return lignes


def convertir(matrice: sp.Matrix, *, depuis: Jeu, vers: Jeu) -> sp.Matrix:
    """La matrice du jeu `vers`, a partir de celle du jeu `depuis`."""
    lignes = _contraintes(matrice, depuis)
    exprimees, libres = vers.value
    a_isoler = lignes.extract([0, 1], list(exprimees))
    if sp.simplify(a_isoler.det()) == 0:
        sens = f"de {depuis.name} vers {vers.name}"
        msg = f"conversion impossible {sens} : matrice singuliere"
        raise ValueError(msg)
    restantes = lignes.extract([0, 1], list(libres))
    return sp.simplify(-a_isoler.inv() * restantes)


def associer(jeu: Jeu, premiere: sp.Matrix, seconde: sp.Matrix) -> sp.Matrix:
    """Deux quadripoles associes selon la topologie du `jeu` : la somme des matrices."""
    del jeu  # la regle est la meme pour les quatre : seule la topologie change
    return premiere + seconde


def abcd_depuis_z(matrice: sp.Matrix) -> sp.Matrix:
    """`(V_1, I_1)` par `(V_2, -I_2)` ; leve si `z_21` est nul."""
    z_11, z_12, z_21, z_22 = matrice
    if sp.simplify(z_21) == 0:
        msg = "z_21 nul : la sortie ne voit pas l'entree, pas de matrice ABCD"
        raise ValueError(msg)
    determinant = z_11 * z_22 - z_12 * z_21
    brute = sp.Matrix([[z_11, determinant], [1, z_22]]) / z_21
    return sp.simplify(brute)
