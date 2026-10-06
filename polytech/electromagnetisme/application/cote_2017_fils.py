"""Exercice cote 2017, question 4 : n fils charges au-dessus d'un plan conducteur.

Methode des images : le plan est remplace par `n` fils de charge opposee,
symetriques des premiers. Le champ en `P` est la somme des `2n` champs
radiaux, chacun donne par le theoreme de Gauss.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import (
    PERMITTIVITE,
    champ_par_gauss,
)
from polytech.electromagnetisme.domain.electrostatique.passage import saut_normal

EPS: Final = PERMITTIVITE
D, H = sp.symbols("d h", positive=True)
LINEIQUE: Final = sp.Symbol("lambda", positive=True)
"""`Q/L`, la charge lineique de chaque fil."""
RANG_DE_P: Final = 3
"""`P` est au pied du troisieme fil."""
_X = sp.Symbol("x", real=True)


def _champ_d_un_fil(lineique: sp.Expr, ecart: sp.Expr, hauteur: sp.Expr) -> sp.Matrix:
    """Le champ en `P` d'un fil situe a `(ecart, hauteur)` de lui."""
    distance = sp.sqrt(ecart**2 + hauteur**2)
    module = champ_par_gauss(lineique, sp.Integer(1), distance)
    return module * sp.Matrix([-ecart, -hauteur]) / distance


def _champ(rangs: range) -> tuple[sp.Expr, sp.Expr]:
    total = sp.zeros(2, 1)
    for rang in rangs:
        ecart = (rang - RANG_DE_P) * D
        total += _champ_d_un_fil(LINEIQUE, ecart, H)
        total += _champ_d_un_fil(-LINEIQUE, ecart, -H)
    return total[0], total[1]


def champ_en_p(nombre: int) -> tuple[sp.Expr, sp.Expr]:
    """`(E_x, E_y)` en `P`, pour `n` fils numerotes de 1 a `n`."""
    if nombre < 1:
        msg = f"nombre de fils nul ou negatif : {nombre}"
        raise ValueError(msg)
    horizontal, vertical = _champ(range(1, nombre + 1))
    return sp.simplify(horizontal), sp.simplify(vertical)


def charge_surfacique(nombre: int) -> sp.Expr:
    """`sigma = u_n . D`, le champ etant nul dans le conducteur."""
    _, vertical = champ_en_p(nombre)
    au_dessus = (0, EPS * vertical, 0)
    return saut_normal(au_dessus, (0, 0, 0), normale_1=(0, 1, 0))


def charge_surfacique_centree(de_chaque_cote: int) -> sp.Expr:
    """La meme, pour `2m + 1` fils autour de `P` ; non simplifiee, a chiffrer."""
    rangs = range(RANG_DE_P - de_chaque_cote, RANG_DE_P + de_chaque_cote + 1)
    return EPS * _champ(rangs)[1]


def charge_induite_par_un_fil() -> sp.Expr:
    """L'integrale, sur tout le plan, de la charge qu'un fil seul y induit."""
    densite = -H * LINEIQUE / (sp.pi * (_X**2 + H**2))
    return sp.integrate(densite, (_X, -sp.oo, sp.oo))


def charge_surfacique_du_reseau_infini() -> sp.Expr:
    """Une infinite de fils : `sigma = -(lambda / d) coth(pi h / d)`.

    La somme `1/(k^2 d^2 + h^2)` sur tous les entiers vaut
    `pi coth(pi h/d) / (d h)` : c'est le developpement en fractions
    simples de la cotangente hyperbolique, que `sympy` ne sait pas sommer.
    La forme close est donc posee ici, et un test la confronte a la somme
    tronquee, dans la borne de son reste.
    """
    return -LINEIQUE / D * sp.coth(sp.pi * H / D)


def reste_de_la_somme(de_chaque_cote: int) -> sp.Expr:
    """Un majorant de ce que la somme tronquee a `m` fils de chaque cote oublie.

    Chaque terme oublie est inferieur a `1/(k^2 d^2)`, et la somme des
    `1/k^2` au-dela de `m` est inferieure a `1/m`.
    """
    return H * LINEIQUE / sp.pi * 2 / (D**2 * de_chaque_cote)


def charge_surfacique_sous_un_fil_seul() -> sp.Expr:
    """Le cas d'un fil unique, `P` a son pied : le terme dominant de la somme."""
    return sp.simplify(EPS * _champ(range(RANG_DE_P, RANG_DE_P + 1))[1])
