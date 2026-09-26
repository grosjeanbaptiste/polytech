"""Domaine de validite des resolutions de la partie II.

Le corrige divise par ``u`` pour former la pente ``dy/dx = v/u``, puis prend
un logarithme. Chacune de ces operations a un domaine, et une formule citee
hors de son domaine est une formule fausse. Ce module calcule les lieux ou
l'equation degenere, et verifie que la solution close satisfait bien
l'equation ailleurs -- par report symbolique, ce qui est independant du
recoupement numerique de :mod:`.corroboration`.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T, X, Y

_VARIABLES = (X, Y, T)


def lieu_singulier(composante: sp.Expr) -> sp.Set:
    """Conditions sous lesquelles ``composante`` s'annule.

    Rend un ensemble d'egalites plutot qu'une reponse numerique : le
    corrige doit pouvoir citer « x = 0 » ou « t = 0 », pas une liste de
    points.
    """
    presentes = [v for v in _VARIABLES if composante.has(v)]
    annulations = {
        sp.Eq(variable, 0)
        for variable in presentes
        if sp.simplify(composante.subs(variable, 0)) == 0
    }
    return sp.FiniteSet(*annulations) if annulations else sp.EmptySet


def residu_des_lignes_de_courant(champ: sp.Expr) -> sp.Expr:
    """Reporte la solution close dans son equation : doit rendre zero.

    C'est la verification la plus directe qui soit : on ne refait pas le
    calcul, on controle son resultat. Elle ne vaut bien sur que hors des
    lieux singuliers, ou l'equation elle-meme n'a pas de sens.
    """
    ordonnee = champ.lignes_de_courant_explicites()
    pente_attendue = (champ.v / champ.u).subs(Y, ordonnee)
    return sp.simplify(sp.diff(ordonnee, X) - pente_attendue)
