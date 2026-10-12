"""Discuter un systeme a parametre par la condition de compatibilite.

`discuter` examine les valeurs qui annulent le determinant. Il ne dit
rien d'un systeme dont le determinant est nul pour **toute** valeur du
parametre, ni d'un systeme qui n'est pas carre. La condition de
compatibilite (planche 19) vaut dans tous les cas :

* `rang(A) != rang(A | b)` : le systeme est impossible ;
* sinon il a une solution, unique si `rang(A)` est le nombre d'inconnues.

Les rangs ne peuvent changer qu'aux valeurs du parametre qui annulent
un mineur de `(A | b)` sans qu'il soit identiquement nul. Hors de ces
valeurs, en nombre fini, chaque mineur garde son caractere nul ou non
nul : la nature du systeme y est partout la meme, et il suffit de la
lire en une seule d'entre elles.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, count

import sympy as sp

from polytech.maths.domain.algebre.solutions import (
    Nature,
    nature_des_solutions,
    nombre_de_parametres,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire


@dataclass(frozen=True, slots=True)
class DiscussionParLeRang:
    """La nature hors des cas particuliers, et celle de chacun d'eux."""

    systeme: SystemeLineaire
    parametre: sp.Symbol
    generique: Nature
    particulieres: dict[sp.Expr, Nature]

    def en(self, valeur: sp.Expr) -> SystemeLineaire:
        """Le systeme, le parametre fixe a cette valeur."""
        return SystemeLineaire(
            self.systeme.matrice.subs(self.parametre, valeur),
            self.systeme.second_membre.subs(self.parametre, valeur),
        )


def _mineurs(matrice: sp.Matrix) -> list[sp.Expr]:
    """Tous les mineurs, de tous les ordres."""
    return [
        matrice.extract(list(lignes), list(colonnes)).det()
        for ordre in range(1, min(matrice.shape) + 1)
        for lignes in combinations(range(matrice.rows), ordre)
        for colonnes in combinations(range(matrice.cols), ordre)
    ]


def _valeurs_critiques(systeme: SystemeLineaire, parametre: sp.Symbol) -> list[sp.Expr]:
    """Les valeurs reelles ou un mineur non identiquement nul s'annule."""
    augmentee = systeme.matrice.row_join(systeme.second_membre)
    racines: set[sp.Expr] = set()
    for mineur in _mineurs(augmentee):
        developpe = sp.expand(mineur)
        if parametre in developpe.free_symbols:
            racines |= set(sp.solve(sp.Eq(developpe, 0), parametre))
    return sorted((r for r in racines if r.is_real), key=sp.default_sort_key)


def _profil(systeme: SystemeLineaire) -> tuple[Nature, int]:
    return nature_des_solutions(systeme), nombre_de_parametres(systeme)


def discuter_par_le_rang(
    systeme: SystemeLineaire, parametre: sp.Symbol
) -> DiscussionParLeRang:
    """La nature du systeme pour toute valeur reelle du parametre."""
    critiques = _valeurs_critiques(systeme, parametre)
    temoin = next(sp.Integer(n) for n in count() if sp.Integer(n) not in critiques)
    vide = DiscussionParLeRang(systeme, parametre, Nature.IMPOSSIBLE, {})
    generique = _profil(vide.en(temoin))
    particulieres = {
        valeur: _profil(vide.en(valeur))[0]
        for valeur in critiques
        if _profil(vide.en(valeur)) != generique
    }
    return DiscussionParLeRang(systeme, parametre, generique[0], particulieres)
