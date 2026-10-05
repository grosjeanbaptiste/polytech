"""Le champ electromoteur, la force electromotrice et la loi des chaines.

Le syllabus d'electricite generale (chapitre 6) : dans un conducteur
heterogene -- deux metaux en contact, ou un metal dont la temperature
n'est pas uniforme -- les charges libres subissent une force `F = q E_e`
(6.1.1) qui n'est pas electrostatique. `E_e` est le champ electromoteur,
et sa circulation la force electromotrice,

    E_12 = integrale de 1 a 2 de E_e . dl                        (6.2.1)

A l'equilibre le champ electrostatique le compense, `E_s = -E_e` (6.3.1) :
`E_12 = U_21 = V_2 - V_1` (6.3.3). Au contact de deux metaux (effet Volta),
`E_12 = V_2,int - V_1,int = P_S / eps_0` (6.4.1, 6.4.2).

Une chaine metallique isotherme fermee n'a pas de force electromotrice --
sinon un courant circulerait sans source d'energie -- d'ou la loi des
chaines : `E_1n = E_12 + ... + E_(n-1)n` (6.7.2) ne depend que des metaux
extremes. La tension voltaique `V_n,ext - V_1,ext = E_01 + E_1n + E_n0`
(6.7.1) y ajoute les doubles couches des deux surfaces libres.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.potentiel import circulation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur


def fem_le_long_d_un_trajet(
    champ: Vecteur, courbe: Vecteur, parametre: sp.Symbol, debut: object, fin: object
) -> sp.Expr:
    """`E_12 = integrale de E_e . dl` le long de la courbe parametree."""
    return circulation(champ, courbe, parametre, debut, fin)


def moment_de_la_double_couche(fem: object) -> sp.Expr:
    """`P_S = eps_0 E_12` : le moment surfacique de la double couche de contact."""
    return PERMITTIVITE * sp.sympify(fem)


@dataclass(frozen=True)
class ChaineIsotherme:
    """Des conducteurs a la meme temperature, dans un meme isolant."""

    potentiels_internes: Mapping[str, float]
    fems_de_surface: Mapping[str, float]

    def _potentiel(self, conducteur: str) -> float:
        if conducteur not in self.potentiels_internes:
            msg = f"conducteur inconnu de la chaine : {conducteur}"
            raise ValueError(msg)
        return self.potentiels_internes[conducteur]

    def fem_de_contact(self, premier: str, second: str) -> float:
        """`E_12 = V_2,int - V_1,int`."""
        return self._potentiel(second) - self._potentiel(premier)

    def fem(self, chaine: Sequence[str]) -> float:
        """`E_1n`, la somme des forces electromotrices de contact successives."""
        if len(chaine) < 2:  # noqa: PLR2004 -- un contact relie deux conducteurs
            msg = f"au moins deux conducteurs attendus, et non {len(chaine)}"
            raise ValueError(msg)
        return sum(self.fem_de_contact(a, b) for a, b in pairwise(chaine))

    def tension_voltaique(self, chaine: Sequence[str]) -> float:
        """`V_n,ext - V_1,ext = E_01 + E_1n + E_n0`."""
        entre_les_extremes = self.fem(chaine)
        entree = self.fems_de_surface[chaine[0]]
        return entree + entre_les_extremes - self.fems_de_surface[chaine[-1]]
