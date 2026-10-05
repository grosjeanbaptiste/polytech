"""Les reactions chimiques : la formule brute et le bilan des atomes.

Une formule brute enumere les atomes d'une molecule : `CH4`, un carbone
et quatre hydrogenes. Une reaction est **equilibree** quand chaque
element se retrouve en meme nombre de part et d'autre, compte tenu des
coefficients : rien ne se cree, rien ne se perd.

Le cours de thermique s'y adosse pour la suie des flammes -- le cracking
`CH4 -> C + 2 H2` -- et pour les deux gaz qui rayonnent dans les fumees,
`CO2` et `H2O`, produits de la combustion des hydrocarbures.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")
FORMULE = re.compile(r"(?:[A-Z][a-z]?\d*)+")


def composition(formule: str) -> dict[str, int]:
    """Le nombre d'atomes de chaque element d'une formule brute."""
    if not FORMULE.fullmatch(formule):
        msg = f"formule brute illisible : {formule!r}"
        raise ValueError(msg)
    atomes: Counter[str] = Counter()
    for element, indice in ELEMENT.findall(formule):
        atomes[element] += int(indice) if indice else 1
    return dict(atomes)


def _bilan(especes: Mapping[str, int]) -> Counter[str]:
    atomes: Counter[str] = Counter()
    for formule, coefficient in especes.items():
        if coefficient <= 0:
            msg = f"coefficient nul ou negatif : {coefficient} {formule}"
            raise ValueError(msg)
        for element, nombre in composition(formule).items():
            atomes[element] += coefficient * nombre
    return atomes


def est_equilibree(reactifs: Mapping[str, int], produits: Mapping[str, int]) -> bool:
    """Vrai si chaque element est conserve entre reactifs et produits."""
    return _bilan(reactifs) == _bilan(produits)
