"""Precision reellement atteinte par le manometre.

Le corrige donnait les sensibilites une a une -- ``dp1/dh_i = rho_i g`` --
sans jamais les composer. Or le nombre de chiffres qu'il affiche est une
affirmation comme une autre : elle doit se justifier.

Convention suivie (GUM, evaluation de type B) : une lecture au millimetre
pres n'est pas connue « a 1 mm » mais uniformement distribuee sur une
demi-etendue de 0,5 mm, d'ou un ecart-type de 0,5 mm sur racine de trois.
Les sources etant independantes, leurs variances s'ajoutent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.fluides.domain.statique.dilatation import (
    EAU_DILATATION,
    HUILE_DILATATION,
    MERCURE_DILATATION,
    variation_relative,
)

if TYPE_CHECKING:
    from polytech.fluides.domain.grandeurs import Pression
    from polytech.fluides.domain.statique.dilatation import DilatationThermique
    from polytech.fluides.domain.statique.manometre import Manometre

_RACINE_DE_TROIS = math.sqrt(3.0)
_PASCALS_PAR_KILOPASCAL = 1_000.0
_DECIMALES_MAX = 6


def incertitude_de_lecture(resolution: float) -> float:
    """Ecart-type associe a une resolution de lecture (loi rectangulaire)."""
    if resolution <= 0:
        msg = f"resolution non strictement positive : {resolution}"
        raise ValueError(msg)
    return (resolution / 2) / _RACINE_DE_TROIS


@dataclass(frozen=True, slots=True)
class Contribution:
    """Part d'une source d'incertitude, exprimee en pascals."""

    nom: str
    ecart_type: float

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.ecart_type < 0:
            msg = f"ecart-type negatif : {self.ecart_type}"
            raise ValueError(msg)

    def part_de_variance(self, budget: BudgetDIncertitude) -> float:
        """Poids de cette source dans la variance totale."""
        return self.ecart_type**2 / budget.ecart_type_compose() ** 2


@dataclass(frozen=True, slots=True)
class BudgetDIncertitude:
    """Ensemble des sources recensees pour une meme grandeur."""

    contributions: tuple[Contribution, ...]

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if not self.contributions:
            msg = "aucune source d'incertitude recensee"
            raise ValueError(msg)

    def ecart_type_compose(self) -> float:
        """Racine de la somme des carres : les sources sont independantes."""
        return math.hypot(*[c.ecart_type for c in self.contributions])

    def dominante(self) -> Contribution:
        """Source qui pese le plus lourd dans le budget."""
        return max(self.contributions, key=lambda c: c.ecart_type)

    def decimales_justifiees_en_kilopascals(self) -> int:
        """Nombre de decimales qu'un resultat en kPa peut legitimement porter.

        On retient le rang du *premier chiffre significatif* de
        l'incertitude : afficher au-dela serait pretendre une precision
        qu'on n'a pas.

        Ce rang se lit sur le logarithme decimal, et non sur un arrondi
        successif : arrondir 0,069 au dixieme donne 0,1, non nul, ce qui
        ferait conclure a tort qu'une seule decimale suffit.
        """
        incertitude = self.ecart_type_compose() / _PASCALS_PAR_KILOPASCAL
        if incertitude <= 0:
            return 0
        rang = -math.floor(math.log10(incertitude))
        return max(0, min(rang, _DECIMALES_MAX))


_DILATATIONS: dict[str, DilatationThermique] = {
    "eau": EAU_DILATATION,
    "huile": HUILE_DILATATION,
    "Hg": MERCURE_DILATATION,
}


def _contributions_thermiques(
    manometre: Manometre,
    plage: float,
) -> list[Contribution]:
    """Une contribution par fluide : sa masse volumique depend de T.

    L'enonce ne donne aucune temperature ; on la suppose donc uniformement
    repartie sur ``plage`` kelvins, d'ou un ecart-type de la demi-etendue
    sur racine de trois -- la meme convention que pour une lecture.
    """
    if plage <= 0:
        msg = f"plage thermique non strictement positive : {plage}"
        raise ValueError(msg)
    incertitude_temperature = incertitude_de_lecture(plage)
    return [
        Contribution(
            nom=f"{troncon.nom} (température)",
            ecart_type=troncon.contribution(manometre.pesanteur).pascals
            * variation_relative(_DILATATIONS[troncon.nom], incertitude_temperature),
        )
        for troncon in manometre.troncons
    ]


def budget_du_manometre(
    manometre: Manometre,
    resolution_hauteur: float,
    resolution_atmospherique: Pression,
    plage_thermique: float | None = None,
) -> BudgetDIncertitude:
    """Compose les incertitudes de lecture du manometre et de la reference.

    La sensibilite a chaque hauteur vaut ``rho g`` -- le coefficient que
    porte ce troncon dans l'equation du manometre -- et son signe n'a aucune
    importance ici, seuls les carres comptent.
    """
    lecture = incertitude_de_lecture(resolution_hauteur)
    contributions = [
        Contribution(
            nom=troncon.nom,
            ecart_type=troncon.fluide.kg_par_m3 * manometre.pesanteur * lecture,
        )
        for troncon in manometre.troncons
    ]
    contributions.append(
        Contribution(
            nom="atmosphere",
            ecart_type=incertitude_de_lecture(resolution_atmospherique.pascals),
        ),
    )
    if plage_thermique is not None:
        contributions.extend(_contributions_thermiques(manometre, plage_thermique))
    return BudgetDIncertitude(contributions=tuple(contributions))
