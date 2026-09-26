"""Dependance en temperature des masses volumiques (cours II.1).

L'enonce donne trois masses volumiques sans dire a quelle temperature. Ce
n'est pas anodin : rho_Hg = 13600 kg/m3 est une valeur de reference proche
de 0 C, alors qu'un manometre de laboratoire travaille plutot vers 20 C, ou
le mercure vaut environ 13546. Un ecart de 0,4 %, la ou l'incertitude de
lecture du meme appareil ne pese que 0,04 %.

Le modele retenu est le plus simple qui soit, et suffisant sur quelques
dizaines de kelvins : une dilatation volumique lineaire, donc une masse
volumique en 1 / (1 + beta * Delta T).
"""

from __future__ import annotations

from dataclasses import dataclass

from polytech.fluides.domain.grandeurs import MasseVolumique


@dataclass(frozen=True, slots=True)
class DilatationThermique:
    """Coefficient de dilatation volumique d'un fluide, en K^-1."""

    coefficient: float
    nom: str = "fluide"

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.coefficient <= 0:
            msg = f"coefficient de dilatation non positif : {self.coefficient}"
            raise ValueError(msg)

    def masse_volumique_a(
        self,
        reference: MasseVolumique,
        ecart: float,
    ) -> MasseVolumique:
        """Masse volumique apres un echauffement de ``ecart`` kelvins."""
        return MasseVolumique(
            kg_par_m3=reference.kg_par_m3 / (1 + self.coefficient * ecart),
        )


def variation_relative(dilatation: DilatationThermique, ecart: float) -> float:
    """Variation relative de masse volumique, au premier ordre en ``ecart``.

    C'est la forme utile pour un budget d'incertitude : une sensibilite,
    non une valeur.
    """
    return dilatation.coefficient * abs(ecart)


MERCURE_DILATATION = DilatationThermique(coefficient=1.81e-4, nom="Hg")
"""Mercure : valeur usuelle, quasi constante entre 0 et 100 C."""

EAU_DILATATION = DilatationThermique(coefficient=2.07e-4, nom="eau")
"""Eau liquide vers 20 C. En dessous de 4 C le signe s'inverse."""

HUILE_DILATATION = DilatationThermique(coefficient=7.0e-4, nom="huile")
"""Huile minerale : ordre de grandeur, les formulations variant beaucoup."""
