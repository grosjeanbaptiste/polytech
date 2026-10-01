"""Les ailettes, droites et circulaires, telles que le formulaire les donne.

Deux formules du formulaire sont reprises mot pour mot, parce qu'elles y
sont ecrites en clair -- ce qui est rare, le reste du document etant fait
d'abaques :

* la temperature en bout d'ailette droite,
  `t_l - t_f = (t_p - t_f) / cosh(m l)` ;
* la correction circulaire, valable pour `m (r_e - r_b) < 1,6`,
  `Omega_circ = Omega_L - 1,672 (1 - Omega_L) (Omega_L (r_e - r_b)/r_e)^1,724`.

L'efficacite de l'ailette droite a bout adiabatique,
`Omega_L = tanh(m l) / (m l)`, est le resultat classique dont ces deux-la
dependent ; le formulaire la donne sous forme d'abaque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.thermique.domain.validation import exiger_strictement_positif

LIMITE_DE_LA_CORRECTION = 1.6
"""Au-dela, le formulaire ne garantit plus la correction circulaire."""

_COEFFICIENT = 1.672
_EXPOSANT = 1.724


@dataclass(frozen=True, slots=True)
class AiletteDroite:
    """Une ailette droite caracterisee par son parametre `m` et sa longueur."""

    parametre: float
    longueur: float

    def __post_init__(self) -> None:
        """Valide les invariants a la construction."""
        exiger_strictement_positif(self.parametre, "parametre d'ailette")
        exiger_strictement_positif(self.longueur, "longueur d'ailette")

    @property
    def argument(self) -> float:
        """`m l`, dont tout le comportement de l'ailette depend."""
        return self.parametre * self.longueur

    def efficacite(self) -> float:
        """`tanh(m l) / (m l)` : bout adiabatique."""
        return math.tanh(self.argument) / self.argument

    def ecart_en_bout(self, ecart_au_pied: float) -> float:
        """`(t_p - t_f) / cosh(m l)`, formule du formulaire."""
        return ecart_au_pied / math.cosh(self.argument)


@dataclass(frozen=True, slots=True)
class AiletteCirculaire:
    """Une ailette circulaire, corrigee depuis l'ailette droite equivalente."""

    droite: AiletteDroite
    rayon_de_base: float
    rayon_externe: float

    def __post_init__(self) -> None:
        """Refuse une ailette dont le rayon externe n'excede pas la base."""
        if self.rayon_externe <= self.rayon_de_base:
            msg = (
                f"rayon externe non superieur a la base : "
                f"{self.rayon_externe} <= {self.rayon_de_base}"
            )
            raise ValueError(msg)

    @property
    def hauteur(self) -> float:
        """`r_e - r_b`, la hauteur radiale de l'ailette."""
        return self.rayon_externe - self.rayon_de_base

    def efficacite(self) -> float:
        """La correction du formulaire ; leve hors de son domaine de validite."""
        argument = self.droite.parametre * self.hauteur
        if argument >= LIMITE_DE_LA_CORRECTION:
            msg = (
                f"correction circulaire hors de son domaine : m (r_e - r_b) = "
                f"{argument:.3f} >= {LIMITE_DE_LA_CORRECTION}"
            )
            raise ValueError(msg)
        droite = self.droite.efficacite()
        facteur = float(
            (droite * self.hauteur / self.rayon_externe) ** _EXPOSANT,
        )
        return droite - _COEFFICIENT * (1 - droite) * facteur
