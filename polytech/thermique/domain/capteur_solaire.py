"""Le capteur solaire plan, modelise par un reseau de resistances.

Syllabus de thermique, tome 1, section 0.8.1. Le flux solaire absorbe,
`Phi_A = Phi_sol tau_V alpha_A`, arrive sur l'absorbeur. De la, deux
chemins : vers l'eau, par la resistance `R_c`, c'est le flux utile ;
vers l'ambiance, par la resistance de pertes `R_p`, qui met en parallele
l'arriere et l'avant du capteur.

Par superposition, et en posant `F_c = R_p / (R_c + R_p)`,

`Phi_ut = F_c (Phi_A - (t_eau - t_amb) / R_p)`.

Le rendement `eta_c = Phi_ut / Phi_sol` est alors une droite de la
variable `(t_eau - t_amb) / phi_sol` : ordonnee a l'origine
`F_c tau_V alpha_A`, pente `-F_c K_p`, avec `K_p = 1 / (R_p S_c)`. C'est
la caracteristique de fonctionnement du capteur.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.thermique.domain.reseaux import en_parallele, en_serie

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""

type Couple = tuple[float, float]
"""Les resistances de convection et de rayonnement d'une meme surface."""


@dataclass(frozen=True, slots=True)
class CapteurPlan:
    """`R_c`, `R_p`, et le produit `tau_V alpha_A` de la vitre et de l'absorbeur."""

    vers_l_eau: Grandeur
    de_pertes: Grandeur
    transmission_absorption: Grandeur

    def facteur(self) -> Grandeur:
        """`F_c = R_p / (R_c + R_p)`."""
        return self.de_pertes / (self.vers_l_eau + self.de_pertes)

    def flux_utile(self, absorbe: Grandeur, ecart: Grandeur) -> Grandeur:
        """`Phi_ut`, pour un flux absorbe `Phi_A` et un ecart `t_eau - t_amb`."""
        return self.facteur() * (absorbe - ecart / self.de_pertes)

    def rendement(self, ecart_reduit: float, surface: float) -> float:
        """`eta_c`, l'ecart reduit etant `(t_eau - t_amb) / phi_sol`."""
        pertes = 1 / (self.de_pertes * surface)
        utile = self.transmission_absorption - pertes * ecart_reduit
        return float(self.facteur() * utile)


def resistance_de_pertes(
    *, isolant: float, arriere: Couple, lame: Couple, vitre: float, exterieur: Couple
) -> float:
    """`R_p` : l'arriere et l'avant du capteur, en parallele.

    A l'arriere, l'isolant puis sa surface. A l'avant, la lame d'air, la
    vitre, puis l'exterieur. Chaque surface met en parallele sa
    convection et son rayonnement.
    """
    par_l_arriere = en_serie(isolant, en_parallele(*arriere))
    par_l_avant = en_serie(en_parallele(*lame), vitre, en_parallele(*exterieur))
    return en_parallele(par_l_arriere, par_l_avant)
