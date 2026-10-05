"""Les echangeurs a courants paralleles : bilan, ecarts d'extremite, surface.

Syllabus de thermique, tome 2, deuxieme partie, sections 2.1 a 2.5. Le
fluide chaud, note 1, rechauffe le fluide froid, note 2 ; `E` et `S`
designent l'entree et la sortie. Sans pertes vers l'ambiance,

    Phi = q_m,1 c_1 (t_1E - t_1S) = q_m,2 c_2 (t_2S - t_2E),       (2.6)

et, a coefficient global uniforme, `Phi = K_gl S Dt_log` (2.9), ou la
moyenne logarithmique se prend entre les ecarts des deux extremites.

Ces ecarts dependent du sens. A **cocourant**, les deux fluides entrent
du meme cote : `t_1E - t_2E` et `t_1S - t_2S`. A **contrecourant**, ils
entrent par des extremites opposees : `t_1E - t_2S` et `t_1S - t_2E`.
Pour quatre temperatures donnees, la moyenne du contrecourant est
toujours la plus grande (section 2.5) : il lui faut moins de surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from polytech.thermique.domain.echangeurs import moyenne_logarithmique


class Sens(Enum):
    """Les deux dispositions que le cours integre."""

    COCOURANT = "courants paralleles et de meme sens, antimethodique"
    CONTRECOURANT = "courants paralleles et de sens contraires, methodique"


def flux_du_bilan(debit: float, chaleur_massique: float, variation: float) -> float:
    """(2.6) : `Phi = q_m c |t_E - t_S|`, pour un fluide sans changement de phase."""
    if debit * chaleur_massique <= 0:
        msg = f"debit de capacite nul ou negatif : {debit * chaleur_massique} W/K"
        raise ValueError(msg)
    return debit * chaleur_massique * variation


def debit_condense(flux: float, chaleur_latente: float) -> float:
    """(2.12) : `q_m = Phi / l_v`, le fluide qui change de phase."""
    if chaleur_latente <= 0:
        msg = f"chaleur latente nulle ou negative : {chaleur_latente} J/kg"
        raise ValueError(msg)
    return flux / chaleur_latente


def surface_d_echange(flux: float, coefficient: float, ecart_moyen: float) -> float:
    """Remarque 5 : `S = Phi / (K_gl Dt_log)`, le dimensionnement."""
    if coefficient <= 0:
        msg = f"coefficient global nul ou negatif : {coefficient} W/(m^2 K)"
        raise ValueError(msg)
    return flux / (coefficient * ecart_moyen)


@dataclass(frozen=True, slots=True)
class Extremites:
    """`t_1E`, `t_1S`, `t_2E` et `t_2S` : les seules temperatures accessibles."""

    entree_chaude: float
    sortie_chaude: float
    entree_froide: float
    sortie_froide: float

    def __post_init__(self) -> None:
        """Le fluide 1 cede de la chaleur, le fluide 2 en recoit."""
        if self.sortie_chaude > self.entree_chaude:
            msg = f"le fluide chaud ne se refroidit pas : {self.sortie_chaude}"
            raise ValueError(msg)
        if self.sortie_froide < self.entree_froide:
            msg = f"le fluide froid ne se rechauffe pas : {self.sortie_froide}"
            raise ValueError(msg)

    def ecarts(self, sens: Sens) -> tuple[float, float]:
        """`(Dt_A, Dt_B)`, a l'entree puis a la sortie du fluide chaud."""
        if sens is Sens.COCOURANT:
            return (
                self.entree_chaude - self.entree_froide,
                self.sortie_chaude - self.sortie_froide,
            )
        return (
            self.entree_chaude - self.sortie_froide,
            self.sortie_chaude - self.entree_froide,
        )

    def moyenne_logarithmique(self, sens: Sens) -> float:
        """(2.10) ; a fluide isotherme, (2.11), et le sens n'importe plus."""
        return moyenne_logarithmique(*self.ecarts(sens))

    def rendements(self) -> tuple[float, float]:
        """(2.13) : `(eta_1, eta_2)`, rapportes a l'ecart des deux entrees."""
        maximal = self.entree_chaude - self.entree_froide
        chaud = (self.entree_chaude - self.sortie_chaude) / maximal
        return chaud, (self.sortie_froide - self.entree_froide) / maximal
