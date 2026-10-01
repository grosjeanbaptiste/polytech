"""Le reservoir cryogenique : deux resistances en serie, et un debit.

Le flux est calcule **deux fois**, par deux chemins qui ne partagent
aucune formule.

`flux` suit le schema en resistances que le cours enseigne : la coque et
le film exterieur sont traverses par le meme flux, leurs resistances
s'additionnent, et l'ecart de temperature total se divise par la somme.

`flux_par_resolution_directe` n'emprunte rien a ce raccourci : il resout
l'equation de la chaleur en coordonnees spheriques avec ses deux
conditions aux limites -- temperature imposee a l'interieur, condition de
Robin a l'exterieur -- et tire le flux du gradient obtenu. C'est le seul
recoupement disponible, puisque la planche de valeurs numeriques du
corrige officiel est une image dont rien ne s'extrait.

La resistance du cote hydrogene est negligee, comme le corrige le dit :
l'ebullition donne un coefficient d'echange tres grand, donc une
resistance tres petite devant les deux autres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.thermique.domain.conduction.symboles import CONSTANTES, RAYON
from polytech.thermique.domain.echanges import ResistanceThermique
from polytech.thermique.domain.grandeurs import Temperature

if TYPE_CHECKING:
    from polytech.thermique.domain.conduction.spherique import CoqueSpherique
    from polytech.thermique.domain.echanges import CoefficientDEchange
    from polytech.thermique.domain.materiaux import ChaleurLatente


@dataclass(frozen=True, slots=True)
class ReservoirCryogenique:
    """Un liquide froid, une coque isolante, un echange avec l'ambiance."""

    coque: CoqueSpherique
    echange_exterieur: CoefficientDEchange
    temperature_interieure: Temperature
    temperature_exterieure: Temperature
    chaleur_latente: ChaleurLatente

    def __post_init__(self) -> None:
        """Refuse un interieur plus chaud : le debit « evapore » serait negatif."""
        if self.temperature_interieure.kelvins >= self.temperature_exterieure.kelvins:
            dedans = self.temperature_interieure.kelvins
            dehors = self.temperature_exterieure.kelvins
            msg = f"interieur plus chaud que l'ambiance : {dedans} K >= {dehors} K"
            raise ValueError(msg)

    def resistance_exterieure(self) -> ResistanceThermique:
        """1 / (K S_e) : le film de convection et de rayonnement reunis."""
        surface = self.coque.exterieur.surface_de_sphere
        conductance = self.echange_exterieur.watts_par_m2_k * surface
        return ResistanceThermique(kelvins_par_watt=1.0 / conductance)

    def resistance_totale(self) -> ResistanceThermique:
        """Les deux resistances en serie ; celle de l'hydrogene est negligee."""
        return self.coque.resistance().en_serie_avec(self.resistance_exterieure())

    def ecart_moteur(self) -> float:
        """L'ecart de temperature qui fait circuler le flux, en kelvins."""
        return self.temperature_exterieure.ecart_a(self.temperature_interieure).kelvins

    def flux(self) -> float:
        """Le flux entrant dans le reservoir, en watts, par le schema serie."""
        return self.ecart_moteur() / self.resistance_totale().kelvins_par_watt

    def temperature_de_paroi_exterieure(self) -> Temperature:
        """L'ambiance moins la chute dans le film exterieur."""
        chute = self.flux() * self.resistance_exterieure().kelvins_par_watt
        return Temperature(kelvins=self.temperature_exterieure.kelvins - chute)

    def debit_evapore(self) -> float:
        """Le debit massique d'hydrogene vaporise, en kg/s."""
        return self.flux() / self.chaleur_latente.joules_par_kg

    def _conditions_aux_limites(self) -> list[sp.Eq]:
        """Temperature imposee au rayon interieur, Robin au rayon exterieur."""
        profil = self.coque.profil_general()
        lam = self.coque.conductivite.watts_par_m_k
        interieur = sp.Eq(
            profil.subs(RAYON, self.coque.interieur.metres),
            self.temperature_interieure.kelvins,
        )
        surface = self.coque.exterieur.surface_de_sphere
        conduit = (
            lam
            * sp.diff(profil, RAYON).subs(
                RAYON,
                self.coque.exterieur.metres,
            )
            * surface
        )
        apporte = (
            self.echange_exterieur.watts_par_m2_k
            * surface
            * (
                self.temperature_exterieure.kelvins
                - profil.subs(RAYON, self.coque.exterieur.metres)
            )
        )
        return [interieur, sp.Eq(conduit, apporte)]

    def flux_par_resolution_directe(self) -> float:
        """Le meme flux, tire de l'equation de la chaleur et non du schema."""
        resolu = sp.solve(self._conditions_aux_limites(), CONSTANTES, dict=True)
        if len(resolu) != 1:
            msg = f"les conditions aux limites n'ont pas de solution unique : {resolu}"
            raise ValueError(msg)
        premiere = resolu[0][CONSTANTES[0]]
        lam = self.coque.conductivite.watts_par_m_k
        return float(4.0 * math.pi * lam * premiere)
