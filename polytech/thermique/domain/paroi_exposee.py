"""La paroi exterieure d'un toit, entre l'interieur froid et le soleil.

Bilan sur la paroi, en flux par metre carre :

    K_gl,i (T_i - T) + alpha phi_sol + K_cf (T_e - T) + eps sigma (T_e^4 - T^4) = 0

ou `K_gl,i` reunit la convection interieure et la conduction de l'isolant.
Le rayonnement vers l'environnement -- grand devant le toit, donc noir et
vu sous un facteur un -- le rend non lineaire. On le resout tel quel : le
bilan decroit avec `T`, il est positif a la plus froide des deux
temperatures et negatif au-dela de la plus chaude augmentee de
l'echauffement solaire, donc une racine unique est encadree.

Le coefficient `K_ray` qu'un calcul a la main linearise et itere se lit
ensuite, comme une consequence et non comme une etape.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import brentq

from polytech.thermique.domain.rayonnement.corps_noir import STEFAN_BOLTZMANN


@dataclass(frozen=True, slots=True)
class ToitExpose:
    """Les donnees du bilan, temperatures en kelvins."""

    conductance_interieure: float
    temperature_interieure: float
    convection: float
    temperature_exterieure: float
    absorptivite: float
    flux_solaire: float
    emissivite: float

    def _bilan(self, temperature: float) -> float:
        rayonnement = (
            self.emissivite
            * STEFAN_BOLTZMANN
            * (self.temperature_exterieure**4 - temperature**4)
        )
        return (
            self.conductance_interieure * (self.temperature_interieure - temperature)
            + self.absorptivite * self.flux_solaire
            + self.convection * (self.temperature_exterieure - temperature)
            + rayonnement
        )

    def temperature(self) -> float:
        """La temperature de la paroi qui annule le bilan."""
        froide = min(self.temperature_interieure, self.temperature_exterieure)
        chaude = max(self.temperature_interieure, self.temperature_exterieure)
        conductances = self.conductance_interieure + self.convection
        borne = chaude + self.absorptivite * self.flux_solaire / conductances + 1
        return float(brentq(self._bilan, froide, borne))

    def flux_vers_l_interieur(self, aire: float) -> float:
        """La charge frigorifique : ce que la paroi cede a l'interieur, en watts."""
        ecart = self.temperature() - self.temperature_interieure
        return self.conductance_interieure * ecart * aire

    def coefficient_de_rayonnement(self) -> float:
        """`K_ray = eps sigma (T^4 - T_e^4) / (T - T_e)`, la linearisation."""
        t, te = self.temperature(), self.temperature_exterieure
        if t == te:
            return 4 * self.emissivite * STEFAN_BOLTZMANN * te**3
        return self.emissivite * STEFAN_BOLTZMANN * (t**4 - te**4) / (t - te)
