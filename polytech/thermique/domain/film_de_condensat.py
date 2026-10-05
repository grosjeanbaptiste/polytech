"""Le film de condensat sur une paroi : la theorie de Nusselt.

Syllabus de thermique, tome 2, premiere partie, sections 5.3 et 5.4.
Pour une vapeur pure, la seule resistance est celle du film liquide qui
ruisselle sur la paroi. Ses vitesses sont faibles : la viscosite y
equilibre le poids, `mu u'' + (rho_l - rho_v) g sin(beta) = 0`, et le
profil est une demi-parabole, de vitesse moyenne les deux tiers de sa
vitesse de surface. Le debit par metre de largeur vaut

    G = (rho_l - rho_v) rho_l g delta^3 sin(beta) / (3 mu).

La chaleur latente liberee a l'interface traverse le film par
conduction : `l_v dG = (lambda / delta)(t_sat - t_p) dx`. Le film
s'epaissit donc comme `x^1/4`, le coefficient local `lambda / delta`
decroit de meme, et le coefficient moyen vaut les quatre tiers de sa
valeur au bas de la paroi.

Le sous-refroidissement du condensat se compte en multipliant la chaleur
latente par `1 + 0,375 c_l (t_sat - t_p) / l_v`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from polytech.thermique.domain.convection.naturelle import PESANTEUR

if TYPE_CHECKING:
    import sympy as sp

_TIERS = 1.0 / 3.0


def profil_de_vitesse_du_film(
    ordonnee: sp.Expr, epaisseur: sp.Expr, poids: sp.Expr, viscosite: sp.Expr
) -> sp.Expr:
    """`u = poids (2 delta y - y^2) / (2 mu)`, `poids = (rho_l - rho_v) g sin(beta)`."""
    return poids * (2 * epaisseur * ordonnee - ordonnee**2) / (2 * viscosite)


@dataclass(frozen=True, slots=True)
class Condensat:
    """`rho_l`, `rho_v`, `mu_l`, `lambda_l` et la chaleur latente `l_v`."""

    masse_volumique: float
    vapeur: float
    viscosite: float
    conductivite: float
    chaleur_latente: float

    def poids_apparent(self) -> float:
        """`(rho_l - rho_v) g`, en N/m^3."""
        return (self.masse_volumique - self.vapeur) * PESANTEUR

    def groupe(self) -> float:
        """`rho_l (rho_l - rho_v) g lambda^3 / mu`."""
        moteur = self.masse_volumique * self.poids_apparent()
        return moteur * self.conductivite**3 / self.viscosite

    def echelle(self) -> float:
        """`[mu^2 / (rho_l (rho_l - rho_v) g lambda^3)]^1/3`, en m^2 K/W."""
        return float((self.viscosite / self.groupe()) ** _TIERS)

    def avec_chaleur_latente(self, chaleur_latente: float) -> Condensat:
        """Le meme condensat, sous une autre chaleur latente."""
        return replace(self, chaleur_latente=chaleur_latente)

    def sous_refroidi(self, chaleur_massique: float, ecart: float) -> Condensat:
        """La chaleur latente multipliee par `1 + 0,375 c_l (t_sat - t_p) / l_v`."""
        corrigee = self.chaleur_latente + 0.375 * chaleur_massique * ecart
        return self.avec_chaleur_latente(corrigee)

    def epaisseur_du_film(self, abscisse: float, ecart: float) -> float:
        """`delta = [4 lambda^4 (t_sat - t_p) x / (l_v X)]^1/4`, `X` etant le groupe."""
        conduit = 4 * self.conductivite**4 * ecart * abscisse
        return float((conduit / (self.chaleur_latente * self.groupe())) ** 0.25)

    def coefficient_local(self, abscisse: float, ecart: float) -> float:
        """`K(x) = lambda / delta`."""
        if abscisse <= 0:
            msg = f"abscisse nulle ou negative : {abscisse} m"
            raise ValueError(msg)
        return self.conductivite / self.epaisseur_du_film(abscisse, ecart)

    def coefficient_vertical(self, hauteur: float, ecart: float) -> float:
        """`K_v = (4 / 3) K(H)`, soit `0,943 [...]^1/4`."""
        return 4 / 3 * self.coefficient_local(hauteur, ecart)

    def coefficient_vertical_par_le_debit(self, debit: float) -> float:
        """`K_v = 0,925 [(rho_l - rho_v) rho_l g lambda^3 / (mu G_H)]^1/3`."""
        return float(4 / 3 * (self.groupe() / (3 * debit)) ** _TIERS)


def debit_du_film(
    condensat: Condensat, epaisseur: float, *, inclinaison: float = math.pi / 2
) -> float:
    """`G`, en kg/(s m), sur une paroi inclinee de `inclinaison` sur l'horizontale."""
    moteur = condensat.masse_volumique * condensat.poids_apparent()
    return moteur * epaisseur**3 * math.sin(inclinaison) / (3 * condensat.viscosite)
