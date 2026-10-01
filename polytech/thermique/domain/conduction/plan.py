"""La couche plane avec generation interne, entre deux parois a echange.

C'est le tas de charbon du TP 1, et c'est un seul modele pour les deux cas
de l'enonce. Le corrige officiel traite « sol adiabatique » et « pertes
vers le sol » sur des planches separees ; ce sont les deux valeurs d'un
meme parametre, puisque la condition de Robin

    lambda dt/dx(0) = K_bas (t(0) - t_sol)

donne exactement `dt/dx(0) = 0` quand `K_bas = 0`.

Convention de signe, et c'est la que l'on se trompe : `x` monte du sol
vers la surface exposee, la densite de flux dans le sens de `x` vaut
`-lambda dt/dx`. En bas, le flux **sort** dans le sens des `x` decroissants,
d'ou le signe oppose entre les deux conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.thermique.domain.conduction.symboles import (
    ABSCISSE,
    CONSTANTES,
    TEMPERATURE,
)
from polytech.thermique.domain.grandeurs import ZERO_ABSOLU, Temperature

if TYPE_CHECKING:
    from polytech.thermique.domain.echanges import (
        CoefficientDEchange,
        DensiteDeFlux,
    )
    from polytech.thermique.domain.materiaux import (
        Conductivite,
        Epaisseur,
        GenerationVolumique,
    )


@dataclass(frozen=True, slots=True)
class CoucheAvecGeneration:
    """Couche plane d'epaisseur donnee, siege d'une generation uniforme."""

    conductivite: Conductivite
    generation: GenerationVolumique
    epaisseur: Epaisseur
    echange_bas: CoefficientDEchange
    temperature_bas: Temperature
    echange_haut: CoefficientDEchange
    temperature_haut: Temperature
    flux_absorbe: DensiteDeFlux

    def loi_locale(self) -> sp.Eq:
        """Fourier-Kirchhoff stationnaire, unidimensionnel, avec source."""
        derivee = sp.Derivative(TEMPERATURE(ABSCISSE), ABSCISSE, 2)
        lam = self.conductivite.watts_par_m_k
        return sp.Eq(lam * derivee + self.generation.watts_par_m3, 0)

    def profil_general(self) -> sp.Expr:
        """Le profil apres deux integrations : une parabole a sommet haut."""
        courbure = self.generation.watts_par_m3 / self.conductivite.watts_par_m_k
        premiere, seconde = CONSTANTES
        return -courbure * ABSCISSE**2 / 2 + premiere * ABSCISSE + seconde

    def gradient_general(self) -> sp.Expr:
        """La derivee du profil, avant que les constantes soient connues."""
        return sp.diff(self.profil_general(), ABSCISSE)

    def condition_au_bas(self) -> sp.Eq:
        """Le flux descendant egale la perte vers le sol (Robin en x = 0)."""
        lam = self.conductivite.watts_par_m_k
        descendant = lam * self.gradient_general().subs(ABSCISSE, 0)
        ecart = self.profil_general().subs(ABSCISSE, 0) - self.temperature_bas.kelvins
        return sp.Eq(descendant, self.echange_bas.watts_par_m2_k * ecart)

    def condition_au_sommet(self) -> sp.Eq:
        """Conduction plus flux absorbe egalent la perte vers l'air (x = e)."""
        lam = self.conductivite.watts_par_m_k
        epaisseur = self.epaisseur.metres
        conduit = -lam * self.gradient_general().subs(ABSCISSE, epaisseur)
        ecart = (
            self.profil_general().subs(ABSCISSE, epaisseur)
            - self.temperature_haut.kelvins
        )
        arrive = conduit + self.flux_absorbe.watts_par_m2
        return sp.Eq(arrive, self.echange_haut.watts_par_m2_k * ecart)

    def constantes(self) -> dict[sp.Symbol, sp.Expr]:
        """Les deux constantes, tirees des deux conditions aux limites."""
        systeme = [self.condition_au_bas(), self.condition_au_sommet()]
        resolu = sp.solve(systeme, CONSTANTES, dict=True)
        if len(resolu) != 1:
            msg = f"les conditions aux limites n'ont pas de solution unique : {resolu}"
            raise ValueError(msg)
        premiere, seconde = CONSTANTES
        trouve = resolu[0]
        return {premiere: trouve[premiere], seconde: trouve[seconde]}

    def profil(self) -> sp.Expr:
        """Le profil de temperature, constantes substituees, en kelvins."""
        return sp.expand(self.profil_general().subs(self.constantes()))

    def profil_en_celsius(self) -> sp.Expr:
        """Le meme profil, en degres Celsius.

        L'enonce donne le charbon en Celsius et l'hydrogene en kelvins ; la
        bibliotheque calcule en kelvins pour n'avoir qu'une echelle, et ce
        profil-ci est celui que le corrige affiche. Tout le document reste
        ainsi dans l'echelle de son enonce.
        """
        return sp.expand(self.profil() - ZERO_ABSOLU)

    def temperature_en(self, abscisse: float) -> Temperature:
        """La temperature a cette profondeur dans la couche."""
        return Temperature(kelvins=float(self.profil().subs(ABSCISSE, abscisse)))

    def abscisse_du_sommet(self) -> float:
        """L'abscisse ou le gradient s'annule -- le sommet de la parabole."""
        if self.generation.watts_par_m3 == 0.0:
            msg = "sans generation interne le profil est affine : pas de sommet"
            raise ValueError(msg)
        gradient = sp.diff(self.profil(), ABSCISSE)
        return float(sp.solve(sp.Eq(gradient, 0), ABSCISSE)[0])

    def abscisse_du_maximum(self) -> float:
        """Le maximum sur la couche : le sommet, ramene dans [0, e] s'il en sort.

        Le ramener n'est pas une commodite : une parabole a sommet haut est
        monotone de part et d'autre, donc son maximum sur un segment qui
        n'en contient pas le sommet est atteint au bord le plus proche.
        """
        return min(max(self.abscisse_du_sommet(), 0.0), self.epaisseur.metres)

    def temperature_maximale(self) -> Temperature:
        """La temperature la plus haute atteinte dans la couche."""
        return self.temperature_en(self.abscisse_du_maximum())

    def flux_vers_le_bas(self) -> float:
        """La densite de flux qui part vers le sol, en W/m^2."""
        lam = self.conductivite.watts_par_m_k
        return float(lam * sp.diff(self.profil(), ABSCISSE).subs(ABSCISSE, 0))

    def flux_conductif_au_sommet(self) -> float:
        """La densite de flux qui arrive par conduction a la surface, en W/m^2."""
        lam = self.conductivite.watts_par_m_k
        gradient = sp.diff(self.profil(), ABSCISSE)
        return float(-lam * gradient.subs(ABSCISSE, self.epaisseur.metres))
