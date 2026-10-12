"""Enonce C -- efforts hydrostatiques sur un parement.

La poussee est obtenue par *integration* de la pression relative sur la
surface mouillee :

    F = int_0^H rho g z * largeur(z) dz

Chaque geometrie ne declare donc que sa largeur a la profondeur ``z`` ; la
formule, son exposant en ``H`` et la valeur numerique en decoulent tous, sans
qu'aucune expression ne soit recopiee.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import sympy as sp

from polytech.fluides.domain.grandeurs import PESANTEUR, Hauteur, MasseVolumique
from polytech.fluides.domain.statique.symboles import COTE, PROFONDEUR, RHO, G, H
from polytech.mecanique.domain.statique.charges import ChargeRepartie


def _exiger_fraction(fraction: sp.Expr | float) -> None:
    if not 0 < float(fraction) <= 1:
        msg = f"fraction de volume hors de ]0, 1] : {fraction}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Parement(ABC):
    """Surface mouillee sur laquelle s'exerce la poussee."""

    fluide: MasseVolumique
    pesanteur: float = PESANTEUR

    @abstractmethod
    def largeur_mouillee(self) -> sp.Expr:
        """Largeur mouillee a la profondeur ``h``, hauteur d'eau ``H``."""

    @abstractmethod
    def section_horizontale(self) -> sp.Expr:
        """Aire de la surface libre lorsque la hauteur d'eau vaut ``H``.

        C'est la seule chose qu'une geometrie de bassin ait a declarer : le
        volume stocke et son exposant en ``H`` s'en deduisent par
        integration, comme la poussee se deduit de la largeur mouillee.
        """

    @abstractmethod
    def _valeurs(self) -> dict[sp.Symbol, float]:
        """Substitutions numeriques propres a la geometrie."""

    def _charge_de_pression(self) -> ChargeRepartie:
        """La pression relative `rho g z`, repartie sur la largeur mouillee.

        C'est une charge repartie au sens de la statique : sa charge totale
        est la poussee, son moment a l'origine celui des pressions autour
        de la surface libre.
        """
        densite = RHO * G * PROFONDEUR * self.largeur_mouillee()
        return ChargeRepartie(densite, PROFONDEUR, sp.Integer(0), H)

    def expression_poussee(self) -> sp.Expr:
        """Resultante litterale, integree sur la surface mouillee."""
        return self._charge_de_pression().charge_totale()

    def moment_des_pressions(self) -> sp.Expr:
        """Moment des forces de pression autour de la surface libre."""
        return self._charge_de_pression().moment_a_l_origine()

    def equation_ligne_action(self) -> sp.Eq:
        """Definition de la ligne d'action : ``h_a F = int h dF``.

        La resultante doit produire, seule, le meme moment que la
        distribution de pression entiere. C'est cette egalite qui *definit*
        la ligne d'action ; le quotient n'en est que la resolution.
        """
        return sp.Eq(
            sp.Symbol("h_a") * sp.Symbol("F"),
            self.moment_des_pressions(),
            evaluate=False,
        )

    def expression_ligne_action(self) -> sp.Expr:
        """Profondeur ou passe la resultante, mesuree sous la surface libre.

        C'est le barycentre des forces de pression : le meme integrande que
        la poussee, pondere par la profondeur, divise par la poussee.
        """
        return sp.simplify(self.moment_des_pressions() / self.expression_poussee())

    def expression_bras_de_levier(self) -> sp.Expr:
        """Hauteur de la ligne d'action au-dessus du fond, soit ``H - z_a``.

        Elle vaut ``H / 3`` pour un parement de largeur uniforme, mais
        remonte des que la largeur mouillee croit avec la hauteur : le bras
        se deduit donc de la ligne d'action, il ne se postule pas.
        """
        return sp.simplify(H - self.expression_ligne_action())

    def ligne_action(self, hauteur: Hauteur) -> float:
        """Profondeur numerique de la ligne d'action, en metres."""
        return self._evaluer(self.expression_ligne_action(), hauteur)

    def expression_moment(self) -> sp.Expr:
        """Moment de renversement litteral autour du pied."""
        return sp.simplify(
            self.expression_poussee() * self.expression_bras_de_levier(),
        )

    def exposant_poussee(self) -> int:
        """Degre en ``H`` de la poussee, lu sur l'integrale elle-meme."""
        return int(sp.degree(sp.expand(self.expression_poussee()), H))

    def expression_volume(self) -> sp.Expr:
        """Volume stocke : empilement des sections horizontales de 0 a ``H``."""
        tranche = self.section_horizontale().subs(H, COTE)
        return sp.simplify(sp.integrate(tranche, (COTE, 0, H)))

    def exposant_volume(self) -> int:
        """Degre en ``H`` du volume, lu sur l'integrale elle-meme."""
        return int(sp.degree(sp.expand(self.expression_volume()), H))

    def hauteur_ligne_action(self, hauteur: Hauteur) -> float:
        """Hauteur de la ligne d'action au-dessus du fond, en metres.

        Le cours dit « ligne d'action » et jamais « centre de poussee » :
        le nom de la methode suit son vocabulaire.
        """
        return self._evaluer(self.expression_bras_de_levier(), hauteur)

    def _evaluer(self, expression: sp.Expr, hauteur: Hauteur) -> float:
        valeurs = {
            RHO: self.fluide.kg_par_m3,
            G: self.pesanteur,
            H: hauteur.metres,
            **self._valeurs(),
        }
        return float(expression.subs(valeurs))

    def poussee(self, hauteur: Hauteur) -> float:
        """Resultante numerique, evaluee sur l'integrale litterale."""
        return self._evaluer(self.expression_poussee(), hauteur)

    def moment(self, hauteur: Hauteur) -> float:
        """Moment numerique, evalue sur l'expression litterale."""
        return self._evaluer(self.expression_moment(), hauteur)

    def hauteur_pour_fraction_de_volume(self, fraction: sp.Expr | float) -> sp.Expr:
        """Rapport ``H'/H`` quand le volume est ramene a ``fraction``."""
        _exiger_fraction(fraction)
        return sp.nsimplify(fraction) ** sp.Rational(1, self.exposant_volume())

    def rapport_poussee(self, fraction: sp.Expr | float) -> sp.Expr:
        """Rapport ``F'/F`` exact, sous forme rationnelle."""
        return sp.nsimplify(
            self.hauteur_pour_fraction_de_volume(fraction) ** self.exposant_poussee(),
        )

    def rapport_moment(self, fraction: sp.Expr | float) -> sp.Expr:
        """Rapport ``M'/M`` exact : un degre de plus que la poussee."""
        return sp.nsimplify(
            self.hauteur_pour_fraction_de_volume(fraction)
            ** (self.exposant_poussee() + 1),
        )
