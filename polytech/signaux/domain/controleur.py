"""La synthese du controleur : de la specification aux parametres.

Le cours distingue explicitement l'**analyse** -- on connait `C G1 G2` et
on en deduit les performances -- de la **synthese**, ou `C` est inconnu et
seules les performances souhaitees sont donnees. Sa planche le dit :
« lors de la demarche de synthese du controleur seul `Gs G1 G2` est
connu ! ». C'est cette seconde direction que ce module porte, puisque la
premiere est deja celle de `harmonique` et de `performances`.

Un correcteur proportionnel-integral annule l'erreur statique parce qu'il
ajoute un pole a l'origine. La verification est donc intrinseque : apres
synthese, l'erreur statique doit valoir zero -- et le test l'exige au lieu
de le supposer.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE
from polytech.signaux.domain.transmittance import Transmittance


@dataclass(frozen=True, slots=True)
class Proportionnel:
    """`C(p) = K` : le plus simple, et celui qui laisse une erreur statique."""

    gain: sp.Expr

    def transmittance(self) -> Transmittance:
        """Le correcteur comme transmittance, pour l'assembler a la chaine."""
        return Transmittance(expression=self.gain)


@dataclass(frozen=True, slots=True)
class ProportionnelIntegral:
    """`C(p) = K (1 + 1/(T p))`, le correcteur PI du cours."""

    gain: sp.Expr
    temps_integral: sp.Expr

    def __post_init__(self) -> None:
        """Refuse un temps integral nul : il ferait exploser le correcteur."""
        if sp.simplify(self.temps_integral) == 0:
            msg = "temps integral nul : le correcteur PI n'est pas defini"
            raise ValueError(msg)

    def transmittance(self) -> Transmittance:
        """Le correcteur comme transmittance."""
        return Transmittance(
            expression=self.gain * (1 + 1 / (self.temps_integral * IMAGE)),
        )

    def erreur_statique(self, chaine: Transmittance) -> sp.Expr:
        """L'erreur statique de la boucle corrigee : `1 / (1 + C G)` en zero.

        Elle doit valoir zero -- c'est la raison d'etre de l'integrateur.
        La calculer plutot que l'affirmer est ce qui distingue une synthese
        verifiee d'une synthese esperee.
        """
        corrigee = self.transmittance().en_serie_avec(chaine)
        return sp.limit(1 / (1 + corrigee.expression), IMAGE, 0, "+")


@dataclass(frozen=True, slots=True)
class ProportionnelIntegralDerive:
    """`C(p) = G (1 + 1/(T_i p) + T_d p)`, le PID du cours.

    « Comme seuls trois parametres Gc, Ti, Td disponibles », dit la
    planche -- et c'est ce qui borne la synthese : trois parametres ne
    peuvent satisfaire que trois contraintes. La demarche du fascicule les
    choisit donc dans un ordre precis, en partant des specifications
    temporelles traduites en contraintes frequentielles.
    """

    gain: sp.Expr
    temps_integral: sp.Expr
    temps_derive: sp.Expr

    def __post_init__(self) -> None:
        """Refuse un temps integral nul et un temps derive negatif."""
        if sp.simplify(self.temps_integral) == 0:
            msg = "temps integral nul : le terme integral n'est pas defini"
            raise ValueError(msg)
        if sp.simplify(self.temps_derive) < 0:
            msg = f"temps derive negatif : {self.temps_derive}"
            raise ValueError(msg)

    def parametres(self) -> int:
        """Trois, et c'est ce qui borne la synthese."""
        return 3

    def transmittance(self) -> Transmittance:
        """Le correcteur comme transmittance, pour l'assembler a la chaine."""
        return Transmittance(
            expression=self.gain
            * (1 + 1 / (self.temps_integral * IMAGE) + self.temps_derive * IMAGE),
        )

    def erreur_statique(self, chaine: Transmittance) -> sp.Expr:
        """Nulle, comme pour le PI : le terme derive n'agit pas en continu."""
        corrigee = self.transmittance().en_serie_avec(chaine)
        return sp.limit(1 / (1 + corrigee.expression), IMAGE, 0, "+")


def pulsation_de_correction(
    chaine: Transmittance,
    marge_visee: sp.Expr,
) -> sp.Expr:
    """La pulsation ou l'argument de la chaine vaut `-pi + marge visee`.

    « Comment choisir la pulsation de correction ? Comme -l'argument... »,
    demande la planche. C'est la que le correcteur doit agir, parce que
    c'est la que la marge de phase se lira.
    """
    pulsation = sp.Symbol("omega_c", positive=True)
    frequentielle = chaine.expression.subs(IMAGE, sp.I * pulsation)
    argument = sp.atan2(sp.im(frequentielle), sp.re(frequentielle))
    cible = -sp.pi + marge_visee
    racines = sp.solve(sp.Eq(sp.simplify(argument), cible), pulsation)
    positives = [r for r in racines if r.is_positive]
    if not positives:
        msg = f"aucune pulsation ne donne un argument de {cible}"
        raise ValueError(msg)
    return sp.simplify(positives[0])


def contraintes_frequentielles(
    depassement: float,
    temps_de_montee: float,
) -> tuple[sp.Expr, sp.Expr]:
    """Des specifications temporelles aux contraintes frequentielles.

    « Des specifications temporelles vers les caracteristiques
    frequentielles » : c'est la traduction que la synthese exige, parce
    qu'un correcteur se regle dans le domaine frequentiel alors que le
    cahier des charges est temporel.

    Le depassement devient une **marge de phase** -- via l'amortissement
    qu'il implique -- et le temps de montee une **bande passante**. Les
    deux relations sont celles du second ordre, donc approchees au-dela, et
    c'est pourquoi on parle de contraintes et non d'egalites.
    """
    if depassement <= 0:
        msg = (
            f"depassement nul ou negatif : {depassement} -- il exigerait une "
            f"marge de phase de quatre-vingt-dix degres, inatteignable"
        )
        raise ValueError(msg)
    if temps_de_montee <= 0:
        msg = f"temps de montee nul ou negatif : {temps_de_montee} s"
        raise ValueError(msg)
    logarithme = sp.log(sp.Float(depassement))
    amortissement = -logarithme / sp.sqrt(sp.pi**2 + logarithme**2)
    marge = sp.atan(
        2
        * amortissement
        / sp.sqrt(sp.sqrt(1 + 4 * amortissement**4) - 2 * amortissement**2)
    )
    bande = sp.Float(2.2) / sp.Float(temps_de_montee)
    return sp.simplify(marge), bande
