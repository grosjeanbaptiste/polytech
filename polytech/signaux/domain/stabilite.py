"""La stabilite en boucle fermee, par les deux chemins du cours.

Routh-Hurwitz regarde le **denominateur** de la boucle fermee et n'a
besoin d'aucune frequence. Nyquist regarde la **reponse frequentielle** de
la boucle ouverte et n'a besoin d'aucun polynome. Les deux doivent rendre
la meme limite de gain : c'est le recoupement que la feuille de reponses
du cours permet de verifier, puisqu'elle publie les deux cotes.

La garde sur le coefficient nul n'est pas formelle. Le critere de Routh
suppose tous les coefficients presents ; sur `1/p^3` le tableau est
degenere, et en tirer une limite donnerait un nombre faux sans le
paraitre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE, PULSATION

if TYPE_CHECKING:
    from collections.abc import Mapping

    from polytech.signaux.domain.transmittance import Transmittance


@dataclass(frozen=True, slots=True)
class Stabilite:
    """La stabilite de la boucle fermee d'une transmittance donnee."""

    boucle_ouverte: Transmittance

    def _coefficients(self) -> list[sp.Expr]:
        """Les coefficients du denominateur de la boucle fermee, degre decroissant."""
        denominateur = self.boucle_ouverte.en_boucle_fermee().denominateur()
        return list(sp.Poly(denominateur, IMAGE).all_coeffs())

    def condition_de_routh(self) -> sp.Expr:
        """Pour un troisieme degre : `a_1 a_2 - a_0 a_3`, positif si stable."""
        coefficients = self._coefficients()
        attendu = 4
        if len(coefficients) != attendu or any(c == 0 for c in coefficients):
            msg = (
                f"coefficient nul ou degre different de trois : {coefficients} "
                f"-- le tableau de Routh y serait degenere"
            )
            raise ValueError(msg)
        trois, deux, un, zero = coefficients
        return sp.expand(deux * un - trois * zero)

    def gain_critique(self, gain: sp.Symbol) -> sp.Expr:
        """Le gain qui annule la condition de Routh : la limite de stabilite."""
        racines = sp.solve(sp.Eq(self.condition_de_routh(), 0), gain)
        return sp.simplify(racines[0])

    def est_stable(self, valeurs: Mapping[sp.Symbol, sp.Expr]) -> bool:
        """Vrai si la condition de Routh est positive pour ces valeurs."""
        return bool(self.condition_de_routh().subs(dict(valeurs)).is_positive)

    def pulsation_du_croisement(self) -> sp.Expr:
        """La pulsation ou la courbe de Nyquist coupe l'axe reel."""
        imaginaire = sp.im(
            sp.expand(sp.simplify(self.boucle_ouverte.reponse_frequentielle())),
        )
        racines = sp.solve(sp.Eq(sp.simplify(imaginaire), 0), PULSATION)
        positives = [r for r in racines if r.is_positive]
        if not positives:
            msg = "la courbe de Nyquist ne coupe pas l'axe reel"
            raise ValueError(msg)
        return sp.simplify(positives[0])

    def reel_au_croisement(self, valeurs: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
        """L'abscisse du croisement, celle que le corrige lit sur la courbe."""
        frequentielle = self.boucle_ouverte.reponse_frequentielle()
        au_croisement = frequentielle.subs(PULSATION, self.pulsation_du_croisement())
        return sp.simplify(sp.re(sp.simplify(au_croisement.subs(dict(valeurs)))))

    def gain_critique_par_nyquist(self, gain: sp.Symbol) -> sp.Expr:
        """La limite tiree du module au croisement, sans aucun polynome.

        Nyquist dit que la boucle devient instable des que le point de
        croisement atteint `-1`. Si pour `K = 1` il vaut `-a`, la limite
        est `1 / a`.
        """
        unitaire = self.reel_au_croisement({gain: sp.Integer(1)})
        return sp.simplify(1 / abs(unitaire))


def equation_caracteristique(boucle_ouverte: Transmittance) -> sp.Expr:
    """`1 + G(p)`, mis sur denominateur commun : ses racines sont les poles BF.

    C'est la premiere des trois methodes que le fascicule enumere, et la
    seule qui soit une condition **necessaire et suffisante**. Nyquist et
    Bode donnent le meme verdict mais par la frequence, ce qui les rend
    applicables aux systemes a retard -- que le polynome, lui, ne sait pas
    representer.
    """
    ferme = sp.cancel(1 + boucle_ouverte.expression)
    return sp.expand(sp.fraction(ferme)[0])


def est_stable_par_les_racines(boucle_ouverte: Transmittance) -> bool:
    """Vrai si toutes les racines caracteristiques sont a partie reelle negative."""
    racines = sp.solve(sp.Eq(equation_caracteristique(boucle_ouverte), 0), IMAGE)
    if not racines:
        msg = "equation caracteristique sans racine : la stabilite ne se decide pas"
        raise ValueError(msg)
    return all(bool(sp.re(racine) < 0) for racine in racines)


def phase_ajoutee_par_un_retard(duree: sp.Expr, pulsation: sp.Expr) -> sp.Expr:
    """`-omega tau` : un retard ne change pas le module, il mange la phase.

    C'est tout son danger. Un diagramme de gain ne le montre pas, et la
    marge de phase se consomme proportionnellement a la pulsation -- donc
    d'autant plus que la boucle est rapide.
    """
    return -pulsation * duree


def marge_de_phase_avec_retard(
    marge_sans_retard: sp.Expr,
    duree: sp.Expr,
    pulsation: sp.Expr,
) -> sp.Expr:
    """La marge restante une fois le retard pris en compte."""
    return sp.simplify(
        marge_sans_retard + phase_ajoutee_par_un_retard(duree, pulsation)
    )


def retard_critique(marge_sans_retard: sp.Expr, pulsation: sp.Expr) -> sp.Expr:
    """`tau_c = marge / omega` : le retard qui epuise exactement la marge."""
    if sp.simplify(pulsation) == 0:
        msg = "pulsation nulle : un retard n'y consomme aucune phase"
        raise ValueError(msg)
    return sp.simplify(marge_sans_retard / pulsation)


def poles_instables_par_nyquist(encerclements: int, poles_ouverts: int) -> int:
    """`Z = N + P` : le nombre de poles instables en boucle fermee.

    `N` compte les encerclements du point `-1` dans le sens horaire, `P`
    les poles instables en boucle ouverte. C'est un compte **entier**,
    donc exact -- et c'est ce qui fait la force du critere sur un systeme
    a retard, qu'aucun polynome ne representerait.

    Un pole instable en ouvert exige donc exactement un encerclement dans
    le sens **inverse** pour que la boucle fermee soit stable : c'est la
    forme precise du « une boucle peut stabiliser un instable ».
    """
    if poles_ouverts < 0:
        msg = f"nombre de poles ouverts negatif : {poles_ouverts}"
        raise ValueError(msg)
    return encerclements + poles_ouverts


def critere_simplifie_de_nyquist(abscisse_du_croisement: float) -> bool:
    """Vrai si la courbe n'entoure pas le point `-1`.

    « Critere simplifie d'application : stable en BF car n'entoure pas... »
    C'est la forme que le fascicule emploie en pratique, et elle ne vaut
    que pour un systeme **stable en boucle ouverte** -- sinon il faut
    compter les encerclements, et `poles_instables_par_nyquist` est la pour
    cela.

    La garde sur un croisement positif n'est pas formelle : la courbe de
    Nyquist d'un systeme a gain negatif croise l'axe reel du cote positif,
    et le critere simplifie n'y dit rien.
    """
    if abscisse_du_croisement > 0:
        msg = (
            f"croisement positif : {abscisse_du_croisement} -- le critere "
            f"simplifie ne porte que sur l'axe reel negatif"
        )
        raise ValueError(msg)
    return abscisse_du_croisement > -1
