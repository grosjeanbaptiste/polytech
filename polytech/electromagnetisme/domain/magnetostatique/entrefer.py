"""Les forces d'un entrefer, tirees de l'energie -- exercices 15 et 16.

Les deux exercices portent la meme physique sous **deux conventions
opposees**, et c'est leur mise en regard qui les rend instructifs.

L'exercice 15 travaille a **flux constant** -- l'enonce fixe `B_H`, donc
le flux -- et la force vaut `-dW/dl`. L'exercice 16 travaille a **courant
constant**, et elle vaut `+dW/da`. Les deux donnent une force qui
**referme** l'entrefer, ce qui est physiquement obligatoire : un circuit
magnetique tend toujours a se refermer. Les signes opposes des deux
formules ne decrivent donc pas des forces opposees, et c'est ce que le
chapitre IV.2 enseigne.

Un fait de l'exercice 15 n'est pas dans son corrige officiel, et c'est le
plus frappant : les **deux entrefers de 0,1 mm consomment 78 % des
ampere-tours**, contre 22 % pour le metre de fer. Un facteur dix-huit
mille separe les deux champs a induction egale, et c'est la permeabilite
relative du materiau qui le produit. Toute la raison d'etre d'un circuit
magnetique ferme tient dans ce rapport.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE


def champ_dans_l_entrefer(induction: float) -> float:
    """`H_e = B / mu_0`, en A/m.

    Dans l'air la permeabilite vaut `mu_0` et le champ se **calcule** ;
    dans le fer elle depend de `B` et seule la courbe le donne. C'est la
    distinction du chapitre, et elle explique le facteur dix-huit mille
    entre les deux champs a induction egale.
    """
    return induction / PERMEABILITE_DU_VIDE


@dataclass(frozen=True, slots=True)
class CircuitAEntrefers:
    """Un circuit magnetique ferme par un ou plusieurs entrefers.

    Cinq grandeurs et pas cinq arguments : elles decrivent **un** objet,
    et les separer dans chaque signature laissait croire a cinq donnees
    independantes -- alors qu'aucune n'a de sens sans les autres.

    Le nombre d'entrefers est explicite parce que l'exercice 15 en a
    **deux**, les deux demi-tores se touchant en deux endroits. En
    oublier un ferait sous-estimer le courant de quarante pour cent.
    """

    champ_dans_le_fer: float
    longueur_du_fer: float
    induction: float
    epaisseur: float
    entrefers: int

    def __post_init__(self) -> None:
        """Refuse un circuit sans entrefer : ce n'est pas le cas du chapitre."""
        if self.entrefers <= 0:
            msg = f"nombre d'entrefers nul ou negatif : {self.entrefers}"
            raise ValueError(msg)

    def amperes_tours_du_fer(self) -> float:
        """`H_H l_H` : la part que la culasse consomme."""
        return self.champ_dans_le_fer * self.longueur_du_fer

    def amperes_tours_des_entrefers(self) -> float:
        """`n H_e l_e` : la part que les entrefers consomment."""
        return self.entrefers * champ_dans_l_entrefer(self.induction) * self.epaisseur

    def amperes_tours(self) -> float:
        """La force magnetomotrice totale, somme des deux parts."""
        return self.amperes_tours_du_fer() + self.amperes_tours_des_entrefers()

    def courant_d_excitation(self, spires: int) -> float:
        """`I = (H_H l_H + n H_e l_e) / N`, en amperes."""
        if spires <= 0:
            msg = f"nombre de spires nul ou negatif : {spires}"
            raise ValueError(msg)
        return self.amperes_tours() / spires

    def part_des_entrefers(self) -> float:
        """La fraction des ampere-tours que les entrefers consomment.

        La part du fer est fixe -- elle ne depend que de sa longueur et de
        son champ -- tandis que celle de l'air croit avec l'epaisseur.
        C'est pourquoi un entrefer mal maitrise fait deriver tout le
        dimensionnement.
        """
        return self.amperes_tours_des_entrefers() / self.amperes_tours()


def force_a_flux_constant_sur_un_entrefer(induction: float, section: float) -> float:
    """`F = H_e Phi = B^2 S / mu_0`, en newtons.

    La forme `H_e Phi` est celle du manuscrit ; la reecrire en
    `B^2 S / mu_0` montre la dependance en **carre** de l'induction, que
    la premiere cache -- et c'est la forme que tout dimensionnement
    emploie.
    """
    if section <= 0:
        msg = f"section nulle ou negative : {section} m^2"
        raise ValueError(msg)
    return champ_dans_l_entrefer(induction) * induction * section


def force_d_un_electroaimant(
    spires: sp.Expr,
    courant: sp.Expr,
    section: sp.Expr,
    entrefer: sp.Expr,
) -> sp.Expr:
    """`F = +dW/da = -N^2 mu_0 I^2 S / (2 a^2)`, symbolique.

    A courant constant la force vaut `+dW/da`. L'energie decroit quand
    l'entrefer croit, donc la derivee est negative, donc la force tend a
    **refermer** -- meme sens qu'a flux constant, malgre la formule de
    signe oppose.

    La decroissance en `1/a^2` rend un electroaimant tres puissant au
    contact et faible a un millimetre, ce qui le rend difficile a
    asservir.
    """
    permeabilite = sp.Symbol("mu_0", positive=True)
    inductance = spires**2 * permeabilite * section / entrefer
    energie = inductance * courant**2 / 2
    return sp.diff(energie, entrefer)
