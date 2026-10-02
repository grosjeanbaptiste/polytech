"""Les circuits magnetiques : reluctance et loi d'Hopkinson.

L'analogie avec les circuits electriques est ce qui rend le chapitre
calculable : la force magnetomotrice `N I` joue le role de la tension, le
flux celui du courant, et la reluctance `l / (mu S)` celle de la
resistance. Les reluctances en serie s'additionnent.

Mais l'analogie a une limite que le chapitre insiste a nommer : `mu` n'est
pas constante. Dans un materiau ferromagnetique on **lit** `H` sur la
courbe `B(H)` au lieu de le calculer -- c'est exactement ce que fait
l'exercice 6, qui part de `B_H = 1,6 T` et lit `H_H = 70 A/m`. La
bibliotheque accepte donc un `H` lu, et ne pretend pas le deduire.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE


def reluctance(longueur: float, section: float, relative: float = 1.0) -> float:
    """`l / (mu_r mu_0 S)`, l'analogue magnetique de la resistance."""
    if longueur <= 0 or section <= 0:
        msg = f"geometrie invalide : longueur {longueur} m, section {section} m^2"
        raise ValueError(msg)
    if relative <= 0:
        msg = f"permeabilite relative nulle ou negative : {relative}"
        raise ValueError(msg)
    return longueur / (relative * PERMEABILITE_DU_VIDE * section)


def flux_par_hopkinson(force_magnetomotrice: float, reluctance_totale: float) -> float:
    """`Phi = N I / R` : la loi d'Hopkinson, analogue de la loi d'Ohm."""
    if reluctance_totale <= 0:
        msg = f"reluctance nulle ou negative : {reluctance_totale} A/Wb"
        raise ValueError(msg)
    return force_magnetomotrice / reluctance_totale


@dataclass(frozen=True, slots=True)
class BrancheFerromagnetique:
    """Une branche dont `H` est **lu** sur la courbe du materiau, non calcule.

    C'est la difference avec un entrefer : dans l'air `mu` vaut `mu_0` et
    `H` se calcule ; dans le fer `mu` depend de `B`, et seule la courbe du
    materiau donne `H`. Melanger les deux est l'erreur du chapitre.
    """

    induction: float
    champ_lu: float
    section: float
    longueur: float

    def __post_init__(self) -> None:
        """Refuse une geometrie nulle."""
        if self.section <= 0 or self.longueur <= 0:
            msg = (
                f"geometrie invalide : section {self.section} m^2, "
                f"longueur {self.longueur} m"
            )
            raise ValueError(msg)

    def flux(self) -> float:
        """`Phi = B S`."""
        return self.induction * self.section

    def amperes_tours(self) -> float:
        """`H l` : la part de force magnetomotrice que cette branche consomme."""
        return self.champ_lu * self.longueur

    def courant_pour(self, spires: int) -> float:
        """`I = H l / N` : le courant qu'il faut pour obtenir cette induction."""
        if spires <= 0:
            msg = f"nombre de spires nul ou negatif : {spires}"
            raise ValueError(msg)
        return self.amperes_tours() / spires


class Magnetisme(Enum):
    """Les trois comportements que le fascicule distingue."""

    DIAMAGNETIQUE = "diamagnetique"
    PARAMAGNETIQUE = "paramagnetique"
    FERROMAGNETIQUE = "ferromagnetique"


SEUIL_FERROMAGNETIQUE = 10.0
"""Au-dela de cette permeabilite relative, le materiau est ferromagnetique.

Le seuil est choisi et dit : les corps paramagnetiques plafonnent a
quelques millionnemes au-dessus de l'unite, les ferromagnetiques montent a
plusieurs milliers. Dix separe les deux familles sans ambiguite, et aucun
materiau usuel ne tombe entre.
"""


def classe_de_magnetisme(relative: float) -> Magnetisme:
    """Le comportement magnetique que cette permeabilite relative designe.

    Sous l'unite le materiau **repousse** le champ -- c'est le
    diamagnetisme, et il est universel mais faible. Juste au-dessus, le
    paramagnetisme. Tres au-dessus, le ferromagnetisme, ou la permeabilite
    cesse d'etre une constante : le fascicule note qu'elle « passe par un
    maximum ».
    """
    if relative <= 0:
        msg = f"permeabilite relative nulle ou negative : {relative}"
        raise ValueError(msg)
    if relative < 1:
        return Magnetisme.DIAMAGNETIQUE
    if relative < SEUIL_FERROMAGNETIQUE:
        return Magnetisme.PARAMAGNETIQUE
    return Magnetisme.FERROMAGNETIQUE


def pertes_par_hysteresis(
    aire_du_cycle: float,
    frequence: float,
    volume: float,
) -> float:
    """`f x aire x volume`, en watts.

    L'aire du cycle est une energie par unite de volume : la parcourir une
    fois dissipe cette energie, donc la parcourir `f` fois par seconde
    dissipe cette puissance. C'est pourquoi un transformateur chauffe huit
    fois plus a 400 Hz qu'a 50 Hz, et pourquoi on empile des toles minces
    de materiau a cycle etroit.
    """
    if frequence < 0:
        msg = f"frequence negative : {frequence} Hz"
        raise ValueError(msg)
    if aire_du_cycle < 0 or volume < 0:
        msg = f"aire {aire_du_cycle} J/m^3 ou volume {volume} m^3 negatif"
        raise ValueError(msg)
    return aire_du_cycle * frequence * volume


def potentiel_scalaire_magnetique(champ: float, longueur: float) -> float:
    """`H l` entre deux points : l'analogue du potentiel electrique.

    Le fascicule lui consacre une planche, et sa limite est ce qui compte :
    il n'existe que dans les regions **sans courant**. Ailleurs `H` n'est
    pas a circulation nulle, aucun potentiel ne le decrit, et c'est tout
    l'ecart entre le magnetisme et l'electrostatique.
    """
    if longueur < 0:
        msg = f"longueur negative : {longueur} m"
        raise ValueError(msg)
    return champ * longueur


def champ_pour_les_amperes_tours(amperes_tours: float, longueur: float) -> float:
    """`H = N I / l` : le champ d'une branche sous force magnetomotrice donnee.

    Les ampere-tours sont **partages** entre les branches : le theoreme
    d'Ampere sur deux contours entourant le meme enroulement donne la
    meme force magnetomotrice. Connaissant `H l` d'une branche, on obtient
    donc le champ de l'autre sans rien savoir de son materiau -- et c'est
    ce qui permet de remonter a son induction par la courbe.
    """
    if longueur <= 0:
        msg = f"longueur de contour nulle ou negative : {longueur} m"
        raise ValueError(msg)
    return amperes_tours / longueur


def inductance_depuis_les_flux(
    spires: int,
    flux_de_branche: tuple[float, ...],
    courant: float,
) -> float:
    """`L = N Phi_total / I`, en henrys.

    Le flux **total**, somme des flux de branche, et non celui d'une
    seule : sur le circuit de l'exercice 6, ne garder que la branche sans
    entrefer donnerait une inductance sept fois trop petite, et ne garder
    que l'autre la sous-estimerait de quinze pour cent -- ce qui est bien
    plus insidieux qu'un facteur sept.
    """
    if courant <= 0:
        msg = f"courant nul ou negatif : {courant} A"
        raise ValueError(msg)
    if not flux_de_branche:
        msg = "aucun flux de branche : le circuit ne porte pas d'inductance"
        raise ValueError(msg)
    if spires <= 0:
        msg = f"nombre de spires nul ou negatif : {spires}"
        raise ValueError(msg)
    return spires * sum(flux_de_branche) / courant
