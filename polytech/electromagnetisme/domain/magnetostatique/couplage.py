"""Le couplage magnetique : inductance mutuelle et coefficient de couplage.

`k = M / racine(L1 L2)` ne peut pas depasser l'unite -- un couplage
parfait transfere tout le flux, il n'en cree pas. La garde n'est donc pas
formelle : un `k > 1` signale une donnee mal lue ou une mutuelle calculee
de travers, et le laisser passer donnerait une inductance equivalente
negative.

Le signe de `M` dans la mise en serie depend du sens des enroulements :
`L1 + L2 + 2M` s'ils s'aident, `L1 + L2 - 2M` s'ils s'opposent. C'est le
point que les exercices verifient, et c'est pourquoi le sens est un
argument explicite et non une convention tacite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE


@dataclass(frozen=True, slots=True)
class CouplageMagnetique:
    """Deux inductances et leur mutuelle."""

    premiere: float
    seconde: float
    mutuelle: float

    def __post_init__(self) -> None:
        """Refuse une inductance negative et un couplage superieur a un."""
        for valeur, nom in ((self.premiere, "premiere"), (self.seconde, "seconde")):
            if valeur <= 0:
                msg = f"inductance {nom} nulle ou negative : {valeur} H"
                raise ValueError(msg)
        if abs(self.mutuelle) > math.sqrt(self.premiere * self.seconde):
            msg = (
                f"couplage superieur a l'unite : M = {self.mutuelle} depasse "
                f"racine(L1 L2) = {math.sqrt(self.premiere * self.seconde):.6g}"
            )
            raise ValueError(msg)

    def coefficient(self) -> float:
        """`k = M / racine(L1 L2)`, entre zero et un."""
        return self.mutuelle / math.sqrt(self.premiere * self.seconde)

    def en_serie(self, *, aidant: bool) -> float:
        """`L1 + L2 +- 2M` selon que les enroulements s'aident ou s'opposent."""
        signe = 1 if aidant else -1
        return self.premiere + self.seconde + signe * 2 * self.mutuelle

    def est_parfait(self) -> bool:
        """Vrai si tout le flux de l'un traverse l'autre (`k = 1`)."""
        return math.isclose(self.coefficient(), 1.0)


def inductance_propre_d_un_solenoide(
    spires: int,
    section: float,
    longueur: float,
    relative: float = 1.0,
) -> float:
    """`mu N^2 S / l`, en henrys.

    Le carre des spires est ce qu'on oublie : doubler le bobinage
    quadruple l'inductance, parce que chaque spire voit le flux de toutes
    les autres.
    """
    if spires <= 0:
        msg = f"nombre de spires nul ou negatif : {spires}"
        raise ValueError(msg)
    if section <= 0 or longueur <= 0:
        msg = f"geometrie invalide : section {section} m^2, longueur {longueur} m"
        raise ValueError(msg)
    if relative <= 0:
        msg = f"permeabilite relative nulle ou negative : {relative}"
        raise ValueError(msg)
    permeabilite = relative * PERMEABILITE_DU_VIDE
    return permeabilite * spires**2 * section / longueur


def signe_du_couplage(*, flux_de_meme_sens: bool) -> int:
    """`+1` si les deux flux s'ajoutent, `-1` s'ils s'opposent.

    Le fascicule l'ecrit : « le signe des inductances est lie au sens
    positif des flux », et « l'association d'inductances en serie depend
    du sens du flux ». La meme paire de bobines donne donc deux
    inductances equivalentes selon le branchement, et le signe n'est pas
    une convention libre -- il se lit sur les sens de bobinage.
    """
    return 1 if flux_de_meme_sens else -1


def flux_a_travers_un_circuit(inductance: float, courant: float) -> float:
    """`Phi = L i`, en webers : la definition meme de l'inductance propre.

    Le fascicule part de la pour tout le chapitre III.4, parce que c'est
    cette relation qui lie « les flux aux courants » -- et que c'est elle
    qui rend l'inductance mesurable sans ouvrir le circuit.
    """
    if inductance < 0:
        msg = f"inductance negative : {inductance} H"
        raise ValueError(msg)
    return inductance * courant
