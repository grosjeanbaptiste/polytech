"""Le modele de bruit en tension et en courant d'un quadripole lineaire.

Chapitre 4, partie II, planches 30 a 64. Tout quadripole lineaire bruyant
se remplace par un quadripole sans bruit, precede d'une source de tension
`V_n` et d'une source de courant `I_n`, supposees decorrelees. Devant une
source de resistance `R_S`, le bruit ramene a l'entree a pour densite

    S_NN(f) = 4 k_B T R_S + S_VnVn(f) + R_S^2 S_InIn(f).

Le **facteur de bruit** compare ce total au seul bruit de la source :
`F = 1 + (V_n^2 + R_S^2 I_n^2) / (4 k_B T R_S B)`. Il est minimal pour
`R_S = V_n / I_n`, sans que le bruit, lui, y soit minimal.

Le bruit d'un composant est le plus souvent blanc plus rose, `P_w (1 +
f_c / f)` : la frequence de coin `f_c` est celle ou les deux se valent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.electronique.domain.bruit_thermique import densite_thermique
from polytech.electronique.domain.facteur_de_bruit import TEMPERATURE_DE_REFERENCE


@dataclass(frozen=True, slots=True)
class ModeleBlancEtRose:
    """`S(f) = P_w + K / f`, donne par son palier blanc et sa frequence de coin."""

    blanc: float
    coin: float

    def __post_init__(self) -> None:
        """Refuse un palier negatif ou une frequence de coin negative."""
        if self.blanc < 0:
            msg = f"densite blanche negative : {self.blanc}"
            raise ValueError(msg)
        if self.coin < 0:
            msg = f"frequence de coin negative : {self.coin} Hz"
            raise ValueError(msg)

    @classmethod
    def depuis_deux_points(
        cls, premier: tuple[float, float], second: tuple[float, float]
    ) -> ModeleBlancEtRose:
        """Le modele passant par deux `(f, e_n)` d'une fiche technique."""
        (f1, e1), (f2, e2) = premier, second
        if f1 <= 0 or f2 <= 0 or f1 == f2:
            msg = f"frequences invalides ou confondues : {f1} et {f2} Hz"
            raise ValueError(msg)
        rose = (e1**2 - e2**2) / (1 / f1 - 1 / f2)
        blanc = e1**2 - rose / f1
        return cls(blanc=blanc, coin=rose / blanc)

    def coefficient_rose(self) -> float:
        """`K = P_w f_c`."""
        return self.blanc * self.coin

    def densite(self, frequence: float) -> float:
        """`P_w (1 + f_c / f)`."""
        if frequence <= 0:
            msg = f"frequence nulle ou negative : {frequence} Hz"
            raise ValueError(msg)
        return self.blanc * (1 + self.coin / frequence)

    def puissance(self, basse: float, haute: float) -> float:
        """`P_w (f_h - f_l + f_c ln(f_h / f_l))`."""
        if basse <= 0 or haute <= basse:
            msg = f"bande invalide : {basse} Hz a {haute} Hz"
            raise ValueError(msg)
        return self.blanc * (haute - basse + self.coin * math.log(haute / basse))


def densite_ramenee_a_l_entree(
    source: float,
    en_tension: float,
    en_courant: float,
    *,
    temperature: float = TEMPERATURE_DE_REFERENCE,
) -> float:
    """`4 k_B T R_S + S_Vn + R_S^2 S_In`, pour une entree d'impedance infinie."""
    if en_tension < 0 or en_courant < 0:
        msg = f"densite negative : {en_tension} V^2/Hz, {en_courant} A^2/Hz"
        raise ValueError(msg)
    return densite_thermique(source, temperature) + en_tension + source**2 * en_courant


def facteur_de_bruit_du_modele(
    tension: float,
    courant: float,
    source: float,
    *,
    bande: float,
    temperature: float = TEMPERATURE_DE_REFERENCE,
) -> float:
    """`F = 1 + (V_n^2 + R_S^2 I_n^2) / (4 k_B T R_S B)`, en valeurs efficaces."""
    thermique = densite_thermique(source, temperature) * bande
    if thermique <= 0:
        msg = "source sans bruit thermique : le facteur de bruit n'est pas defini"
        raise ValueError(msg)
    return 1 + (tension**2 + (source * courant) ** 2) / thermique


def resistance_de_source_optimale(tension: float, courant: float) -> float:
    """`R_S = V_n / I_n` : celle qui rend le facteur de bruit minimal."""
    if courant <= 0:
        msg = "bruit en courant nul : le facteur decroit sans fin avec R_S"
        raise ValueError(msg)
    return tension / courant
