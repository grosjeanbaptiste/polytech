"""La cellule de Sallen-Key passe-bas a gain `mu` (chapitre 3, partie II).

Planches 22 et 23 : `R_1` puis `R_2` en serie vers l'entree non inverseuse,
`C_2` de cette entree a la masse, `C_1` du noeud milieu a la sortie, et un
etage non inverseur de gain `mu = 1 + R_b / R_a`. La cellule a suiveur de
`sallen_key.CelluleSallenKey` en est le cas `mu = 1`.

Trois equations -- `omega_0`, `Q`, `G = mu` -- pour six composants : c'est
la liberte que les quatre methodes de `methodes_sallen_key` emploient.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.electronique.domain.etage_de_gain import (
    ENTREE,
    transfert_autour_d_un_gain,
)
from polytech.electronique.domain.netliste import Admittance

_MILIEU, _PLUS, _SORTIE = 2, 3, 4


@dataclass(frozen=True, slots=True)
class SallenKeyPasseBas:
    """Deux resistances, deux capacites et le gain de l'etage."""

    r1: sp.Expr
    r2: sp.Expr
    c1: sp.Expr
    c2: sp.Expr
    gain: sp.Expr

    def __post_init__(self) -> None:
        """Refuse un composant ou un gain dont on sait qu'il n'est pas positif."""
        for nom in ("r1", "r2", "c1", "c2", "gain"):
            valeur = sp.sympify(getattr(self, nom))
            if valeur.is_positive is False:
                msg = f"{nom} nul ou negatif : {valeur}"
                raise ValueError(msg)
            object.__setattr__(self, nom, valeur)

    def transfert(self, variable: sp.Expr) -> sp.Expr:
        """`E_2 / E_1`, par l'analyse nodale du montage de la planche 22."""
        reseau = (
            Admittance(ENTREE, _MILIEU, 1 / self.r1),
            Admittance(_MILIEU, _PLUS, 1 / self.r2),
            Admittance(_MILIEU, _SORTIE, variable * self.c1),
            Admittance(_PLUS, 0, variable * self.c2),
        )
        return transfert_autour_d_un_gain(
            reseau, _SORTIE, commande=_PLUS, sortie=_SORTIE, gain=self.gain
        )

    def pulsation(self) -> sp.Expr:
        """`omega_0 = 1 / racine(R_1 R_2 C_1 C_2)` (planche 23)."""
        return 1 / sp.sqrt(self.r1 * self.r2 * self.c1 * self.c2)

    def amortissement(self) -> sp.Expr:
        """Le coefficient de `s` : `1/R_1C_1 + 1/R_2C_1 + (1 - mu)/R_2C_2`.

        Il vaut `omega_0 / Q`. Nul ou negatif, la cellule oscille : c'est
        le prix d'un gain `mu` trop grand.
        """
        direct = 1 / (self.r1 * self.c1) + 1 / (self.r2 * self.c1)
        return direct + (1 - self.gain) / (self.r2 * self.c2)

    def qualite(self) -> sp.Expr:
        """`Q = omega_0` divise par le coefficient de `s` (planche 23)."""
        return self.pulsation() / self.amortissement()

    def denormalisee(
        self, *, coupure: sp.Expr, impedance: sp.Expr
    ) -> SallenKeyPasseBas:
        """La cellule reelle : `C -> C / (coupure * impedance)`, `R -> impedance * R`.

        Planche 32, etapes 3 et 4 : la transformation passe-bas vers
        passe-bas ne touche qu'aux capacites, la mise a l'echelle des
        impedances multiplie les resistances et divise les capacites.
        Ni l'une ni l'autre ne change `Q` ni le gain.
        """
        if not (coupure > 0 and impedance > 0):
            msg = f"coupure ou facteur d'echelle non positif : {coupure}, {impedance}"
            raise ValueError(msg)
        diviseur = coupure * impedance
        return SallenKeyPasseBas(
            self.r1 * impedance,
            self.r2 * impedance,
            self.c1 / diviseur,
            self.c2 / diviseur,
            self.gain,
        )
