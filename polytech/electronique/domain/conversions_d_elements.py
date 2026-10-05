"""Les tables de conversion de composants des transformations de frequence.

Chapitre 3, partie I, planches 86 a 131. Une transformation `s -> f(s)`
de la fonction de transfert se realise composant par composant : chaque
inductance `L`, d'impedance `s L`, devient la branche d'impedance
`f(s) L` ; chaque capacite `C` celle d'impedance `1 / (f(s) C)`. Les
resistances ne dependent pas de `s` et ne bougent pas.

* passe-bas, `s -> (Omega_c / w_c) s` : `L` et `C` multiplies ;
* passe-haut, `s -> Omega_c w_c / s` : `L` et `C` echangent leur nature ;
* passe-bande, `s -> Omega_c (s^2 + w_0^2) / (B s)` : chacun devient un
  circuit resonant, serie pour `L`, parallele pour `C` ;
* coupe-bande, l'inverse : parallele pour `L`, serie pour `C`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True)
class Branche:
    """Une inductance, une capacite, ou les deux -- en serie ou en parallele."""

    inductance: float | None = None
    capacite: float | None = None
    serie: bool = True

    def __post_init__(self) -> None:
        """Refuse une branche vide ou un element non positif."""
        valeurs = [v for v in (self.inductance, self.capacite) if v is not None]
        if not valeurs:
            msg = "branche sans element : ni inductance ni capacite"
            raise ValueError(msg)
        if any(valeur <= 0 for valeur in valeurs):
            msg = f"element nul ou negatif dans la branche : {valeurs}"
            raise ValueError(msg)

    def impedance(self, variable: sp.Expr) -> sp.Expr:
        """`s L`, `1 / (s C)`, leur somme en serie ou leur mise en parallele."""
        parties = []
        if self.inductance is not None:
            parties.append(variable * sp.nsimplify(self.inductance, rational=True))
        if self.capacite is not None:
            parties.append(1 / (variable * sp.nsimplify(self.capacite, rational=True)))
        if self.serie:
            return sum(parties, sp.Integer(0))
        return 1 / sum((1 / partie for partie in parties), sp.Integer(0))

    def element(self) -> tuple[float | None, float | None]:
        """`(L, C)` d'une branche a un seul element ; leve si elle en a deux."""
        if self.inductance is not None and self.capacite is not None:
            msg = "un seul element reactif attendu : la branche en porte deux"
            raise ValueError(msg)
        return self.inductance, self.capacite


def changer_la_coupure(
    branche: Branche, coupure: float, *, prototype: float = 1.0
) -> Branche:
    """`L -> (Omega_c / w_c) L`, `C -> (Omega_c / w_c) C`."""
    inductance, capacite = branche.element()
    rapport = prototype / coupure
    if inductance is not None:
        return Branche(inductance=rapport * inductance)
    return Branche(capacite=rapport * (capacite or 0.0))


def en_passe_haut(
    branche: Branche, coupure: float, *, prototype: float = 1.0
) -> Branche:
    """`C -> L = 1 / (w_c Omega_c C)`, `L -> C = 1 / (w_c Omega_c L)`."""
    inductance, capacite = branche.element()
    produit = coupure * prototype
    if inductance is not None:
        return Branche(capacite=1 / (produit * inductance))
    return Branche(inductance=1 / (produit * (capacite or 0.0)))


def en_passe_bande(
    branche: Branche, centre: float, largeur: float, *, prototype: float = 1.0
) -> Branche:
    """`L -> L Omega_c / B` en serie avec `B / (w_0^2 Omega_c L)` ; dual pour `C`."""
    inductance, capacite = branche.element()
    if inductance is not None:
        grande = inductance * prototype / largeur
        return Branche(grande, largeur / (centre**2 * prototype * inductance))
    valeur = capacite or 0.0
    petite = largeur / (centre**2 * prototype * valeur)
    return Branche(petite, valeur * prototype / largeur, serie=False)


def en_coupe_bande(
    branche: Branche, centre: float, largeur: float, *, prototype: float = 1.0
) -> Branche:
    """`L -> Omega_c B L / w_0^2` en parallele avec `1 / (B Omega_c L)` ; dual."""
    inductance, capacite = branche.element()
    if inductance is not None:
        petite = prototype * largeur * inductance / centre**2
        return Branche(petite, 1 / (largeur * prototype * inductance), serie=False)
    valeur = capacite or 0.0
    grande = 1 / (largeur * prototype * valeur)
    return Branche(grande, prototype * largeur * valeur / centre**2)
