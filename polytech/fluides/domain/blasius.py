"""La solution de Blasius, resolue plutot que citee.

Planches du chapitre IX. Le long d'une plaque plane sous un ecoulement
uniforme, les profils de vitesse sont semblables : `u / U = f'(eta)` avec
`eta = y racine(U / (nu x))`. L'equation de Prandtl devient alors

`2 f''' + f f'' = 0`, `f(0) = f'(0) = 0`, `f'(infini) = 1`.

Elle n'a pas de forme fermee. Si `F` la resout avec `F''(0) = 1` et tend
vers la pente `k`, alors `f(eta) = a F(a eta)` avec `a = 1 / racine(k)`
la resout aussi, et sa pente tend vers un : une seule integration donne
`f''(0) = a^3`, sans tatonnement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

type Etat = tuple[float, float, float]
"""`(f, f', f'')`."""


def variable_de_similitude(
    ordonnee: float, vitesse: float, viscosite: float, abscisse: float
) -> float:
    """`eta = y racine(U / (nu x))`."""
    if abscisse <= 0:
        msg = f"abscisse nulle ou negative : {abscisse} m"
        raise ValueError(msg)
    return ordonnee * math.sqrt(vitesse / (viscosite * abscisse))


def _pente(etat: Etat) -> Etat:
    return (etat[1], etat[2], -etat[0] * etat[2] / 2)


def _avancer(etat: Etat, pas: float) -> Etat:
    """Un pas de Runge-Kutta d'ordre quatre."""

    def decale(pente: Etat, part: float) -> Etat:
        return (
            etat[0] + part * pente[0],
            etat[1] + part * pente[1],
            etat[2] + part * pente[2],
        )

    k1 = _pente(etat)
    k2 = _pente(decale(k1, pas / 2))
    k3 = _pente(decale(k2, pas / 2))
    k4 = _pente(decale(k3, pas))
    somme = tuple(
        a + 2 * b + 2 * c + d for a, b, c, d in zip(k1, k2, k3, k4, strict=True)
    )
    return decale((somme[0], somme[1], somme[2]), pas / 6)


def _integrer(courbure: float, pas: float, nombre: int) -> list[Etat]:
    etats: list[Etat] = [(0.0, 0.0, courbure)]
    for _ in range(nombre):
        etats.append(_avancer(etats[-1], pas))
    return etats


@dataclass(frozen=True, slots=True)
class SolutionDeBlasius:
    """La table `(eta, f, f', f'')`, a pas constant."""

    pas: float
    table: tuple[tuple[float, float, float, float], ...]

    def courbure_a_la_paroi(self) -> float:
        """`f''(0)`, qui donne le frottement."""
        return self.table[0][3]

    def vitesse_reduite(self, eta: float) -> float:
        """`u / U = f'(eta)`, interpolee dans la table."""
        rang = int(eta / self.pas)
        if eta < 0 or rang >= len(self.table) - 1:
            msg = f"eta hors de la table : {eta}"
            raise ValueError(msg)
        part = eta / self.pas - rang
        return (1 - part) * self.table[rang][2] + part * self.table[rang + 1][2]

    def epaisseur_a_99_pour_cent(self) -> float:
        """Le `eta` ou la vitesse atteint 99 pour cent de l'ecoulement libre."""
        seuil = 0.99
        rang = next(k for k, ligne in enumerate(self.table) if ligne[2] >= seuil)
        avant, apres = self.table[rang - 1][2], self.table[rang][2]
        return self.pas * (rang - 1 + (seuil - avant) / (apres - avant))

    def deplacement(self) -> float:
        """`integrale de (1 - f')`, soit `delta* racine(U / (nu x))`."""
        return self.table[-1][0] - self.table[-1][1]

    def quantite_de_mouvement(self) -> float:
        """`integrale de f' (1 - f')`, soit `theta racine(U / (nu x))`."""
        valeurs = [ligne[2] * (1 - ligne[2]) for ligne in self.table]
        return self.pas * (sum(valeurs) - (valeurs[0] + valeurs[-1]) / 2)


def resoudre_blasius(*, etendue: float = 10.0, pas: float = 0.02) -> SolutionDeBlasius:
    """Integre l'equation, la pente a l'infini ramenee a un par changement d'echelle."""
    nombre = round(etendue / pas)
    limite = _integrer(1.0, pas, 2 * nombre)[-1][1]
    etats = _integrer(limite**-1.5, pas, nombre)
    table = tuple((k * pas, *etat) for k, etat in enumerate(etats))
    return SolutionDeBlasius(pas, table)
