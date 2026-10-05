"""L'ailette droite : son profil, le flux qu'elle cede, ce qu'elle ajoute.

Syllabus de thermique, tome 1, premiere partie, section 2.3. Un corps
allonge, de section `S` et de perimetre `P`, fixe sur une surface a
`t_1`, echange avec un fluide a `t_f`. Le gradient etant surtout
longitudinal, le probleme est a une dimension, et la perte laterale joue
le role d'une source negative : avec `theta = t - t_f`,

`theta'' - m^2 theta = 0`, `m = racine(K P / (lambda S))`.

En negligeant le flux du bout, `theta / theta_1 = cosh(m (l - x)) /
cosh(m l)`. Le flux cede au fluide est celui qui entre par le pied,
`Phi_1 = racine(K P lambda S) theta_1 tanh(m l)`. Rapporte a celui d'une
ailette qui serait partout a `t_1`, il donne l'efficacite
`Omega = tanh(m l) / (m l)` : ajouter une surface ailetee `S_a` revient a
ajouter `Omega S_a` de surface primaire.
"""

from __future__ import annotations

import math

import sympy as sp

type Grandeur = sp.Expr | float


def parametre_d_ailette(
    echange: float, perimetre: float, conductivite: float, section: float
) -> float:
    """`m = racine(K P / (lambda S))`, en m^-1."""
    if section <= 0:
        msg = f"section nulle ou negative : {section} m^2"
        raise ValueError(msg)
    return math.sqrt(echange * perimetre / (conductivite * section))


def parametre_d_ailette_mince(
    echange: float, conductivite: float, epaisseur: float
) -> float:
    """`m = racine(2 K / (lambda b))`, pour une plaque d'epaisseur `b`."""
    return parametre_d_ailette(echange, 2.0, conductivite, epaisseur)


def profil_d_ailette(
    abscisse: Grandeur, parametre: Grandeur, longueur: Grandeur
) -> sp.Expr:
    """`theta / theta_1 = cosh(m (l - x)) / cosh(m l)`."""
    return sp.cosh(parametre * (longueur - abscisse)) / sp.cosh(parametre * longueur)


def flux_au_pied(
    parametre: float, conductance: float, *, ecart: float, longueur: float
) -> float:
    """`Phi_1 = lambda S theta_1 m tanh(m l)`, `conductance` etant `lambda S`."""
    return conductance * ecart * parametre * math.tanh(parametre * longueur)


def flux_sans_resistance(
    echange: float, perimetre: float, *, ecart: float, longueur: float
) -> float:
    """`K theta_1 P l` : ce que cederait une ailette de conductivite infinie."""
    return echange * ecart * perimetre * longueur


def flux_d_une_surface_ailetee(
    echange: float,
    ecart: float,
    *,
    primaire: float,
    secondaire: float,
    efficacite: float,
) -> float:
    """`Phi = K theta_1 (S_0 + Omega S_a)`."""
    return echange * ecart * (primaire + efficacite * secondaire)
