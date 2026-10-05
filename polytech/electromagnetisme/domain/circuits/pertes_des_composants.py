"""Les pertes d'un condensateur, d'une bobine et d'une resistance reels.

Le syllabus d'electricite generale (16.23 a 16.25). Un condensateur reel
dissipe : il equivaut a une capacite pure en serie avec une resistance de
pertes, `U = (R_s - j / (omega C_s)) I` (16.23.1), ou en parallele avec
une autre (16.23.4). Son facteur de dissipation est
`d = tg(delta) = omega C_s R_s` (16.23.3), le rapport de sa puissance
active a la valeur absolue de sa puissance reactive.

Une bobine reelle se represente de meme, avec le facteur de qualite
`Q = omega L_s / R_s = R_p / (omega L_p)` (16.24.2), inverse de son
facteur de dissipation.

Une resistance reelle porte une inductance et une capacite parasites :
c'est le circuit antiresonnant (16.25.1), soit a basse frequence
`R + j omega (L - C R^2)` (16.25.2). Faible, elle est inductive ; elevee,
capacitive.
"""

from __future__ import annotations

import sympy as sp

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def facteur_de_dissipation(
    pulsation: Grandeur, capacite: Grandeur, serie: Grandeur
) -> Grandeur:
    """`d = omega C_s R_s` (16.23.3)."""
    return pulsation * capacite * serie


def _exiger_un_facteur(facteur: Grandeur) -> None:
    if (sp.sympify(facteur) <= 0) == sp.true:
        msg = f"facteur nul ou negatif : {facteur}"
        raise ValueError(msg)


def condensateur_en_parallele(
    serie: Grandeur, capacite: Grandeur, pertes: Grandeur
) -> tuple[Grandeur, Grandeur]:
    """`(R_p, C_p) = ((1 + d^2) R_s / d^2, C_s / (1 + d^2))` (16.23.4)."""
    _exiger_un_facteur(pertes)
    carre = pertes**2
    return (1 + carre) / carre * serie, capacite / (1 + carre)


def bobine_en_parallele(
    serie: Grandeur, inductance: Grandeur, qualite: Grandeur
) -> tuple[Grandeur, Grandeur]:
    """`(R_p, L_p) = ((Q^2 + 1) R_s, (Q^2 + 1) L_s / Q^2)` (16.24.1)."""
    _exiger_un_facteur(qualite)
    carre = qualite**2
    return (carre + 1) * serie, (carre + 1) / carre * inductance


def impedance_d_une_resistance_reelle(
    resistance: Grandeur, inductance: Grandeur, capacite: Grandeur, pulsation: Grandeur
) -> sp.Expr:
    """L'impedance complexe (16.25.1) : `C` en parallele avec `R` et `L` en serie."""
    accord = 1 - pulsation**2 * inductance * capacite
    residuelle = inductance * accord - capacite * resistance**2
    denominateur = pulsation**2 * capacite**2 * resistance**2 + accord**2
    return (resistance + sp.I * pulsation * residuelle) / denominateur


def resistance_en_serie_equivalente(
    parallele: Grandeur, capacite: Grandeur, pulsation: Grandeur
) -> tuple[Grandeur, Grandeur]:
    """`(R_s, tau)` de (16.25.8), ou `tau' = R_p C_p` et `tau = -tau'`."""
    constante = parallele * capacite
    return parallele / (1 + pulsation**2 * constante**2), -constante
