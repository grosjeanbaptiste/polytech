"""Les permeabilites conventionnelles d'un materiau ferromagnetique.

Le syllabus d'electricite generale (17.9), apres les permeabilites
initiale, normale, de recul et differentielle de la section 10.5. La
permeabilite d'amplitude est le quotient des valeurs de crete de
l'induction et du champ sur un cycle symetrique. La permeabilite avec
champ statique superpose rapporte les excursions de crete a creux,
`Delta B / Delta H` (17.9.1), d'un petit cycle decrit autour d'un champ
continu `H_0` ; sa limite, quand la composante alternative s'eteint, est
la permeabilite reversible.

La courbe est ici une expression `B(H)` a une seule valeur : sans
hysteresis, la permeabilite reversible se confond avec la pente de la
courbe.
"""

from __future__ import annotations

import sympy as sp

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def permeabilite_d_amplitude(
    crete_de_l_induction: float, crete_du_champ: float
) -> float:
    """`mu_a = B_crete / H_crete`, en henrys par metre."""
    if crete_du_champ == 0:
        msg = "champ de crete nul : aucun cycle n'est decrit"
        raise ValueError(msg)
    return crete_de_l_induction / crete_du_champ


def permeabilite_incrementale(
    induction: sp.Expr, champ: sp.Symbol, *, statique: Grandeur, excursion: Grandeur
) -> sp.Expr:
    """`Delta B / Delta H` (17.9.1) entre `H_0 - excursion` et `H_0 + excursion`."""
    if (sp.sympify(excursion) <= 0) == sp.true:
        msg = f"excursion nulle ou negative : {excursion} A/m"
        raise ValueError(msg)
    haut = induction.subs(champ, statique + excursion)
    bas = induction.subs(champ, statique - excursion)
    return sp.simplify((haut - bas) / (2 * excursion))


def permeabilite_reversible(
    induction: sp.Expr, champ: sp.Symbol, *, statique: Grandeur
) -> sp.Expr:
    """La limite de la precedente quand l'excursion tend vers zero."""
    petite = sp.Symbol("epsilon", positive=True)
    rapport = permeabilite_incrementale(
        induction, champ, statique=statique, excursion=petite
    )
    return sp.limit(rapport, petite, 0)
