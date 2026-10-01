"""Stabilite et reponse indicielle, lues sur la fonction de transfert.

Le corrige decompose en fractions simples a la main. La decomposition n'est
pas le resultat : le resultat est la fonction du temps. Ces deux fonctions
la produisent sans passer par les residus, ce qui donne au test un chemin
different de celui du corrige -- condition pour qu'il puisse le contredire.
"""

from __future__ import annotations

import sympy as sp


def _denominateur(transfert: sp.Expr, variable: sp.Expr) -> sp.Poly:
    """Le denominateur de la fonction de transfert, en polynome."""
    _, denominateur = sp.fraction(sp.together(transfert))
    polynome = sp.Poly(sp.expand(denominateur), variable)
    if polynome.degree() < 1:
        msg = f"denominateur sans pole : {denominateur}"
        raise ValueError(msg)
    return polynome


def est_stable(transfert: sp.Expr, variable: sp.Expr) -> bool:
    """Vrai si tous les poles sont a partie reelle strictement negative."""
    racines = sp.roots(_denominateur(transfert, variable))
    if not racines:
        msg = "poles non calculables : la stabilite ne se decide pas"
        raise ValueError(msg)
    return all(bool(sp.re(racine) < 0) for racine in racines)


def reponse_a_un_echelon(
    transfert: sp.Expr,
    amplitude: sp.Expr,
    variable: sp.Expr,
    temps: sp.Expr,
) -> sp.Expr:
    """La sortie temporelle pour une entree `amplitude * u(t)`.

    L'echelon vaut `amplitude / s` en Laplace ; l'echelon d'Heaviside est
    ensuite ramene a 1, l'expression ne valant que pour `t > 0`.
    """
    image = sp.simplify(transfert * amplitude / variable)
    reponse = sp.inverse_laplace_transform(image, variable, temps)
    return sp.simplify(reponse.subs(sp.Heaviside(temps), 1))


def impedance_de_resistance(resistance: sp.Expr, variable: sp.Expr) -> sp.Expr:
    """`Z = R` : la seule impedance qui ne depende pas de la frequence."""
    del variable
    return sp.sympify(resistance)


def impedance_de_bobine(inductance: sp.Expr, variable: sp.Expr) -> sp.Expr:
    """`Z = p L`.

    C'est l'equivalence que tout le chapitre exploite : `u = L di/dt`
    devient `U = p L I`, donc la derivation devient une multiplication et
    l'equation integro-differentielle devient algebrique.
    """
    return variable * inductance


def impedance_de_condensateur(capacite: sp.Expr, variable: sp.Expr) -> sp.Expr:
    """`Z = 1 / (p C)` ; leve pour une capacite nulle."""
    if sp.simplify(capacite) == 0:
        msg = "capacite nulle : son impedance serait infinie a toute frequence"
        raise ValueError(msg)
    return 1 / (variable * capacite)


def impedances_en_serie(*impedances: sp.Expr) -> sp.Expr:
    """Les impedances traversees par le meme courant s'additionnent."""
    if not impedances:
        msg = "aucune impedance a mettre en serie"
        raise ValueError(msg)
    return sp.simplify(sum(impedances))


def impedances_en_parallele(*impedances: sp.Expr) -> sp.Expr:
    """Ce sont les **admittances** qui s'additionnent : `1/Z = somme 1/Z_i`."""
    if not impedances:
        msg = "aucune impedance a mettre en parallele"
        raise ValueError(msg)
    return sp.simplify(1 / sum(1 / impedance for impedance in impedances))


def diviseur_de_tension(amont: sp.Expr, aval: sp.Expr) -> sp.Expr:
    """`Z_aval / (Z_amont + Z_aval)` : le diviseur generalise aux impedances."""
    total = sp.simplify(amont + aval)
    if total == 0:
        msg = f"impedance nulle en somme : {amont} et {aval}"
        raise ValueError(msg)
    return sp.simplify(aval / total)


def diviseur_de_courant(branche: sp.Expr, autre: sp.Expr) -> sp.Expr:
    """Le dual : le courant se partage a l'**inverse** des impedances.

    `I_branche / I_total = Z_autre / (Z_branche + Z_autre)`. L'inversion
    est le piege du chapitre -- la branche la moins resistive prend le
    plus de courant, l'inverse du diviseur de tension.
    """
    total = sp.simplify(branche + autre)
    if total == 0:
        msg = f"impedance nulle en somme : {branche} et {autre}"
        raise ValueError(msg)
    return sp.simplify(autre / total)


def multiplicites_des_poles(
    transfert: sp.Expr,
    variable: sp.Expr,
) -> dict[sp.Expr, int]:
    """Chaque pole et sa multiplicite.

    Le chapitre distingue trois cas de decomposition selon ce nombre --
    poles simples, poles reels multiples, poles conjugues -- et les
    traiter pareil donne une decomposition fausse.
    """
    racines = sp.roots(_denominateur(transfert, variable))
    if not racines:
        msg = "poles non calculables : la decomposition ne se fait pas"
        raise ValueError(msg)
    return {pole: int(ordre) for pole, ordre in racines.items()}


def decomposition_en_fractions_simples(
    transfert: sp.Expr,
    variable: sp.Expr,
) -> tuple[sp.Expr, ...]:
    """Les fractions simples, une par une, avant toute somme.

    Rendues separees parce que c'est ainsi que le corrige les montre : la
    decomposition est l'etape, la somme n'est que son controle.
    """
    decomposee = sp.apart(sp.together(transfert), variable, full=True).doit()
    return tuple(decomposee.as_ordered_terms())


def residu_en_un_pole(
    transfert: sp.Expr,
    variable: sp.Expr,
    pole: sp.Expr,
) -> sp.Expr:
    """`lim (p - p_i) F(p)` quand `p` tend vers `p_i` : la recette du chapitre.

    Elle ne vaut que pour un pole **simple**. Sur un pole multiple la
    limite diverge, et c'est pourquoi le chapitre donne une seconde
    recette -- la derivee -- que `decomposition_en_fractions_simples`
    applique.
    """
    return sp.simplify(sp.limit((variable - pole) * transfert, variable, pole))


def reponse_impulsionnelle(
    transfert: sp.Expr,
    variable: sp.Expr,
    temps: sp.Expr,
) -> sp.Expr:
    """L'original de la fonction de transfert : la reponse a une impulsion.

    L'echelon d'Heaviside est ramene a un, l'expression ne valant que pour
    `t > 0` -- la transformee de Laplace est unilaterale.
    """
    reponse = sp.inverse_laplace_transform(transfert, variable, temps)
    return sp.simplify(reponse.subs(sp.Heaviside(temps), 1))
