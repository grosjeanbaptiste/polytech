"""La conservation de la masse, locale puis integrale.

Les planches donnent les deux formes, et c'est leur articulation qui
compte : la forme locale `d rho/dt + div(rho u) = 0` se reduit a
`div u = 0` quand le fluide est incompressible, et la forme integrale
`q = S V` constante le long d'un tube de courant en decoule.

La garde sur la section nulle n'est pas formelle : une section qui
s'annule donnerait une vitesse infinie, que le document composerait comme
un resultat.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T, X, Y

MASSE_VOLUMIQUE = sp.Function("rho")


def forme_locale(vitesse: tuple[sp.Expr, sp.Expr]) -> sp.Eq:
    """`d rho/dt + div(rho u) = 0`, la forme generale des planches."""
    rho = MASSE_VOLUMIQUE(X, Y, T)
    horizontale, verticale = vitesse
    divergence = sp.diff(rho * horizontale, X) + sp.diff(
        rho * verticale,
        Y,
    )
    return sp.Eq(sp.diff(rho, T) + divergence, 0)


def divergence(vitesse: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    """`div u`, l'expression elle-meme et non une egalite.

    La rendre nue est deliberer : `sp.Eq(0, 0)` ne vaut pas une `Eq` mais
    un booleen, et un appelant qui lirait son `lhs` tomberait sur un
    `AttributeError` exactement dans le cas interessant -- celui ou le
    champ **est** incompressible.
    """
    horizontale, verticale = vitesse
    return sp.simplify(sp.diff(horizontale, X) + sp.diff(verticale, Y))


def forme_incompressible(
    vitesse: tuple[sp.Expr, sp.Expr],
) -> sp.Eq | sp.logic.boolalg.BooleanTrue:
    """`div u = 0` : ce que devient la forme locale a rho constant."""
    return sp.Eq(divergence(vitesse), 0)


def est_incompressible(vitesse: tuple[sp.Expr, sp.Expr]) -> bool:
    """Vrai si le champ satisfait `div u = 0`."""
    return bool(divergence(vitesse) == 0)


def vitesse_dans_la_section(debit: float, section: float) -> float:
    """`V = q / S` ; leve si la section est nulle."""
    if section <= 0:
        msg = f"section nulle ou negative : {section} m^2"
        raise ValueError(msg)
    return debit / section


def vitesse_apres_resserrement(
    vitesse_amont: float,
    section_amont: float,
    section_aval: float,
) -> float:
    """`S_1 V_1 = S_2 V_2` : la forme integrale le long d'un tube de courant."""
    debit = vitesse_amont * section_amont
    return vitesse_dans_la_section(debit, section_aval)
