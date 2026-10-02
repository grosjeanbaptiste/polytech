"""Une sphere dont la charge varie en racine du rayon -- exercice 8.

`eta = A sqrt(r)` : la densite n'est pas uniforme, et c'est tout
l'interet. Le theoreme de Gauss reste applicable -- la symetrie spherique
suffit, l'uniformite n'est pas requise -- mais la charge contenue n'est
plus proportionnelle au volume : elle croit comme `r^{7/2}`, l'exposant
trois du volume augmente du demi de la densite.

Les deux routes du corrige manuscrit sont menees ici par deux machineries
**genuinement differentes** : `champ_par_poisson` resout une equation
differentielle, `champ_par_gauss` integre la densite sur un volume en
coordonnees spheriques. Qu'elles s'accordent est donc une verification et
non une tautologie.

**Le point ou le manuscrit raisonne a l'envers.** Il ecrit « en `r = 0`,
`E_r = 0`, donc `C = 0` » -- mais on ne peut pas evaluer `r^2 E_r = C` en
zero pour en deduire que `C` est nul : il faudrait deja savoir que `E_r`
y est borne. L'argument correct est l'inverse. Le terme ecarte est
`C/r^2`, il **diverge** a l'origine, et rien ne peut y produire un champ
infini faute de charge ponctuelle au centre. `terme_divergent_ecarte` le
rend explicite -- et il satisfait bel et bien l'equation sans source, ce
qui montre que son rejet demande un argument physique.
"""

from __future__ import annotations

import sympy as sp

PERMITTIVITE = sp.Symbol("varepsilon_0", positive=True)
"""La permittivite du vide, gardee symbolique : l'exercice est litteral."""


def charge_volumique(constante: sp.Expr, rayon: sp.Expr) -> sp.Expr:
    """`eta = A sqrt(r)`, la loi de l'enonce."""
    return constante * sp.sqrt(rayon)


def divergence_spherique(champ_radial: sp.Expr, rayon: sp.Expr) -> sp.Expr:
    """`(1/r^2) d/dr (r^2 E_r)` : la divergence d'un champ purement radial.

    Les deux autres termes de la divergence spherique disparaissent parce
    que la symetrie annule `E_theta` et `E_phi` -- c'est la seule chose
    que la symetrie apporte, et elle suffit.
    """
    return sp.diff(rayon**2 * champ_radial, rayon) / rayon**2


def terme_divergent_ecarte(constante: sp.Expr, rayon: sp.Expr) -> sp.Expr:
    """`C / r^2` : le terme que la solution generale porte et qu'on rejette.

    Rendu pour qu'on puisse montrer **pourquoi** on l'ecarte, au lieu de
    l'affirmer : il diverge en zero, alors qu'aucune charge ponctuelle n'y
    siege. Il satisfait pourtant l'equation sans source, et c'est ce qui
    rend le rejet un argument physique et non algebrique.

    La constante est un argument et non un symbole du module : son nom
    appartient a `dsolve`, qui le choisit, et le fixer ici serait parier
    sur une version de sympy.
    """
    return constante / rayon**2


def _constante_de(solution: sp.Expr, connus: frozenset[sp.Symbol]) -> sp.Symbol:
    """La constante d'integration que `dsolve` a introduite.

    Reperee par difference plutot que par son nom : `dsolve` l'appelle
    `C1`, mais c'est une convention interne qu'une version pourrait
    changer -- et une substitution sur un nom errone ne porterait sur
    rien **sans echouer**. C'est exactement ce qui est arrive au premier
    jet de ce module : le champ sortait avec son terme divergent intact,
    et cinq tests sont tombes pour le dire.

    Le refus est explicite si la difference n'en donne pas exactement
    une : rendre un symbole arbitraire laisserait passer le defaut.
    """
    restantes = sorted(solution.free_symbols - connus, key=str)
    if len(restantes) != 1:
        msg = f"une seule constante attendue, trouve {restantes}"
        raise ValueError(msg)
    return restantes[0]


def champ_par_poisson(constante: sp.Expr, rayon: sp.Expr) -> sp.Expr:
    """Le champ interieur, en resolvant l'equation de Poisson.

    `dsolve` rend la solution generale, qui porte le terme divergent ; on
    annule la constante parce que le champ doit rester borne a l'origine.
    """
    inconnue = sp.Function("E")
    equation = sp.Eq(
        divergence_spherique(inconnue(rayon), rayon),
        charge_volumique(constante, rayon) / PERMITTIVITE,
    )
    generale = sp.dsolve(equation, inconnue(rayon)).rhs
    connus = frozenset({rayon, PERMITTIVITE}) | constante.free_symbols
    return sp.simplify(generale.subs(_constante_de(generale, connus), 0))


def charge_contenue(constante: sp.Expr, borne: sp.Expr) -> sp.Expr:
    """`Q(borne)`, par l'integrale triple de la densite en spheriques.

    L'element de volume est `rho^2 sin(theta) drho dtheta dphi`, et ce
    sont ses deux facteurs angulaires qui donnent le `4 pi` -- non une
    formule memorisee de l'aire d'une sphere.
    """
    distance = sp.Symbol("rho", positive=True)
    colatitude = sp.Symbol("theta", positive=True)
    azimut = sp.Symbol("varphi", positive=True)
    element = charge_volumique(constante, distance) * distance**2 * sp.sin(colatitude)
    return sp.integrate(
        element,
        (azimut, 0, 2 * sp.pi),
        (colatitude, 0, sp.pi),
        (distance, 0, borne),
    )


def champ_par_gauss(constante: sp.Expr, rayon: sp.Expr) -> sp.Expr:
    """Le champ interieur par Gauss : la charge contenue sur `4 pi eps_0 r^2`."""
    contenue = charge_contenue(constante, rayon)
    return sp.simplify(contenue / (4 * sp.pi * PERMITTIVITE * rayon**2))


def champ_exterieur(
    constante: sp.Expr,
    rayon: sp.Expr,
    rayon_de_la_sphere: sp.Expr,
) -> sp.Expr:
    """Le champ au-dela de la sphere : celui d'une charge ponctuelle.

    L'integrale s'arrete a `R` -- au-dela il n'y a plus de charge -- de
    sorte que le champ est exactement celui de la charge **totale**
    placee au centre. C'est le fait le plus utile de l'exercice, et le
    manuscrit l'obtient sans le nommer.
    """
    totale = charge_contenue(constante, rayon_de_la_sphere)
    return sp.simplify(totale / (4 * sp.pi * PERMITTIVITE * rayon**2))
