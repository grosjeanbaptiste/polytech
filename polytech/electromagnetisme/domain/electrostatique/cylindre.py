"""Le champ exterieur d'un cylindre charge -- exercice « Statique 1.1 ».

Deux routes, menees par deux machineries differentes. `champ_par_poisson`
resout l'equation en cylindriques, ce qui laisse une constante, et la fixe
par la **relation de passage** a la surface du conducteur.
`champ_par_gauss` calcule un flux a travers un cylindre de rayon `rho` et
ne touche jamais a cette surface. Qu'elles s'accordent est une
verification.

**Pourquoi la decroissance est en `1/rho` et non en `1/rho^2`.** La
divergence cylindrique d'un champ radial, `(1/rho) d/drho (rho E)`, ne
s'annule que pour `1/rho`. L'exposant est donc un de moins que pour une
charge ponctuelle : la symetrie cylindrique a une dimension de moins ou
s'etaler, et le flux se repartit sur une aire qui croit comme `rho` et
non comme `rho^2`.

**Le potentiel ne peut pas etre pris nul a l'infini.** Il est
logarithmique, donc il diverge au loin. La convention habituelle
`V(oo) = 0` est impossible ici, et il faut une distance de reference
finie. La raison physique : un fil infini porte une charge infinie, et
amener une charge d'essai depuis l'infini demanderait un travail infini.

**Le resultat suppose le cylindre infini.** `Q/L` n'est une densite
lineique uniforme que loin des extremites, donc pour `rho << L`. Pres des
bouts, le champ cesse d'etre radial et aucune des deux routes ne vaut.
"""

from __future__ import annotations

import sympy as sp

PERMITTIVITE = sp.Symbol("varepsilon_0", positive=True)
"""La permittivite du vide, gardee symbolique : l'exercice est litteral."""


def densite_lineique(charge: sp.Expr, longueur: sp.Expr) -> sp.Expr:
    """`lambda = Q/L` : la grandeur qui gouverne tout l'exercice.

    Elle n'a de sens que pour un cylindre long devant son rayon : c'est
    l'hypothese tacite de l'enonce.
    """
    return charge / longueur


def densite_surfacique(
    charge: sp.Expr,
    longueur: sp.Expr,
    rayon: sp.Expr,
) -> sp.Expr:
    """`sigma = Q / (2 pi R L)` : la charge divisee par l'**aire** laterale.

    Le manuscrit l'ecrit `(Q/L) x 1/(2 pi R)` en annotant le second
    facteur « surface du cylindre de longueur L ». Le produit est juste,
    l'annotation non : `1/(2 pi R)` est l'inverse d'une longueur. L'aire
    laterale est `2 pi R L`.
    """
    return charge / (2 * sp.pi * rayon * longueur)


def divergence_cylindrique(champ_radial: sp.Expr, distance: sp.Expr) -> sp.Expr:
    """`(1/rho) d/drho (rho E_rho)` : la divergence d'un champ radial.

    Les deux autres termes disparaissent parce que la symetrie annule
    `E_theta` et `E_z`. Le facteur `1/rho` vient de l'element de volume
    cylindrique, et c'est lui qui fixe l'exposant du champ.
    """
    return sp.diff(distance * champ_radial, distance) / distance


def champ_par_gauss(
    charge: sp.Expr,
    longueur: sp.Expr,
    distance: sp.Expr,
) -> sp.Expr:
    """Le flux a travers un cylindre de rayon `rho` donne la charge enlacee.

    L'aire laterale est `2 pi rho L` ; les deux disques des bouts ne
    comptent pas, le champ y etant parallele a leur surface. C'est la
    seule chose que la symetrie apporte ici, et elle suffit.
    """
    aire_laterale = 2 * sp.pi * distance * longueur
    return charge / (PERMITTIVITE * aire_laterale)


def champ_par_poisson(
    charge: sp.Expr,
    longueur: sp.Expr,
    rayon: sp.Expr,
    distance: sp.Expr,
) -> sp.Expr:
    """`E = K/rho`, la constante fixee par la relation de passage.

    L'equation sans source donne `rho E = K` : la forme, mais pas la
    valeur. C'est la relation de passage `sigma = eps_0 E(R)` qui ferme
    le probleme -- et elle se verifie ensuite a l'envers sur le resultat.

    Le rayon est un argument explicite bien qu'il **se simplifie** :
    `sigma` porte un `1/R` que le `R` de `K = R sigma / eps_0` detruit.
    Le garder dans la signature rend cet effacement verifiable, alors
    qu'un symbole interne le cacherait -- et l'effacement est un fait
    physique : deux cylindres de rayons differents portant la meme
    charge lineique creent le meme champ exterieur.
    """
    a_la_surface = densite_surfacique(charge, longueur, rayon) / PERMITTIVITE
    constante = rayon * a_la_surface
    return sp.simplify(constante / distance)


def potentiel(
    charge: sp.Expr,
    longueur: sp.Expr,
    distance: sp.Expr,
    reference: sp.Expr,
) -> sp.Expr:
    """`V(rho) = -lambda/(2 pi eps_0) ln(rho/rho_0)`, nul en `rho_0`.

    La distance de reference est un **argument obligatoire** et non une
    valeur par defaut : pour un fil infini, il n'y a pas de choix
    naturel. Prendre l'infini est impossible -- le logarithme diverge --
    et laisser une valeur par defaut ferait croire le contraire.
    """
    lineique = densite_lineique(charge, longueur)
    return -lineique / (2 * sp.pi * PERMITTIVITE) * sp.log(distance / reference)
