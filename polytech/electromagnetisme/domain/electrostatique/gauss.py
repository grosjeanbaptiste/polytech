"""La route du theoreme de Gauss -- exercice 7 du chapitre II.

L'exercice 7 reprend la charge devant un plan conducteur, mais par un
autre chemin : il etablit le champ d'une charge ponctuelle par le
**theoreme de Gauss**, puis projette les deux contributions par la
**trigonometrie** plutot qu'en soustrayant des vecteurs unitaires.

Il aboutit au meme resultat que l'exercice 5, aux lettres pres --
`(a, y, z)` y devient `(h, x, 0)`. Les deux exercices sont donc **le meme
probleme**, pose deux fois. Le remarquer donne une verification gratuite
de l'un par l'autre, et c'est l'apport principal de ce corrige : deux
routes partant de principes differents ne peuvent pas s'etre trompees de
la meme facon.

**Pourquoi Gauss donne l'exposant deux.** Hors des charges, l'equation de
Maxwell-Gauss s'ecrit `div E = 0`, et un champ radial en `1/r^n` ne la
satisfait que pour `n = 2`. Ce n'est donc pas une convention : c'est la
dimension de l'espace qui fixe l'exposant. `divergence_du_champ_radial`
le rend verifiable au lieu d'affirme.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.images import PERMITTIVITE
from polytech.electromagnetisme.domain.symboles import CARTESIENNES


def champ_d_une_charge(charge: sp.Expr, rayon: sp.Expr) -> sp.Expr:
    """`E = Q / (4 pi eps_0 r^2)`, le module que Gauss impose.

    La symetrie spherique est ce qui permet de le sortir de l'integrale :
    le champ est uniforme en module sur la sphere et normal a elle, donc
    le flux est le simple produit du module par l'aire `4 pi r^2`.
    """
    return charge / (4 * sp.pi * PERMITTIVITE * rayon**2)


def divergence_du_champ_radial(exposant: int) -> sp.Expr:
    """La divergence cartesienne d'un champ radial en `1/r^exposant`.

    Nulle pour l'exposant deux et pour lui seul : c'est la raison
    profonde de la loi de Coulomb, et elle se verifie ici plutot que de
    se citer. Calculee en cartesiennes a dessein -- en spheriques, la
    formule de la divergence contient deja le `r^2` et le resultat
    paraitrait circulaire.
    """
    abscisse, ordonnee, cote = CARTESIENNES
    rayon = sp.sqrt(abscisse**2 + ordonnee**2 + cote**2)
    composantes = (abscisse, ordonnee, cote)
    champ = [axe / rayon ** (exposant + 1) for axe in composantes]
    return sp.simplify(
        sum(
            sp.diff(composante, axe)
            for composante, axe in zip(champ, composantes, strict=True)
        ),
    )


def distance_au_point(hauteur: sp.Expr, abscisse: sp.Expr) -> sp.Expr:
    """`d = sqrt(h^2 + x^2)` : la distance commune aux deux charges.

    Commune, et seulement sur le plan -- comme dans l'exercice 5. C'est
    cette egalite qui annule les composantes tangentielles.
    """
    return sp.sqrt(hauteur**2 + abscisse**2)


def champ_par_projection(
    charge: sp.Expr,
    hauteur: sp.Expr,
    abscisse: sp.Expr,
) -> sp.Expr:
    """Le champ sur le plan, par la route trigonometrique du manuscrit.

    Les deux contributions ont le meme module `Q / (4 pi eps_0 d^2)`. Les
    composantes tangentielles portent `cos(alpha) = x/d` avec des signes
    opposes et se detruisent ; les normales portent `sin(alpha) = h/d`
    avec le meme signe et s'ajoutent -- d'ou le facteur deux.

    Le signe est negatif : le champ pointe de la charge vers le plan.
    """
    distance = distance_au_point(hauteur, abscisse)
    sinus = hauteur / distance
    return -2 * champ_d_une_charge(charge, distance) * sinus
