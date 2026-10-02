"""La methode des images, menee symboliquement -- exercice 5 de statique.

La methode remplace un probleme a conditions aux limites par un probleme
a deux charges libres, et elle est **exacte** -- pas approchee. L'argument
est l'unicite : si une distribution fictive produit le bon potentiel sur
la frontiere, elle produit le bon champ partout dans le demi-espace,
parce que la solution de Laplace a conditions fixees est unique.

Trois resultats, et le dernier est celui que le corrige officiel ne fait
pas.

**Le champ sur le plan est normal.** Les contributions tangentielles des
deux charges sont egales et opposees, les normales s'ajoutent. Un
conducteur impose precisement cela, et le retrouver verifie la methode
plutot que de la croire.

**La densite induite** suit par la relation de passage, dont l'un des
deux termes est nul : a l'interieur d'un conducteur en equilibre il n'y a
ni champ ni deplacement.

**La charge induite totale vaut exactement `-q`.** C'est la verification
la plus forte possible : elle confirme d'un coup le coefficient
`1/(2 pi)` et l'exposant trois demis, les deux endroits ou une erreur se
logerait. Une faute sur l'un donnerait un multiple de `-q`, sur l'autre
une integrale divergente.

Le champ decroit en `1/r^3` au loin, et non en `1/r^2` : c'est la
signature d'un **dipole**, que la charge et son image forment.
"""

from __future__ import annotations

import sympy as sp

PERMITTIVITE = sp.Symbol("varepsilon_0", positive=True)
"""`epsilon_0`, gardee symbolique : l'exercice est litteral de bout en bout."""


def _distance_au_carre(
    distance: sp.Expr,
    ordonnee: sp.Expr,
    cote: sp.Expr,
) -> sp.Expr:
    """`a^2 + y^2 + z^2` : la meme pour la charge et pour son image.

    C'est l'egalite des deux distances qui fait annuler les composantes
    tangentielles -- et elle n'a lieu que sur le plan.
    """
    return distance**2 + ordonnee**2 + cote**2


def champ_sur_le_plan(
    charge: sp.Expr,
    distance: sp.Expr,
    ordonnee: sp.Expr,
    cote: sp.Expr,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Le champ au point `(0, y, z)` du plan conducteur.

    Rendu en trois composantes pour que les deux nulles soient
    **verifiables** plutot que supposees : c'est leur annulation qui
    etablit que le champ est normal au conducteur.
    """
    rayon_carre = _distance_au_carre(distance, ordonnee, cote)
    facteur = charge / (4 * sp.pi * PERMITTIVITE * rayon_carre ** sp.Rational(3, 2))
    # La charge en (a,0,0) tire vers (-a, y, z) ; son image en (-a,0,0),
    # de signe oppose, tire vers (a, y, z). Les y et les z se detruisent.
    return (
        facteur * (-distance) - facteur * distance,
        facteur * ordonnee - facteur * ordonnee,
        facteur * cote - facteur * cote,
    )


def densite_induite(
    charge: sp.Expr,
    distance: sp.Expr,
    ordonnee: sp.Expr,
    cote: sp.Expr,
) -> sp.Expr:
    """`sigma = epsilon_0 E_n`, la relation de passage sur le conducteur.

    Elle a deux termes et l'un est nul : a l'interieur d'un conducteur en
    equilibre il n'y a ni champ ni deplacement.
    """
    return PERMITTIVITE * champ_sur_le_plan(charge, distance, ordonnee, cote)[0]


def charge_induite_totale(charge: sp.Expr, distance: sp.Expr) -> sp.Expr:
    """L'integrale de la densite sur le plan entier : exactement `-q`.

    Menee en polaires, `dS = r dr dtheta`, de zero a l'infini. Le
    resultat etant exact et independant de `a`, il confirme a la fois le
    coefficient et l'exposant.
    """
    rayon = sp.Symbol("r", positive=True)
    densite = densite_induite(charge, distance, rayon, sp.Integer(0))
    return sp.integrate(densite * 2 * sp.pi * rayon, (rayon, 0, sp.oo))


def force_sur_la_charge(charge: sp.Expr, distance: sp.Expr) -> sp.Expr:
    """`F = q^2 / (4 pi eps_0 (2a)^2)` : la loi de Coulomb a distance `2a`.

    L'autre consequence de la methode, et elle surprend : la force sur la
    charge reelle est **exactement** celle qu'exercerait une charge
    opposee placee a `2a`. La distance qui compte est donc celle de la
    charge a son image, non celle au plan.

    Ecrite ici avec `(2 * distance)` pour que la lecture du code montre
    cette distance -- mais sympy reduit aussitot a
    `q^2 / (16 pi eps_0 a^2)`, et c'est cette forme reduite qui sera
    publiee. Le document doit donc rappeler le `2a` en prose, le
    resultat ne le portant plus.
    """
    return charge**2 / (4 * sp.pi * PERMITTIVITE * (2 * distance) ** 2)


def energie_d_interaction(charge: sp.Expr, distance: sp.Expr) -> sp.Expr:
    """`W = -q^2 / (2 . 4 pi eps_0 . 2a)` : la **moitie** de l'energie de la paire.

    Le piege classique de la methode. La force sur la charge est
    exactement celle de son image, mais l'energie ne vaut que la moitie
    de celle d'une vraie paire -- parce que l'image n'est pas une charge
    reelle et qu'on ne peut pas compter une interaction mutuelle avec un
    objet fictif. Le calcul par le travail d'amenee donne le facteur un
    demi.

    Le `2 *` est ecrit a part dans l'expression pour que ce facteur se
    lise dans le code et non seulement dans ce commentaire.

    Et c'est lui qui rend la verification croisee possible : `-dW/da`
    redonne la force de Coulomb entiere. Avec l'energie de la paire, la
    derivee en donnerait le double, donc une contradiction -- ce qui est
    la facon la plus sure de detecter la faute.
    """
    return -(charge**2) / (2 * 4 * sp.pi * PERMITTIVITE * 2 * distance)
