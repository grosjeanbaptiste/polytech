"""Une nappe de conducteurs en progression geometrique -- exercice 8.

Dix paires de conducteurs portant chacune `3 A`, placees de part et
d'autre de l'axe a des abscisses doublant a chaque fois. Le champ au
centre est la somme de vingt contributions.

**Les deux cotes s'ajoutent, et il faut dire pourquoi.** Les courants
sont opposes **et** les positions symetriques : les deux inversions de
signe se detruisent, et les contributions s'additionnent. Avec une seule
inversion elles s'annuleraient et le champ au centre serait nul --
`champ_d_un_seul_cote` rend le facteur deux verifiable plutot que
suppose.

**Pourquoi ce module distingue deux calculs du meme champ.** Le resultat
encadre par le manuscrit est juste, mais le chemin qui y mene porte deux
erreurs qui se compensent exactement :

* sa somme ecrit les distances `2^i x 10^-2` pour `i` de un a dix, soit
  `2` a `1024` cm, alors que la figure commence a `1` cm -- la somme des
  inverses y vaut donc la **moitie** de la vraie ;
* sa ligne de developpement ecrit `6/pi` la ou sa propre ligne de formule
  donne `3/pi` -- le `2` du `2 pi` a disparu, soit le **double**.

`champ_par_le_developpement_du_manuscrit` reproduit la seconde erreur,
pour que la compensation se mesure au lieu de s'affirmer. Un resultat
juste obtenu par un chemin faux est plus dangereux qu'un resultat faux :
rien ne le signale, et qui reprend une seule des deux lignes se trompe du
facteur deux.

**Tout est en rationnels exacts.** Les distances sont des puissances de
deux sur cent, et la somme geometrique a une forme fermee : aucune
etape n'a besoin de flottants, et l'egalite des deux calculs se verifie
donc exactement.
"""

from __future__ import annotations

import sympy as sp

CENTIMETRE = sp.Rational(1, 100)
"""Un centimetre en metres, exact -- les distances de l'enonce y sont."""

RAISON = 2
"""La raison de la progression : chaque conducteur est deux fois plus loin."""


def distances_geometriques(
    premier_exposant: int,
    nombre: int,
) -> tuple[sp.Expr, ...]:
    """Les `nombre` distances `2^k x 1 cm`, en metres et en exact.

    `premier_exposant` vaut zero pour la lecture de la figure -- le
    premier conducteur a un centimetre -- et un pour la lecture de la
    somme du manuscrit, qui commence a deux.
    """
    return tuple(
        sp.Integer(RAISON) ** exposant * CENTIMETRE
        for exposant in range(premier_exposant, premier_exposant + nombre)
    )


def somme_des_inverses(distances: tuple[sp.Expr, ...]) -> sp.Expr:
    """La somme des `1/d`, terme a terme -- le calcul explicite."""
    return sum((1 / distance for distance in distances), start=sp.Integer(0))


def somme_fermee(premier_exposant: int, nombre: int) -> sp.Expr:
    """`100 . 2^(1 - premier) . (1 - 2^-nombre)` : la serie geometrique.

    Forme fermee, independante du calcul explicite. Le manuscrit
    l'emploie dans sa note de verification sans la demontrer ; les deux
    doivent coincider exactement.
    """
    premier_terme = 1 / (sp.Integer(RAISON) ** premier_exposant * CENTIMETRE)
    return premier_terme * RAISON * (1 - sp.Rational(1, RAISON**nombre))


def champ_d_un_seul_cote(
    courant: sp.Expr,
    distances: tuple[sp.Expr, ...],
) -> sp.Expr:
    """`sum I / (2 pi d)` : la contribution des conducteurs d'un seul cote.

    Rendue a part pour que le facteur deux de la symetrie se verifie, au
    lieu d'etre pose d'emblee.
    """
    return courant / (2 * sp.pi) * somme_des_inverses(distances)


def champ_de_la_nappe(
    courant: sp.Expr,
    distances: tuple[sp.Expr, ...],
) -> sp.Expr:
    """`sum 2 I / (2 pi d) = (I/pi) sum 1/d` : les deux cotes.

    C'est la ligne de **formule** du manuscrit, qui est juste.
    """
    return 2 * champ_d_un_seul_cote(courant, distances)


def champ_par_le_developpement_du_manuscrit(
    courant: sp.Expr,
    distances: tuple[sp.Expr, ...],
) -> sp.Expr:
    """`(2 I / pi) sum 1/d` : la ligne de **developpement** du manuscrit.

    Le double de sa propre ligne de formule : le `2` du `2 pi` a disparu.
    Reproduit ici pour que la compensation avec l'erreur de distances se
    mesure -- non pour etre employe.
    """
    return 2 * champ_de_la_nappe(courant, distances)


def champ_de_la_nappe_infinie(
    courant: sp.Expr,
    premier_exposant: int,
) -> sp.Expr:
    """Le champ d'une nappe prolongee a l'infini.

    La serie converge, et dix paires en donnent deja plus de 99,9 pour
    cent : une onzieme paire changerait le resultat de moins d'un
    millieme. C'est pourquoi l'enonce peut s'arreter a dix, et le
    manuscrit ne le dit pas.
    """
    premier_terme = 1 / (sp.Integer(RAISON) ** premier_exposant * CENTIMETRE)
    return courant / sp.pi * premier_terme * RAISON
