"""L'exercice 6 de la seance 2.1 : deux conducteurs, champ et force.

Deux conducteurs paralleles a `Oz`, en `x = -a` et `x = +a`, parcourus
par des courants **opposes**. Le point d'observation `P(0, b)` est sur la
mediatrice, a egale distance des deux.

Le corrige manuscrit est juste : ses deux resultats encadres -- le champ
puis la force lineique sur un troisieme conducteur place en `P` -- sont
exacts. Quatre choses lui manquent, et ce fichier les etablit.

**La force est orthogonale au champ**, ce qui se verifie par un produit
scalaire nul -- controle independant des expressions elles-memes, et le
plus efficace contre une faute de produit vectoriel.

**Le champ en `P` est maximal quand `a = b`.** Pour une distance `b`
donnee, ecarter les conducteurs augmente d'abord le champ puis le
diminue, et l'optimum est atteint quand le demi-ecartement egale la
distance. Rien n'en parle, et c'est le seul resultat de l'exercice qui
serve a dimensionner quelque chose.

**Au loin, le champ decroit en `1/b^2`** et non en `1/b` comme celui d'un
fil seul. C'est la signature d'un **dipole magnetique** : les deux
courants opposes en forment un, exactement comme la charge et son image
formaient un dipole electrique a l'exercice 5 de statique.

**Le champ s'annule si les courants sont de meme sens**, sur la
mediatrice. C'est le revers : ce qui s'ajoute ici se detruirait la, et
l'inverse.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.mediatrice import (
    champ_si_les_courants_sont_de_meme_sens,
    champ_sur_la_mediatrice,
    force_lineique_en_un_point,
    produit_scalaire,
)

COURANT = sp.Symbol("I", positive=True)
COURANT_DU_TROISIEME = sp.Symbol("I_2", positive=True)
DEMI_ECART = sp.Symbol("a", positive=True)
ORDONNEE = sp.Symbol("b", positive=True)
PERMEABILITE = sp.Symbol("mu_0", positive=True)


def _encadre_du_champ() -> sp.Expr:
    """`H_y = -a I / (pi (a^2 + b^2))`, le resultat encadre du manuscrit."""
    return -DEMI_ECART * COURANT / (sp.pi * (DEMI_ECART**2 + ORDONNEE**2))


def test_la_composante_selon_x_du_champ_est_nulle() -> None:
    """Les deux contributions s'y detruisent, et c'est ce qui fait l'exercice.

    Le point etant sur la mediatrice, les deux conducteurs sont a egale
    distance ; leurs contributions selon `Ox` sont egales et opposees.
    Le retrouver verifie la geometrie plutot que de la croire.
    """
    champ = champ_sur_la_mediatrice(COURANT, DEMI_ECART, ORDONNEE)
    assert sp.simplify(champ[0]) == 0


def test_le_champ_est_celui_qu_encadre_le_manuscrit() -> None:
    """L'oracle est le resultat encadre, ecrit a la main d'apres le scan."""
    champ = champ_sur_la_mediatrice(COURANT, DEMI_ECART, ORDONNEE)
    assert sp.simplify(champ[1] - _encadre_du_champ()) == 0


def test_la_force_lineique_est_celle_qu_encadre_le_manuscrit() -> None:
    """`F/L = -mu_0 I I_2 a / (pi (a^2+b^2))` selon `u_x`, encadre aussi."""
    force = force_lineique_en_un_point(
        COURANT,
        DEMI_ECART,
        ORDONNEE,
        COURANT_DU_TROISIEME,
        PERMEABILITE,
    )
    attendue = (
        -PERMEABILITE
        * COURANT
        * COURANT_DU_TROISIEME
        * DEMI_ECART
        / (sp.pi * (DEMI_ECART**2 + ORDONNEE**2))
    )
    assert sp.simplify(force[0] - attendue) == 0
    assert sp.simplify(force[1]) == 0


def test_la_force_est_orthogonale_au_champ() -> None:
    """Produit scalaire nul : le controle le plus efficace sur le signe.

    Toute force de Laplace sur un conducteur porte par `u_z` est
    orthogonale au champ, parce que le produit vectoriel l'impose. Le
    verifier est independant des deux expressions, et c'est la que les
    fautes de produit vectoriel se logent.
    """
    champ = champ_sur_la_mediatrice(COURANT, DEMI_ECART, ORDONNEE)
    force = force_lineique_en_un_point(
        COURANT,
        DEMI_ECART,
        ORDONNEE,
        COURANT_DU_TROISIEME,
        PERMEABILITE,
    )
    assert sp.simplify(produit_scalaire(force, champ)) == 0


def test_le_champ_est_maximal_quand_l_ecartement_egale_la_distance() -> None:
    """`a = b` : le seul resultat de l'exercice qui serve a dimensionner.

    Pour une distance `b` donnee, ecarter les conducteurs augmente
    d'abord le champ -- on gagne en bras de levier -- puis le diminue --
    on perd en proximite. L'optimum est exactement `a = b`.

    Verifie en annulant la derivee par rapport a `a`, et en gardant la
    seule racine positive.
    """
    champ_y = champ_sur_la_mediatrice(COURANT, DEMI_ECART, ORDONNEE)[1]
    racines = sp.solve(sp.diff(champ_y, DEMI_ECART), DEMI_ECART)
    assert [r for r in racines if r.is_positive] == [ORDONNEE]


def test_le_champ_decroit_comme_le_carre_de_la_distance_au_loin() -> None:
    """`1/b^2`, une puissance de plus qu'un fil seul : c'est un dipole.

    Les deux courants opposes forment un dipole magnetique, et un dipole
    decroit d'une puissance de plus qu'une source seule. C'est exactement
    l'argument de l'exercice 5 de statique, ou la charge et son image
    formaient un dipole electrique.
    """
    champ_y = champ_sur_la_mediatrice(COURANT, DEMI_ECART, ORDONNEE)[1]
    limite = sp.limit(champ_y * ORDONNEE**2, ORDONNEE, sp.oo)
    assert sp.simplify(limite + DEMI_ECART * COURANT / sp.pi) == 0


def test_le_champ_s_annule_si_les_courants_sont_de_meme_sens() -> None:
    """Le revers : ce qui s'ajoute ici se detruirait la.

    Avec deux courants de meme sens, ce sont les composantes selon `Oy`
    qui se detruisent sur la mediatrice, et celles selon `Ox` qui
    s'ajoutent. Le champ y est donc **nul** selon `Oy`.

    Sans ce test, rien ne distinguerait une somme d'une difference : les
    deux donneraient une expression non nulle, et seule l'une est juste.
    """
    champ = champ_si_les_courants_sont_de_meme_sens(
        COURANT,
        DEMI_ECART,
        ORDONNEE,
    )
    assert sp.simplify(champ[1]) == 0
    assert sp.simplify(champ[0]) != 0
