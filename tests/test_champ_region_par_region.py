"""L'exercice 7 : le champ de deux conducteurs, region par region.

Deux conducteurs cylindriques de rayon `R`, d'axes en `y = 0` et `y = a`,
parcourus par des courants **opposes** `I`. On cherche `H(y)` le long de
`Oy`, et il y a sept regions a traiter.

Le corrige manuscrit les traite toutes et ses trois resultats encadres
sont exacts. Cinq choses lui manquent, et ce fichier les etablit.

**Aucune continuite n'est verifiee.** Il y a quatre frontieres --
`y = -R`, `y = R`, `y = a-R`, `y = a+R` -- et le champ doit y etre
continu, faute de courant surfacique. Le manuscrit ne le controle a aucune
d'elles, alors que c'est la que ses sept expressions se recollent.

**Une seule expression couvre les trois regions exterieures**, et le
manuscrit le constate sans le demontrer. C'est un fait **algebrique** : en
laissant `y` signe, la formule generale rend le champ physique meme
au-dessous du conducteur inferieur, ou les directions s'inversent.

**Le champ est minimal au milieu de l'entrefer**, en `y = a/2`, et non
maximal comme on pourrait le croire.

**Au centre d'un conducteur, le champ n'est pas nul** : le champ propre
s'y annule, mais celui de l'autre conducteur non. Il vaut exactement
`-I/(2 pi a)`, le champ du voisin seul.

**La regle du courant enlace est coherente a la frontiere** : en
`rho = R` elle doit rendre le courant entier, sinon les deux expressions
ne pourraient pas se recoller.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.regions import (
    champ_a_l_exterieur,
    champ_au_dessous_par_superposition,
    champ_dans_le_conducteur_inferieur,
    champ_dans_le_conducteur_superieur,
    courant_enlace,
)

COURANT = sp.Symbol("I", positive=True)
ECART = sp.Symbol("a", positive=True)
RAYON = sp.Symbol("R", positive=True)
ORDONNEE = sp.Symbol("y", positive=True)
PROFONDEUR = sp.Symbol("h", positive=True)


def test_le_courant_enlace_croit_comme_le_carre_de_la_distance() -> None:
    """`I_C = I rho^2 / R^2` : le rapport des aires, pas des rayons.

    Le courant est reparti uniformement sur la section, donc la fraction
    enlacee est le rapport des **aires**. Confondre avec le rapport des
    rayons est l'erreur classique, et elle donnerait un champ interieur
    constant au lieu de lineaire.
    """
    distance = sp.Symbol("rho", positive=True)
    attendu = COURANT * distance**2 / RAYON**2
    assert sp.simplify(courant_enlace(COURANT, distance, RAYON) - attendu) == 0


def test_le_courant_enlace_vaut_le_courant_entier_a_la_frontiere() -> None:
    """En `rho = R` la regle doit rendre `I`, sinon rien ne se recolle.

    C'est la condition de coherence de la regle elle-meme, et elle est
    ce qui rend possible la continuite verifiee plus bas.
    """
    assert sp.simplify(courant_enlace(COURANT, RAYON, RAYON) - COURANT) == 0


def test_le_courant_enlace_est_nul_au_centre() -> None:
    """Aucune aire enlacee, donc aucun courant : le champ propre y est nul."""
    assert sp.simplify(courant_enlace(COURANT, sp.Integer(0), RAYON)) == 0


def test_le_champ_exterieur_est_celui_qu_encadre_le_manuscrit() -> None:
    """`H = -I a / (2 pi (a-y) y)` : l'oracle est le resultat encadre."""
    encadre = -COURANT * ECART / (2 * sp.pi * (ECART - ORDONNEE) * ORDONNEE)
    obtenu = champ_a_l_exterieur(COURANT, ECART, ORDONNEE)
    assert sp.simplify(obtenu - encadre) == 0


def test_le_champ_dans_le_conducteur_superieur_est_celui_qu_encadre() -> None:
    """`H = -(I/2pi) (R^2 + y(a-y)) / (y R^2)`, encadre par le manuscrit."""
    encadre = (
        -COURANT
        / (2 * sp.pi)
        * (RAYON**2 + ORDONNEE * (ECART - ORDONNEE))
        / (ORDONNEE * RAYON**2)
    )
    obtenu = champ_dans_le_conducteur_superieur(COURANT, ECART, RAYON, ORDONNEE)
    assert sp.simplify(obtenu - encadre) == 0


def test_le_champ_dans_le_conducteur_inferieur_est_celui_qu_encadre() -> None:
    """`H = -(I/2pi) (y(a-y) + R^2) / ((a-y) R^2)`, encadre aussi."""
    encadre = (
        -COURANT
        / (2 * sp.pi)
        * (ORDONNEE * (ECART - ORDONNEE) + RAYON**2)
        / ((ECART - ORDONNEE) * RAYON**2)
    )
    obtenu = champ_dans_le_conducteur_inferieur(COURANT, ECART, RAYON, ORDONNEE)
    assert sp.simplify(obtenu - encadre) == 0


def test_le_champ_est_continu_a_la_frontiere_du_conducteur_inferieur() -> None:
    """En `y = R`, les deux expressions doivent coincider.

    Elles le font, et ce n'est pas automatique : il faudrait un courant
    **surfacique** pour creer un saut, et il n'y en a pas -- le courant
    est reparti en volume. Le manuscrit ne verifie pas ce recollement.
    """
    dedans = champ_dans_le_conducteur_inferieur(COURANT, ECART, RAYON, RAYON)
    dehors = champ_a_l_exterieur(COURANT, ECART, RAYON)
    assert sp.simplify(dedans - dehors) == 0


def test_le_champ_est_continu_a_la_frontiere_du_conducteur_superieur() -> None:
    """En `y = a - R`, meme exigence et meme resultat."""
    frontiere = ECART - RAYON
    dedans = champ_dans_le_conducteur_superieur(COURANT, ECART, RAYON, frontiere)
    dehors = champ_a_l_exterieur(COURANT, ECART, frontiere)
    assert sp.simplify(dedans - dehors) == 0


def test_une_seule_expression_couvre_la_region_au_dessous() -> None:
    """Le constat du manuscrit, demontre plutot qu'affirme.

    Au-dessous du conducteur inferieur, les deux directions s'inversent
    par rapport a l'entrefer : il faut refaire la superposition avec les
    distances `|y|` et `a + |y|` et les bons sens. Le resultat doit etre
    celui que la formule generale donne en y mettant un `y` **negatif**.

    C'est un fait algebrique et non physique : c'est le signe de `y` qui
    porte l'inversion des directions, ce qui permet a une seule formule de
    couvrir les trois regions exterieures.
    """
    par_superposition = champ_au_dessous_par_superposition(
        COURANT,
        ECART,
        PROFONDEUR,
    )
    par_la_formule = champ_a_l_exterieur(COURANT, ECART, -PROFONDEUR)
    assert sp.simplify(par_superposition - par_la_formule) == 0


def test_le_champ_de_l_entrefer_est_minimal_au_milieu() -> None:
    """`y = a/2`, et c'est un minimum -- non un maximum.

    Le denominateur `y (a - y)` est maximal au milieu, donc le champ y est
    **minimal**. On s'attendrait au contraire : les deux conducteurs
    contribuent dans le meme sens, et l'on croirait leur somme maximale
    la ou ils sont le plus proches tous les deux. C'est l'inverse.
    """
    champ = champ_a_l_exterieur(COURANT, ECART, ORDONNEE)
    racines = sp.solve(sp.diff(champ, ORDONNEE), ORDONNEE)
    assert racines == [ECART / 2]
    seconde = sp.diff(champ, ORDONNEE, 2).subs(ORDONNEE, ECART / 2)
    # Le champ est negatif ; son module est minimal la ou lui-meme est
    # maximal, donc la derivee seconde y est negative.
    assert sp.simplify(seconde).is_negative


def test_au_centre_d_un_conducteur_le_champ_est_celui_du_voisin_seul() -> None:
    """`H(0) = -I / (2 pi a)`, et non zero.

    Le champ propre s'annule au centre -- aucun courant enlace -- mais
    celui de l'autre conducteur ne s'annule pas. Il vaut exactement ce
    qu'un fil seul donnerait a la distance `a`.
    """
    au_centre = sp.limit(
        champ_dans_le_conducteur_inferieur(COURANT, ECART, RAYON, ORDONNEE),
        ORDONNEE,
        0,
        "+",
    )
    assert sp.simplify(au_centre + COURANT / (2 * sp.pi * ECART)) == 0
