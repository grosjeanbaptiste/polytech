"""L'exercice 7 : le meme probleme que le 5, par une autre route.

Le corrige manuscrit de l'exercice 7 reprend la charge devant un plan,
mais autrement : il part du **theoreme de Gauss** pour etablir le champ
d'une charge ponctuelle, puis projette les deux contributions par la
**trigonometrie** au lieu de manipuler des vecteurs unitaires.

Et il aboutit exactement au meme resultat que l'exercice 5, aux lettres
pres -- `(a, y, z)` y devient `(h, x, 0)`. Ce fichier le verifie, et
cette identite est la verification la plus forte des deux exercices : ni
l'un ni l'autre n'est son propre oracle, et deux routes independantes qui
se rejoignent ne peuvent pas s'etre trompees de la meme facon.

La divergence nulle est le troisieme controle, et le plus profond : elle
atteste l'**exposant** du champ coulombien. Seul `1/r^2` a une divergence
nulle dans l'espace a trois dimensions, et c'est pourquoi Gauss donne cet
exposant-la.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.gauss import (
    champ_d_une_charge,
    champ_hors_d_une_sphere_uniforme,
    champ_par_projection,
    charge_dans_une_boule_uniforme,
    distance_au_point,
    divergence_du_champ_radial,
)
from polytech.electromagnetisme.domain.electrostatique.images import (
    PERMITTIVITE,
    champ_sur_le_plan,
    densite_induite,
)

CHARGE = sp.Symbol("Q", positive=True)
HAUTEUR = sp.Symbol("h", positive=True)
ABSCISSE = sp.Symbol("x", real=True)
RAYON = sp.Symbol("r", positive=True)


def test_le_flux_du_champ_de_gauss_vaut_la_charge_sur_la_permittivite() -> None:
    """`E . 4 pi r^2 = Q / eps_0` : le theoreme lui-meme, referme.

    Le champ est uniforme en module sur la sphere et normal a elle --
    c'est la symetrie spherique qui le permet -- donc le flux est le
    produit du module par l'aire. Retrouver `Q / eps_0` verifie que le
    champ rendu est bien celui que Gauss impose.
    """
    flux = champ_d_une_charge(CHARGE, RAYON) * 4 * sp.pi * RAYON**2
    assert sp.simplify(flux - CHARGE / PERMITTIVITE) == 0


def test_la_divergence_du_champ_coulombien_est_nulle_hors_de_la_source() -> None:
    """Zero, et c'est ce qui **impose** l'exposant deux.

    Hors des charges, l'equation de Maxwell-Gauss s'ecrit
    `div E = 0`. Un champ radial en `1/r^n` ne la satisfait que pour
    `n = 2` : c'est la dimension de l'espace qui fixe l'exposant, non une
    convention. Le test l'etablit en calculant la divergence en
    cartesiennes, sans supposer le resultat.
    """
    assert sp.simplify(divergence_du_champ_radial(2)) == 0


def test_la_divergence_n_est_pas_nulle_pour_un_autre_exposant() -> None:
    """Le revers, sans lequel le test precedent ne prouverait rien.

    Un test qui ne rend zero que pour la bonne valeur doit aussi montrer
    qu'il ne le rend pas pour les autres -- sinon il pourrait rendre zero
    partout, et ne rien distinguer.
    """
    assert sp.simplify(divergence_du_champ_radial(3)) != 0
    assert sp.simplify(divergence_du_champ_radial(1)) != 0


def test_la_distance_au_point_du_plan_est_la_meme_pour_les_deux_charges() -> None:
    """`d = sqrt(h^2 + x^2)`, et c'est ce qui fait fonctionner la methode.

    Comme dans l'exercice 5 : l'egalite des deux distances n'a lieu que
    sur le plan, et c'est elle qui annule les composantes tangentielles.
    """
    attendue = sp.sqrt(HAUTEUR**2 + ABSCISSE**2)
    assert sp.simplify(distance_au_point(HAUTEUR, ABSCISSE) - attendue) == 0


def test_la_route_trigonometrique_donne_le_champ_de_l_exercice_cinq() -> None:
    """Les deux routes se rejoignent, et c'est la verification croisee.

    L'exercice 7 projette par `sin(alpha) = h/d` ; l'exercice 5 soustrait
    des vecteurs unitaires. Aucun des deux ne sert d'oracle a l'autre :
    ils partent de principes differents -- Gauss d'un cote, la
    superposition de Coulomb de l'autre -- et aboutissent au meme champ.
    """
    par_la_trigo = champ_par_projection(CHARGE, HAUTEUR, ABSCISSE)
    par_les_vecteurs = champ_sur_le_plan(
        CHARGE,
        HAUTEUR,
        ABSCISSE,
        sp.Integer(0),
    )[0]
    assert sp.simplify(par_la_trigo - par_les_vecteurs) == 0


def test_la_densite_encadree_par_le_manuscrit_sept_est_celle_du_cinq() -> None:
    """Les deux resultats encadres sont la meme expression, aux lettres pres.

    C'est le constat que ce corrige apporte : les exercices 5 et 7 sont le
    **meme probleme**, pose deux fois avec d'autres notations et resolu
    par deux chemins. Le remarquer n'est pas une curiosite -- cela donne
    au lecteur une verification gratuite de l'un par l'autre.

    Le resultat du manuscrit 7 est ecrit ici tel qu'il l'encadre ; c'est
    donc un oracle externe, pas une sortie de la bibliotheque.
    """
    encadre_par_le_sept = (
        -CHARGE
        * HAUTEUR
        / (2 * sp.pi * (HAUTEUR**2 + ABSCISSE**2) ** sp.Rational(3, 2))
    )
    par_le_cinq = densite_induite(CHARGE, HAUTEUR, ABSCISSE, sp.Integer(0))
    assert sp.simplify(encadre_par_le_sept - par_le_cinq) == 0


RAYON_DE_LA_SPHERE = sp.Symbol("R", positive=True)


def test_la_boule_entiere_porte_toute_la_charge() -> None:
    """En integrant la densite uniforme jusqu'au bord, on retrouve `Q`.

    L'enonce du recueil ne parle pas d'une charge ponctuelle mais d'une
    **sphere isolante uniformement chargee**, de rayon `R`. C'est pour
    cela que le manuscrit ouvre sur le theoreme de Gauss : il faut
    d'abord etablir qu'elle agit, au-dehors, comme une charge ponctuelle.
    """
    enlacee = charge_dans_une_boule_uniforme(
        CHARGE, RAYON_DE_LA_SPHERE, RAYON_DE_LA_SPHERE
    )
    assert sp.simplify(enlacee - CHARGE) == 0


def test_a_l_interieur_la_charge_enlacee_croit_comme_le_cube_du_rayon() -> None:
    """`Q r^3 / R^3` : le rapport des volumes, la densite etant uniforme."""
    enlacee = charge_dans_une_boule_uniforme(CHARGE, RAYON_DE_LA_SPHERE, RAYON)
    assert sp.simplify(enlacee - CHARGE * RAYON**3 / RAYON_DE_LA_SPHERE**3) == 0


def test_hors_de_la_sphere_le_champ_ne_depend_plus_de_son_rayon() -> None:
    """Le champ exterieur est celui d'une charge ponctuelle au centre.

    C'est ce qui rend l'exercice 7 identique a l'exercice 5 : le rayon
    `R` disparait, et il ne reste que la charge totale et la distance au
    centre. La methode des images peut alors s'appliquer telle quelle.
    """
    champ = champ_hors_d_une_sphere_uniforme(CHARGE, RAYON_DE_LA_SPHERE, RAYON)
    assert RAYON_DE_LA_SPHERE not in champ.free_symbols
    assert sp.simplify(champ - champ_d_une_charge(CHARGE, RAYON)) == 0
