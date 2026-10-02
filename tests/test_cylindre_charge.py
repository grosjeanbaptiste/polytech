"""L'exercice « Statique 1.1 » : le champ exterieur d'un cylindre charge.

Le corrige manuscrit resout l'equation de Poisson en cylindriques, obtient
`E = K/rho`, et fixe `K` par la relation de passage a la surface. Son
resultat encadre est exact.

Quatre choses lui manquent, et ce fichier les etablit.

**La route de Gauss**, qui donne le meme resultat sans passer par la
relation de passage : le flux a travers un cylindre de rayon `rho` et de
longueur `L` vaut la charge enlacee sur la permittivite. Deux routes
independantes.

**Pourquoi l'exposant est un et non deux.** Le champ d'une charge
ponctuelle decroit en `1/r^2`, celui d'un fil en `1/rho`. La raison est
que la divergence cylindrique n'est nulle que pour `1/rho` -- une
puissance de moins, parce que la symetrie a une dimension de moins ou
s'etaler.

**Le potentiel ne peut pas etre pris nul a l'infini.** Il est
logarithmique, donc il **diverge** au loin : pour un fil infini, la
reference habituelle est impossible. C'est une des rares configurations
ou la convention `V(oo) = 0` ne tient pas, et aucun enonce ne le signale.

**Le resultat suppose le cylindre infini.** `Q/L` n'est une densite
lineique uniforme que loin des extremites. Le manuscrit ecrit `Q/L` sans
dire a quelle condition.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import (
    PERMITTIVITE,
    champ_par_gauss,
    champ_par_poisson,
    densite_lineique,
    densite_surfacique,
    divergence_cylindrique,
    potentiel,
)

CHARGE = sp.Symbol("Q", positive=True)
LONGUEUR = sp.Symbol("L", positive=True)
RAYON = sp.Symbol("R", positive=True)
DISTANCE = sp.Symbol("rho", positive=True)
REFERENCE = sp.Symbol("rho_0", positive=True)


def test_la_densite_lineique_est_la_charge_par_unite_de_longueur() -> None:
    """`lambda = Q/L`, et c'est la grandeur qui gouverne tout l'exercice."""
    assert sp.simplify(densite_lineique(CHARGE, LONGUEUR) - CHARGE / LONGUEUR) == 0


def test_la_densite_surfacique_se_repartit_sur_toute_la_surface_laterale() -> None:
    """`sigma = Q / (2 pi R L)`, et le denominateur est une **aire**.

    Le manuscrit l'ecrit `(Q/L) x 1/(2 pi R)` en annotant le second
    facteur « surface du cylindre de longueur L ». Le produit est juste,
    mais l'annotation ne l'est pas : `1/(2 pi R)` n'est pas une surface
    -- c'est l'inverse d'une longueur. L'aire laterale est `2 pi R L`, et
    c'est elle qui divise la charge.
    """
    attendue = CHARGE / (2 * sp.pi * RAYON * LONGUEUR)
    obtenue = densite_surfacique(CHARGE, LONGUEUR, RAYON)
    assert sp.simplify(obtenue - attendue) == 0


def test_le_champ_encadre_par_le_manuscrit_est_retrouve() -> None:
    """`E = (Q/L) / (2 pi eps_0 rho)` : l'oracle est le resultat encadre.

    Ecrit ici a la main d'apres le scan, donc externe a la bibliotheque.
    """
    encadre = densite_lineique(CHARGE, LONGUEUR) / (2 * sp.pi * PERMITTIVITE * DISTANCE)
    obtenu = champ_par_poisson(CHARGE, LONGUEUR, RAYON, DISTANCE)
    assert sp.simplify(obtenu - encadre) == 0


def test_les_deux_routes_donnent_le_meme_champ() -> None:
    """Poisson et Gauss s'accordent, et aucun ne sert d'oracle a l'autre.

    La route de Poisson resout une equation differentielle puis fixe sa
    constante par la **relation de passage** a la surface ; celle de Gauss
    calcule un flux et ne touche jamais a la surface du conducteur. Les
    deux arrivent au meme champ.
    """
    par_poisson = champ_par_poisson(CHARGE, LONGUEUR, RAYON, DISTANCE)
    par_gauss = champ_par_gauss(CHARGE, LONGUEUR, DISTANCE)
    assert sp.simplify(par_poisson - par_gauss) == 0


def test_la_divergence_du_champ_trouve_est_nulle() -> None:
    """Le controle en boucle fermee : on etait parti de `div E = 0`.

    Le vide est sans charge, donc la divergence doit etre nulle partout
    hors du conducteur. Repartir du champ obtenu et la recalculer verifie
    l'integration.
    """
    champ = champ_par_poisson(CHARGE, LONGUEUR, RAYON, DISTANCE)
    assert sp.simplify(divergence_cylindrique(champ, DISTANCE)) == 0


def test_la_divergence_n_est_pas_nulle_pour_une_decroissance_en_carre() -> None:
    """Et voila pourquoi l'exposant est un, non deux.

    `1/rho^2` -- l'exposant du champ d'une charge ponctuelle -- laisse une
    divergence non nulle en cylindriques. La symetrie a une dimension de
    moins ou s'etaler, donc le champ decroit d'une puissance de moins.
    Sans ce test, le test precedent pourrait rendre zero pour tout.
    """
    en_carre = CHARGE / DISTANCE**2
    assert sp.simplify(divergence_cylindrique(en_carre, DISTANCE)) != 0


def test_la_relation_de_passage_se_referme_a_la_surface() -> None:
    """`sigma = eps_0 E(R)` : la route de Poisson, verifiee a l'envers.

    Le manuscrit **emploie** cette relation pour fixer sa constante. On
    la verifie ici sur le resultat final : reprendre le champ en
    `rho = R` et le multiplier par la permittivite doit redonner la
    densite surfacique. Une erreur de coefficient ne survivrait pas.
    """
    a_la_surface = champ_par_poisson(CHARGE, LONGUEUR, RAYON, RAYON)
    densite = densite_surfacique(CHARGE, LONGUEUR, RAYON)
    assert sp.simplify(PERMITTIVITE * a_la_surface - densite) == 0


def test_le_gradient_du_potentiel_redonne_le_champ() -> None:
    """`E = -dV/drho`, et c'est le second controle en boucle fermee."""
    tension = potentiel(CHARGE, LONGUEUR, DISTANCE, REFERENCE)
    champ = champ_par_poisson(CHARGE, LONGUEUR, RAYON, DISTANCE)
    assert sp.simplify(-sp.diff(tension, DISTANCE) - champ) == 0


def test_le_potentiel_s_annule_a_la_distance_de_reference() -> None:
    """Par construction : c'est la que la constante d'integration est fixee."""
    assert sp.simplify(potentiel(CHARGE, LONGUEUR, REFERENCE, REFERENCE)) == 0


def test_le_potentiel_diverge_a_l_infini_donc_la_reference_n_y_est_pas() -> None:
    """Le fait que le manuscrit ne dit pas, et qui compte.

    Pour un fil infini, le potentiel est logarithmique : il **diverge**
    au loin, et la convention habituelle `V(oo) = 0` est donc
    **impossible**. Il faut choisir une distance de reference finie, ce
    qui est une des rares configurations ou cela arrive.

    La raison physique : un fil infini porte une charge infinie. Amener
    une charge d'essai depuis l'infini demanderait un travail infini.
    """
    tension = potentiel(CHARGE, LONGUEUR, DISTANCE, REFERENCE)
    assert sp.limit(tension, DISTANCE, sp.oo) is -sp.oo


def test_le_champ_exterieur_ne_depend_pas_du_rayon_du_cylindre() -> None:
    """Le rayon se simplifie, et c'est un fait physique, non un accident.

    La route de Poisson fixe sa constante a `K = R sigma(R) / eps_0`. Or
    `sigma` porte deja un `1/R`, de sorte que les deux se detruisent :
    `K = lambda / (2 pi eps_0)`, sans trace du rayon.

    Autrement dit **deux cylindres de rayons differents portant la meme
    charge par unite de longueur creent le meme champ exterieur**. C'est
    la version cylindrique de ce qu'on a vu a l'exercice 8 : vu de
    dehors, seule la charge compte, pas la facon dont elle est disposee.

    Le rayon est donc un argument explicite de la fonction, et non un
    symbole interne : son effacement se verifie au lieu d'etre cache.
    """
    premier = sp.Symbol("R_a", positive=True)
    second = sp.Symbol("R_b", positive=True)
    champ_a = champ_par_poisson(CHARGE, LONGUEUR, premier, DISTANCE)
    champ_b = champ_par_poisson(CHARGE, LONGUEUR, second, DISTANCE)
    assert sp.simplify(champ_a - champ_b) == 0
    assert premier not in champ_a.free_symbols


def test_la_divergence_d_un_champ_lineaire_en_rho_vaut_exactement_deux() -> None:
    """Le seul test qui voit le facteur `1/rho` de l'operateur.

    Ecrit apres qu'une mutation l'ait prouve necessaire : retirer le
    `1/rho` de `divergence_cylindrique` ne faisait echouer aucun des onze
    tests precedents. La raison est que les deux tests de divergence
    portaient sur des champs ou le facteur **disparait** -- pour `K/rho`
    la divergence reste nulle, pour `K/rho^2` elle reste non nulle. Ni
    l'un ni l'autre ne mesure sa valeur.

    Un champ croissant lineairement, `E = rho`, a une divergence
    cylindrique **constante et egale a deux** : `(1/rho) d/drho (rho^2)`.
    Sans le facteur on obtiendrait `2 rho`, qui n'est pas constant.

    Et ce n'est pas un cas d'ecole : c'est le champ a l'interieur d'une
    distribution volumique **uniforme**, dont la divergence doit
    precisement etre constante.
    """
    assert sp.simplify(divergence_cylindrique(DISTANCE, DISTANCE) - 2) == 0


def test_la_divergence_rend_la_densite_d_une_distribution_uniforme() -> None:
    """Et la version physique du test precedent, avec les constantes.

    A l'interieur d'une distribution volumique uniforme `eta_v`, le champ
    est `eta_v rho / (2 eps_0)` -- consequence de Gauss sur un cylindre
    interieur. Sa divergence doit redonner `eta_v / eps_0`, ce qui boucle
    sur l'equation de Maxwell-Gauss.
    """
    volumique = sp.Symbol("eta_v", positive=True)
    champ = volumique * DISTANCE / (2 * PERMITTIVITE)
    attendue = volumique / PERMITTIVITE
    assert sp.simplify(divergence_cylindrique(champ, DISTANCE) - attendue) == 0
