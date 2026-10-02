"""L'exercice 8 : une nappe de conducteurs en progression geometrique.

Dix paires de conducteurs, chacune portant `3 A`, placees de part et
d'autre de l'axe a des abscisses doublant a chaque fois. On cherche le
champ au centre.

Le resultat encadre par le manuscrit -- `190,8 A/m` -- est **juste**. Mais
le chemin qui y mene porte **deux erreurs qui se compensent exactement**,
et ce fichier le demontre.

**La premiere erreur est dans les distances.** La figure place les
conducteurs a `+-1, +-2, +-4, +-8...` cm. La somme du manuscrit, elle,
ecrit les distances `2^i x 10^-2` pour `i` de un a dix, soit `2` a
`1024` cm -- le premier conducteur a saute. La somme des inverses y vaut
donc la **moitie** de la vraie.

**La seconde est dans le facteur.** La ligne de formule du manuscrit
ecrit `(3 x 2) / (2 pi d)`, soit `3/(pi d)` -- c'est juste. Sa ligne de
developpement ecrit `(6/pi) x somme`, soit **le double**. Le `2` du
`2 pi` a disparu.

**Et les deux se compensent.** Doubler le facteur et diviser la somme par
deux rend exactement la bonne valeur. Le resultat encadre est donc juste,
alors qu'aucune des deux lignes qui y menent ne l'est -- et un etudiant
qui reprend l'une sans l'autre se trompe du facteur deux.

C'est la raison d'etre de ce corrige : un resultat juste obtenu par un
chemin faux est plus dangereux qu'un resultat faux, parce que rien ne
le signale.
"""

from __future__ import annotations

import itertools

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.nappe import (
    champ_d_un_seul_cote,
    champ_de_la_nappe,
    champ_de_la_nappe_infinie,
    champ_par_le_developpement_du_manuscrit,
    distances_geometriques,
    somme_des_inverses,
    somme_fermee,
)

COURANT = sp.Integer(3)
NOMBRE_DE_PAIRES = 10
DEPUIS_UN_CENTIMETRE = 0
DEPUIS_DEUX_CENTIMETRES = 1
ENCADRE_PAR_LE_MANUSCRIT = sp.Rational("190.8")


def test_les_distances_doublent_a_chaque_conducteur() -> None:
    """Progression geometrique de raison deux, en metres.

    La premiere distance est `1 cm` quand l'exposant part de zero, et la
    dixieme `512 cm` : c'est la lecture de la figure.
    """
    distances = distances_geometriques(DEPUIS_UN_CENTIMETRE, NOMBRE_DE_PAIRES)
    assert len(distances) == NOMBRE_DE_PAIRES
    assert distances[0] == sp.Rational(1, 100)
    assert distances[-1] == sp.Rational(512, 100)
    assert all(
        suivante == 2 * actuelle for actuelle, suivante in itertools.pairwise(distances)
    )


def test_la_forme_fermee_egale_la_somme_explicite() -> None:
    """`100 . 2^(1-premier) . (1 - 2^-n)` : deux calculs independants.

    La somme explicite additionne dix termes ; la forme fermee applique la
    formule de la serie geometrique. Qu'elles coincident verifie la
    formule, que le manuscrit emploie dans sa note de verification sans
    la demontrer.
    """
    for premier in (DEPUIS_UN_CENTIMETRE, DEPUIS_DEUX_CENTIMETRES):
        distances = distances_geometriques(premier, NOMBRE_DE_PAIRES)
        explicite = somme_des_inverses(distances)
        fermee = somme_fermee(premier, NOMBRE_DE_PAIRES)
        assert sp.simplify(explicite - fermee) == 0, premier


def test_la_lecture_de_la_figure_donne_le_resultat_encadre() -> None:
    """Partir de `1 cm` avec la formule juste donne `190,8 A/m`.

    C'est l'oracle : la valeur encadree par le manuscrit, lue sur le scan.
    Elle est donc externe a la bibliotheque.
    """
    distances = distances_geometriques(DEPUIS_UN_CENTIMETRE, NOMBRE_DE_PAIRES)
    champ = champ_de_la_nappe(COURANT, distances)
    # On compare les arrondis et non les rationnels : `sp.Rational` d'un
    # flottant rend son developpement binaire, qui n'egale jamais le
    # decimal qu'on voulait. Le manuscrit donne un chiffre apres la
    # virgule, et c'est a ce chiffre que la comparaison se fait.
    assert round(float(champ), 1) == float(ENCADRE_PAR_LE_MANUSCRIT)


def test_partir_de_deux_centimetres_donne_la_moitie() -> None:
    """Oublier le premier conducteur divise le champ par deux exactement.

    Exactement, et non approximativement : doubler toutes les distances
    d'une progression geometrique de raison deux revient a decaler
    l'indice, ce qui divise la somme des inverses par deux.
    """
    depuis_un = champ_de_la_nappe(
        COURANT,
        distances_geometriques(DEPUIS_UN_CENTIMETRE, NOMBRE_DE_PAIRES),
    )
    depuis_deux = champ_de_la_nappe(
        COURANT,
        distances_geometriques(DEPUIS_DEUX_CENTIMETRES, NOMBRE_DE_PAIRES),
    )
    assert sp.simplify(depuis_un - 2 * depuis_deux) == 0


def test_les_deux_erreurs_du_manuscrit_se_compensent_exactement() -> None:
    """Le coeur de ce corrige : facteur double, somme moitie, resultat juste.

    La ligne de developpement du manuscrit emploie `6/pi` au lieu de
    `3/pi` -- le double -- et l'applique aux distances partant de `2 cm`
    -- dont la somme des inverses est la moitie. Le produit retombe sur
    la bonne valeur.
    """
    par_le_manuscrit = champ_par_le_developpement_du_manuscrit(
        COURANT,
        distances_geometriques(DEPUIS_DEUX_CENTIMETRES, NOMBRE_DE_PAIRES),
    )
    juste = champ_de_la_nappe(
        COURANT,
        distances_geometriques(DEPUIS_UN_CENTIMETRE, NOMBRE_DE_PAIRES),
    )
    assert sp.simplify(par_le_manuscrit - juste) == 0


def test_le_developpement_du_manuscrit_double_sa_propre_formule() -> None:
    """Et la compensation n'est pas fortuite : c'est deux fois deux.

    Applique aux **memes** distances, le developpement du manuscrit rend
    exactement le double de ce que sa ligne de formule rend. Les deux
    lignes du meme corrige se contredisent donc, independamment de la
    question des distances.
    """
    distances = distances_geometriques(DEPUIS_DEUX_CENTIMETRES, NOMBRE_DE_PAIRES)
    par_la_formule = champ_de_la_nappe(COURANT, distances)
    par_le_developpement = champ_par_le_developpement_du_manuscrit(
        COURANT,
        distances,
    )
    assert sp.simplify(par_le_developpement - 2 * par_la_formule) == 0


def test_les_deux_cotes_contribuent_a_parts_egales() -> None:
    """Le facteur deux de la symetrie, verifie plutot que suppose.

    Les courants etant opposes **et** les positions symetriques, les deux
    inversions de signe se detruisent et les contributions s'ajoutent.
    Avec une seule inversion elles s'annuleraient, et le champ au centre
    serait nul.
    """
    distances = distances_geometriques(DEPUIS_UN_CENTIMETRE, NOMBRE_DE_PAIRES)
    total = champ_de_la_nappe(COURANT, distances)
    un_cote = champ_d_un_seul_cote(COURANT, distances)
    assert sp.simplify(total - 2 * un_cote) == 0


def test_dix_paires_donnent_presque_tout_le_champ_d_une_nappe_infinie() -> None:
    """La serie converge, et dix paires en donnent plus de 99,9 pour cent.

    Consequence : ajouter une onzieme paire, a `1024 cm`, changerait le
    resultat de moins d'un millieme. C'est pourquoi l'enonce peut
    s'arreter a dix sans rien perdre -- et le manuscrit ne le dit pas.
    """
    distances = distances_geometriques(DEPUIS_UN_CENTIMETRE, NOMBRE_DE_PAIRES)
    fini = champ_de_la_nappe(COURANT, distances)
    infini = champ_de_la_nappe_infinie(COURANT, DEPUIS_UN_CENTIMETRE)
    assert fini < infini
    assert float(fini / infini) > 0.999


def test_le_champ_de_la_nappe_infinie_a_une_forme_fermee() -> None:
    """`2 . 100 . I / pi` : la serie geometrique sommee jusqu'a l'infini.

    Le facteur deux est celui de la serie `sum 2^-k` de zero a l'infini.
    La forme fermee se verifie contre la limite de la somme finie.
    """
    infini = champ_de_la_nappe_infinie(COURANT, DEPUIS_UN_CENTIMETRE)
    rang = sp.Symbol("n", positive=True, integer=True)
    limite = sp.limit(
        COURANT / sp.pi * 100 * 2 * (1 - 2 ** (-rang)),
        rang,
        sp.oo,
    )
    assert sp.simplify(infini - limite) == 0
