"""L'exercice 8 : dix spires rectangulaires emboitees, de largeurs doublees.

L'enonce du recueil : dix spires rectangulaires infiniment longues, de
**largeurs** 2, 4, 8, ... 1024 cm, placees dans le plan `Oxy` avec `Oy`
pour axe de symetrie, toutes parcourues dans le sens anti-horlogique par
3 A. On demande le champ au point `(0, 0)`.

Chaque spire a donc ses deux longs cotes a la **demi-largeur** de l'axe :
1, 2, 4, ... 512 cm. C'est le seul piege de l'exercice -- prendre la
largeur pour la distance divise le resultat par deux, exactement -- et le
manuscrit ne s'y trompe pas : il ecrit le facteur un demi sous son trait
de fraction, annote « distance entre le conducteur et l'axe y ».

**Ce fichier a d'abord affirme le contraire, et c'etait une faute de
l'agent.** Sa transcription du manuscrit avait perdu ce facteur un demi.
Le calcul independant, voyant alors la formule donner la moitie de la
valeur encadree, en avait conclu que le manuscrit portait « deux erreurs
qui se compensent exactement » -- these d'un corrige entier, deux tests a
l'appui. La contradiction venait de la transcription, non du manuscrit.
Elle n'a ete levee qu'en ouvrant le recueil, qui parle de largeurs, puis
en relisant le scan.

La lecon vaut pour tous les corriges : quand un calcul contredit un
manuscrit, on relit le **scan** avant de conclure. Une transcription est
une lecture, et c'est la lecture qui s'etait trompee.
"""

from __future__ import annotations

import itertools

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.nappe import (
    champ_d_un_seul_cote,
    champ_de_la_nappe,
    champ_de_la_nappe_infinie,
    demi_largeurs,
    distances_geometriques,
    largeurs_des_spires,
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
    formule.
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


def test_les_largeurs_sont_celles_de_l_enonce() -> None:
    """De 2 a 1024 cm, en doublant : dix largeurs, lues dans le recueil."""
    largeurs = largeurs_des_spires(NOMBRE_DE_PAIRES)
    assert len(largeurs) == NOMBRE_DE_PAIRES
    assert largeurs[0] == sp.Rational(2, 100)
    assert largeurs[-1] == sp.Rational(1024, 100)


def test_la_distance_a_l_axe_est_la_demi_largeur() -> None:
    """L'axe `Oy` partage chaque spire : ses cotes sont a la demi-largeur.

    Les demi-largeurs sont donc exactement les distances que la figure du
    manuscrit porte sur son axe : 1, 2, 4, 8... cm.
    """
    distances = demi_largeurs(largeurs_des_spires(NOMBRE_DE_PAIRES))
    assert distances == distances_geometriques(DEPUIS_UN_CENTIMETRE, NOMBRE_DE_PAIRES)


def test_la_formule_du_manuscrit_donne_la_valeur_qu_il_encadre() -> None:
    """`sum (3 x 2) / (2 pi . 2^i . 10^-2 . 1/2)`, pour `i` de un a dix.

    Ecrite terme a terme, telle que le scan la porte, sans passer par les
    fonctions du domaine : c'est l'oracle. Le facteur un demi y est, et
    avec lui la formule rend bien `190,8`.
    """
    centieme = sp.Rational(1, 100)
    somme = sum(
        (
            sp.Integer(3) * 2 / (2 * sp.pi * 2**rang * centieme * sp.Rational(1, 2))
            for rang in range(1, NOMBRE_DE_PAIRES + 1)
        ),
        start=sp.Integer(0),
    )
    distances = demi_largeurs(largeurs_des_spires(NOMBRE_DE_PAIRES))
    assert sp.simplify(somme - champ_de_la_nappe(COURANT, distances)) == 0
    assert round(float(somme), 1) == float(ENCADRE_PAR_LE_MANUSCRIT)


def test_prendre_la_largeur_pour_la_distance_donne_la_moitie() -> None:
    """Le piege de l'enonce, chiffre : exactement la moitie.

    C'est aussi ce que donnait la formule du manuscrit **sans** son
    facteur un demi -- donc ce que la transcription fautive faisait
    calculer, et d'ou venait la fausse contradiction.
    """
    largeurs = largeurs_des_spires(NOMBRE_DE_PAIRES)
    par_les_largeurs = champ_de_la_nappe(COURANT, largeurs)
    juste = champ_de_la_nappe(COURANT, demi_largeurs(largeurs))
    assert sp.simplify(juste - 2 * par_les_largeurs) == 0


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
    s'arreter a dix sans rien perdre.
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
