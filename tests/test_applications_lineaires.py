"""Les chapitres VII et VIII du syllabus : espaces vectoriels et applications.

VII.2 Familles liees et familles libres, VII.3 Familles generatrices d'un
sous-espace, bases et dimension, VII.5 Lien avec les systemes lineaires ;
VIII.5 Representation matricielle d'une application lineaire, VIII.6
Changement de base.

Le lien que VII.5 annonce est ce qui rend tout cela calculable : une
famille est libre si le systeme homogene qu'elle forme n'a que la
solution nulle, c'est-a-dire si son rang est plein. Les tests l'emploient
dans les deux sens.

Le changement de base est verifie par la propriete qui le definit, et non
par sa formule : la matrice changee de base doit representer **la meme**
application, donc transformer les memes vecteurs en les memes images --
exprimes dans la nouvelle base.
"""

from __future__ import annotations

import pytest

from polytech.maths.domain.algebre.applications import (
    applique,
    changement_de_base,
    dimension_engendree,
    est_generatrice,
    est_libre,
    matrice_d_une_application,
)
from polytech.maths.domain.algebre.matrices import produit, transposee

LIBRE = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
IDENTITE = ((1.0, 0.0), (0.0, 1.0))
LIEE = ((1.0, 2.0), (2.0, 4.0))


def test_une_famille_libre_a_un_rang_plein() -> None:
    """VII.5, le lien avec les systemes : liberte et rang disent la meme chose.

    Une famille est libre si le systeme homogene qu'elle forme n'a que la
    solution nulle -- donc si son rang egale le nombre de vecteurs.
    """
    assert est_libre(LIBRE)
    assert not est_libre(LIEE)


def test_deux_vecteurs_colineaires_sont_lies() -> None:
    """Le cas le plus simple, et celui qu'on verifie a vue."""
    assert not est_libre(((3.0, 6.0), (1.0, 2.0)))
    assert est_libre(((3.0, 6.0), (1.0, 3.0)))


def test_la_dimension_engendree_est_le_rang_de_la_famille() -> None:
    """Et elle ne compte pas les vecteurs, mais ceux qui apportent.

    Trois vecteurs dont deux colineaires engendrent un plan, non un
    espace de dimension trois.
    """
    assert dimension_engendree(LIBRE) == 3
    assert dimension_engendree(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0))) == 2


def test_une_famille_est_generatrice_si_elle_engendre_tout_l_espace() -> None:
    """VII.3 : generatrice et libre sont independants, et une base est les deux."""
    assert est_generatrice(LIBRE, dimension=3)
    assert not est_generatrice(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dimension=3)


def test_une_famille_peut_etre_generatrice_sans_etre_libre() -> None:
    """Quatre vecteurs du plan : generatrice, et liee.

    C'est ce qui justifie de definir une base par les **deux**
    proprietes, et non par l'une.
    """
    famille = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    assert est_generatrice(famille, dimension=2)
    assert not est_libre(famille)


def test_la_matrice_d_une_application_a_les_images_pour_colonnes() -> None:
    """VIII.5 : « representation matricielle d'une application lineaire ».

    Les colonnes sont les images des vecteurs de la base. C'est la
    convention, et l'oublier transpose la matrice -- erreur qui passe
    inapercue sur une application symetrique.
    """
    images = ((2.0, 0.0), (1.0, 3.0))
    matrice = matrice_d_une_application(images)
    assert matrice == ((2.0, 1.0), (0.0, 3.0))
    assert transposee(matrice) == images


def test_appliquer_la_matrice_a_un_vecteur_de_base_rend_son_image() -> None:
    """La verification qui ferme la boucle de VIII.5.

    Si les colonnes sont bien les images, appliquer la matrice au i-eme
    vecteur de la base canonique doit rendre la i-eme image.
    """
    images = ((2.0, 0.0), (1.0, 3.0))
    matrice = matrice_d_une_application(images)
    assert applique(matrice, (1.0, 0.0)) == pytest.approx(images[0])
    assert applique(matrice, (0.0, 1.0)) == pytest.approx(images[1])


def test_l_application_est_lineaire() -> None:
    """Propriete qui definit l'objet, donc a verifier plutot qu'a supposer."""
    matrice = ((2.0, 1.0), (0.0, 3.0))
    x, y = (1.0, 2.0), (-3.0, 0.5)
    a, b = 2.0, -1.5
    combinaison = tuple(a * u + b * v for u, v in zip(x, y, strict=True))
    attendu = tuple(
        a * u + b * v
        for u, v in zip(applique(matrice, x), applique(matrice, y), strict=True)
    )
    assert applique(matrice, combinaison) == pytest.approx(attendu)


def _a_plat(matrice: tuple[tuple[float, ...], ...]) -> list[float]:
    """Les coefficients en une seule liste.

    `pytest.approx` ne compare pas les structures imbriquees ; aplatir
    est plus honnete qu'une boucle de comparaisons, qui s'arreterait a la
    premiere ligne fausse sans dire les autres.
    """
    return [coefficient for ligne in matrice for coefficient in ligne]


def test_le_changement_de_base_conserve_l_application() -> None:
    """VIII.6, verifie par ce qui le definit et non par sa formule.

    `A' = P^-1 A P` doit representer la **meme** application. Si `P`
    envoie les coordonnees de la nouvelle base vers l'ancienne, alors
    pour tout vecteur `x'` exprime dans la nouvelle base, les deux
    chemins doivent concorder :

    * appliquer directement `A'` a `x'` ;
    * passer `x'` dans l'ancienne base par `P`, y appliquer `A`, puis
      revenir par `P^-1`.

    Le test compare les deux. Une faute sur l'inverse, ou une inversion
    de l'ordre du double produit, les ferait diverger.
    """
    matrice = ((2.0, 1.0), (0.0, 3.0))
    passage = ((1.0, 1.0), (0.0, 1.0))
    nouvelle = changement_de_base(matrice, passage)
    retour = changement_de_base(IDENTITE, passage)
    for coordonnees in ((3.0, -2.0), (1.0, 0.0), (0.0, 1.0), (-4.5, 2.5)):
        direct = applique(nouvelle, coordonnees)
        par_l_ancienne = applique(
            _inverse_par_le_retour(retour, passage),
            applique(matrice, applique(passage, coordonnees)),
        )
        assert list(direct) == pytest.approx(list(par_l_ancienne))


def _inverse_par_le_retour(
    retour: tuple[tuple[float, ...], ...],
    passage: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """L'inverse de `P`, obtenu sans le recalculer.

    `changement_de_base(I, P)` vaut `P^-1 I P`, soit l'identite -- ce qui
    ne donne pas l'inverse. On le reconstruit donc en resolvant, ce qui
    est un chemin different de celui du module : c'est ce qui rend la
    comparaison informative.
    """
    assert _a_plat(retour) == pytest.approx(_a_plat(IDENTITE))
    (a, b), (c, d) = passage
    determinant = a * d - b * c
    return ((d / determinant, -b / determinant), (-c / determinant, a / determinant))


def test_le_changement_de_base_par_l_identite_ne_change_rien() -> None:
    """Cas limite, et garde la plus economique du changement de base."""
    matrice = ((2.0, 1.0), (0.0, 3.0))
    changee = changement_de_base(matrice, IDENTITE)
    assert _a_plat(changee) == pytest.approx(_a_plat(matrice))


def test_le_changement_de_base_conserve_la_trace_et_le_determinant() -> None:
    """Deux invariants de similitude, et ils gardent le calcul.

    Une matrice changee de base represente la meme application : sa trace
    et son determinant ne peuvent pas bouger. Une faute sur l'inverse de
    la matrice de passage les deplacerait.
    """
    matrice = ((2.0, 1.0), (0.0, 3.0))
    passage = ((1.0, 1.0), (0.0, 1.0))
    nouvelle = changement_de_base(matrice, passage)
    trace = sum(matrice[i][i] for i in range(2))
    nouvelle_trace = sum(nouvelle[i][i] for i in range(2))
    assert nouvelle_trace == pytest.approx(trace)
    det = matrice[0][0] * matrice[1][1] - matrice[0][1] * matrice[1][0]
    nouveau = nouvelle[0][0] * nouvelle[1][1] - nouvelle[0][1] * nouvelle[1][0]
    assert nouveau == pytest.approx(det)


def test_une_matrice_de_passage_singuliere_est_refusee() -> None:
    """Elle ne definit pas une base : ses colonnes sont liees."""
    with pytest.raises(ValueError, match="matrice de passage singuliere"):
        changement_de_base(((1.0, 0.0), (0.0, 1.0)), ((1.0, 2.0), (2.0, 4.0)))


def test_une_famille_vide_est_refusee() -> None:
    """La liberte de la famille vide est une convention, non un calcul."""
    with pytest.raises(ValueError, match="famille vide"):
        est_libre(())


def test_un_vecteur_de_mauvaise_taille_est_refuse() -> None:
    """Appliquer une matrice a un vecteur trop court n'a pas de sens."""
    with pytest.raises(ValueError, match="vecteur de taille"):
        applique(((1.0, 0.0), (0.0, 1.0)), (1.0, 2.0, 3.0))


def test_le_produit_sert_bien_au_changement_de_base() -> None:
    """Garde de coherence entre les deux modules du chapitre.

    Le changement de base est un double produit matriciel ; si les deux
    modules divergeaient, ce test le dirait.
    """
    matrice = ((2.0, 1.0), (0.0, 3.0))
    passage = ((1.0, 1.0), (0.0, 1.0))
    assert produit(matrice, passage) == ((2.0, 3.0), (0.0, 3.0))
