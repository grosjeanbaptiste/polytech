"""Le chapitre IX du syllabus d'algebre : les espaces metriques.

Produit scalaire, norme, distance, familles orthogonales, procede de
Gram-Schmidt, projection orthogonale. Rien ne le portait.

Les tests n'evaluent pas des formules recopiees : ils verifient les
**proprietes** que le syllabus pose comme definitions.

Les trois axiomes du produit scalaire en sont, bien sur. Mais le plus
instructif est la projection : le syllabus la definit deux fois, et les
deux definitions doivent coincider. C'est « le seul vecteur `Y` de `W`
tel que `X - Y` soit orthogonal a tout `Z` de `W` », et c'est aussi
« parmi tous les vecteurs `Z` de `W`, celui qui minimise la distance
`||X - Z||` ». Les tests exigent les deux, et la seconde est verifiee en
echantillonnant d'autres vecteurs de `W` -- ce qui est une verification
numerique de la minimalite, et non la relecture d'une formule.
"""

from __future__ import annotations

import math

import pytest

from polytech.maths.domain.algebre.metrique import (
    distance,
    distance_au_sous_espace,
    est_orthogonale,
    est_orthonormale,
    gram_schmidt,
    norme,
    produit_scalaire,
    projection_orthogonale,
)

LIBRE = ((1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))


def test_le_produit_scalaire_est_positif_et_defini() -> None:
    """Premier axiome : `<X|X> >= 0`, et nul si et seulement si `X = 0`."""
    for x in ((1.0, 2.0), (-3.0, 0.5), (0.0, 0.0)):
        assert produit_scalaire(x, x) >= 0
    assert produit_scalaire((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0)
    assert produit_scalaire((1e-8, 0.0), (1e-8, 0.0)) > 0


def test_le_produit_scalaire_est_lineaire_en_son_premier_argument() -> None:
    """Deuxieme axiome, et il porte sur une combinaison, non sur un facteur.

    `<a X_1 + b X_2 | Y> = a <X_1|Y> + b <X_2|Y>` : le tester sur une
    combinaison des deux a la fois est plus fort que de tester
    l'additivite et l'homogeneite separement.
    """
    x1, x2, y = (1.0, 2.0, -1.0), (0.0, 3.0, 2.0), (2.0, -1.0, 4.0)
    a, b = 2.5, -1.5
    combinaison = tuple(a * u + b * v for u, v in zip(x1, x2, strict=True))
    attendu = a * produit_scalaire(x1, y) + b * produit_scalaire(x2, y)
    assert produit_scalaire(combinaison, y) == pytest.approx(attendu)


def test_le_produit_scalaire_est_symetrique() -> None:
    """Troisieme axiome, celui que le cas complexe remplace par l'hermitien."""
    x, y = (1.0, 2.0, -1.0), (2.0, -1.0, 4.0)
    assert produit_scalaire(x, y) == pytest.approx(produit_scalaire(y, x))


def test_la_norme_decoule_du_produit_scalaire() -> None:
    """La norme est celle que le produit scalaire induit.

    « Une fois le produit scalaire choisi, on utilisera la norme qui en
    decoule » : `||X|| = racine(<X|X>)`.
    """
    x = (3.0, 4.0)
    assert norme(x) == pytest.approx(5.0)
    assert norme(x) ** 2 == pytest.approx(produit_scalaire(x, x))


def test_la_distance_est_la_norme_de_la_difference() -> None:
    """La distance est la norme de la difference, et rien d'autre.

    « La distance entre un vecteur X et un vecteur Y est la norme de
    X - Y » : la definition du syllabus, mot pour mot.
    """
    assert distance((1.0, 1.0), (4.0, 5.0)) == pytest.approx(5.0)
    assert distance((1.0, 1.0), (1.0, 1.0)) == pytest.approx(0.0)


def test_l_inegalite_triangulaire_tient() -> None:
    """Consequence des axiomes, et ce qui fait de `d` une distance.

    Elle n'est pas posee dans le module : elle en decoule, et le test la
    verifie sur des triplets -- dont un aligne, ou elle est une egalite.
    """
    for x, y, z in (
        ((0.0, 0.0), (1.0, 2.0), (3.0, -1.0)),
        ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
    ):
        assert distance(x, z) <= distance(x, y) + distance(y, z) + 1e-12


def test_gram_schmidt_rend_une_famille_orthogonale() -> None:
    """Le procede rend bien une famille orthogonale de vecteurs non nuls.

    « A partir d'une famille libre, on peut construire une famille
    orthogonale de vecteurs non nuls », dit le syllabus.
    """
    orthogonale = gram_schmidt(LIBRE)
    assert est_orthogonale(orthogonale)
    assert all(norme(v) > 0 for v in orthogonale)
    assert len(orthogonale) == len(LIBRE)


def test_gram_schmidt_preserve_le_premier_vecteur_en_direction() -> None:
    """Le procede part du premier vecteur et ne le change pas.

    C'est ce qui le rend reproductible : l'ordre de la famille de depart
    determine le resultat, et un procede qui reordonnerait en silence
    rendrait des bases differentes d'une execution a l'autre.
    """
    orthogonale = gram_schmidt(LIBRE)
    assert orthogonale[0] == pytest.approx(LIBRE[0])


def test_normer_une_famille_orthogonale_la_rend_orthonormale() -> None:
    """Normer suffit a passer de l'orthogonal a l'orthonormal.

    « Il suffit de normer chacun des v_i pour obtenir une famille
    orthonormale », dit le syllabus.
    """
    orthogonale = gram_schmidt(LIBRE)
    normee = tuple(tuple(c / norme(v) for c in v) for v in orthogonale)
    assert est_orthonormale(normee)
    assert not est_orthonormale(orthogonale)


def test_la_projection_est_orthogonale_au_sous_espace() -> None:
    """Premiere definition : `X - Y` orthogonal a tout `Z` de `W`.

    Il suffit de le verifier sur une base de `W`, la linearite faisant
    le reste -- et c'est ce que le test fait.
    """
    base = gram_schmidt(((1.0, 1.0, 0.0), (1.0, 0.0, 1.0)))
    x = (2.0, -1.0, 3.0)
    projection = projection_orthogonale(x, base)
    residu = tuple(a - b for a, b in zip(x, projection, strict=True))
    for vecteur in base:
        assert produit_scalaire(residu, vecteur) == pytest.approx(0.0, abs=1e-12)


def test_la_projection_minimise_la_distance_au_sous_espace() -> None:
    """Seconde definition, et les deux doivent coincider.

    « La projection orthogonale de X sur W est, parmi tous les vecteurs Z
    de W, celui qui minimise la distance ||X - Z||. » Le test
    l'echantillonne : on construit d'autres vecteurs de `W` et aucun ne
    doit faire mieux.
    """
    base = gram_schmidt(((1.0, 1.0, 0.0), (1.0, 0.0, 1.0)))
    x = (2.0, -1.0, 3.0)
    meilleure = distance_au_sous_espace(x, base)
    for a in (-2.0, -0.5, 0.0, 0.5, 1.0, 3.0):
        for b in (-1.5, 0.0, 0.7, 2.0):
            z = tuple(a * u + b * v for u, v in zip(base[0], base[1], strict=True))
            assert distance(x, z) >= meilleure - 1e-12


def test_le_theoreme_de_pythagore_relie_projection_et_residu() -> None:
    """`||X||^2 = ||proj||^2 + ||X - proj||^2`.

    Consequence de l'orthogonalite, et verification croisee des deux :
    une projection fausse romprait l'egalite meme si elle restait dans
    le sous-espace.
    """
    base = gram_schmidt(((1.0, 1.0, 0.0), (1.0, 0.0, 1.0)))
    x = (2.0, -1.0, 3.0)
    projection = projection_orthogonale(x, base)
    ecart = distance_au_sous_espace(x, base)
    assert norme(x) ** 2 == pytest.approx(norme(projection) ** 2 + ecart**2)


def test_un_vecteur_du_sous_espace_est_sa_propre_projection() -> None:
    """Et sa distance au sous-espace est nulle."""
    base = gram_schmidt(((1.0, 1.0, 0.0), (1.0, 0.0, 1.0)))
    z = tuple(2.0 * u - 3.0 * v for u, v in zip(base[0], base[1], strict=True))
    assert projection_orthogonale(z, base) == pytest.approx(z)
    assert distance_au_sous_espace(z, base) == pytest.approx(0.0, abs=1e-12)


def test_des_vecteurs_de_dimensions_differentes_sont_refuses() -> None:
    """Le produit scalaire n'a pas de sens entre espaces differents."""
    with pytest.raises(ValueError, match="dimensions differentes"):
        produit_scalaire((1.0, 2.0), (1.0, 2.0, 3.0))


def test_une_famille_liee_est_refusee_par_gram_schmidt() -> None:
    """« A partir d'une famille **libre** », precise le syllabus.

    Sur une famille liee, le procede produit un vecteur nul -- et une
    famille contenant le vecteur nul n'est pas une base. Le refus dit
    laquelle des hypotheses a manque.
    """
    with pytest.raises(ValueError, match="famille liee"):
        gram_schmidt(((1.0, 1.0), (2.0, 2.0)))


def test_une_base_non_orthogonale_est_refusee_par_la_projection() -> None:
    """La formule de projection suppose une base orthogonale.

    « Si {v_1, ..., v_k} est une base orthogonale de W, alors... » : la
    formule du syllabus ne vaut qu'a cette condition, et l'appliquer a
    une base quelconque donnerait un vecteur qui n'est pas la projection.
    Le refus previent une erreur silencieuse.
    """
    with pytest.raises(ValueError, match="base non orthogonale"):
        projection_orthogonale((1.0, 0.0), ((1.0, 0.0), (1.0, 1.0)))


def test_la_norme_du_vecteur_nul_est_nulle() -> None:
    """Cas limite, et il ne doit pas lever : zero a une norme."""
    assert norme((0.0, 0.0, 0.0)) == pytest.approx(0.0)
    assert math.isfinite(norme((0.0, 0.0, 0.0)))
