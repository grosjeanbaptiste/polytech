"""Les sections VII.7 a VII.10 du syllabus : espaces vectoriels quelconques.

« Espaces vectoriels de matrices », « de polynomes », « de fonctions »,
puis « Espaces vectoriels quelconques ». L'ordre du syllabus va du plus
concret au plus general, et le point d'arrivee est qu'un espace vectoriel
n'est pas necessairement un espace de colonnes.

Le contenu calculable est la **dimension**, et elle reserve une surprise
a chaque etape :

* les matrices `n x m` forment un espace de dimension `n m` -- le produit
  et non la somme, parce qu'une base est faite des matrices n'ayant qu'un
  seul coefficient non nul ;
* les polynomes de degre au plus `n` forment un espace de dimension
  `n + 1` -- le « plus un » est le terme constant, et c'est l'erreur
  classique ;
* les fonctions continues sur un intervalle forment un espace de
  dimension **infinie**, et c'est la raison d'etre de la section VII.10 :
  tout ce qui precede s'y applique sauf ce qui suppose une base finie.
"""

from __future__ import annotations

import pytest

from polytech.maths.domain.algebre.dimensions import (
    DIMENSION_INFINIE,
    base_canonique_des_matrices,
    dimension_des_matrices,
    dimension_des_polynomes,
    est_de_dimension_finie,
)


def test_la_dimension_des_matrices_est_le_produit_des_formats() -> None:
    """`n m`, et non `n + m` : une base n'a qu'un coefficient non nul par terme."""
    assert dimension_des_matrices(2, 3) == 6
    assert dimension_des_matrices(3, 3) == 9
    assert dimension_des_matrices(1, 1) == 1


def test_la_base_canonique_des_matrices_a_bien_cette_cardinalite() -> None:
    """La dimension n'est pas posee : elle est le compte de la base.

    Et chaque element de la base a exactement un coefficient non nul,
    ce que le test verifie -- c'est ce qui en fait une base, et ce qui
    explique le produit.
    """
    base = base_canonique_des_matrices(2, 3)
    assert len(base) == dimension_des_matrices(2, 3)
    for element in base:
        non_nuls = [c for ligne in element for c in ligne if c != 0]
        assert non_nuls == [1.0]


def test_la_dimension_des_polynomes_compte_le_terme_constant() -> None:
    """`n + 1` pour le degre au plus `n` : le « plus un » est l'erreur classique.

    Les polynomes de degre au plus deux forment un espace de dimension
    trois -- `1`, `x`, `x^2` -- et non deux.
    """
    assert dimension_des_polynomes(2) == 3
    assert dimension_des_polynomes(0) == 1


def test_les_espaces_de_dimension_finie_se_distinguent_des_autres() -> None:
    """VII.10 : « espaces vectoriels quelconques », et c'est la le point.

    Un espace de fonctions n'a pas de base finie, et tout ce que les
    sections precedentes etablissent cesse d'y valoir des que la
    demonstration passe par une base. Le distinguer explicitement vaut
    mieux que de rendre un grand nombre.
    """
    assert est_de_dimension_finie(dimension_des_matrices(2, 2))
    assert est_de_dimension_finie(dimension_des_polynomes(5))
    assert not est_de_dimension_finie(DIMENSION_INFINIE)


def test_un_format_nul_est_refuse_pour_les_matrices() -> None:
    """Une matrice sans ligne n'est pas un cas du cours."""
    with pytest.raises(ValueError, match="format nul ou negatif"):
        dimension_des_matrices(0, 3)


def test_un_degre_negatif_est_refuse_pour_les_polynomes() -> None:
    """Le polynome nul se traite a part, et le cours ne lui donne pas de degre."""
    with pytest.raises(ValueError, match="degre negatif"):
        dimension_des_polynomes(-1)
