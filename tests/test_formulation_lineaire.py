"""Chapitre I.1 d'optimisation : forme generale, canonique, standard.

L'oracle est l'illustration du cours lui-meme
(`Optimisation/SLIDES/Slides_01_Les_bases_de_la_programmation_lineaire.pdf`,
page 25) : un probleme sous forme generale, puis ses formes canonique et
standard, recopiees coefficient par coefficient. Un second chemin
verifie l'equivalence : un point admissible du probleme d'origine,
traduit, reste admissible et garde sa valeur economique.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formes import forme_canonique, forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral

INF, SUP, EGA = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL


def _illustration() -> ProgrammeGeneral:
    """`max x1 - 4 x2`, `x1 <= 7`, `3 x1 - 2 x2 >= 4`, `x1 + x2 = 5`, x2 libre."""
    return ProgrammeGeneral(
        optimisation=Optimisation.MAX,
        objectif=(1, -4),
        contraintes=(
            Contrainte((1, 0), INF, 7),
            Contrainte((3, -2), SUP, 4),
            Contrainte((1, 1), EGA, 5),
        ),
        libres=frozenset({1}),
    )


def test_la_variable_libre_se_decompose_en_deux_positives() -> None:
    """`x2 = x2+ - x2-` : x2+ garde sa place, x2- devient x3 (page 25)."""
    positif = _illustration().a_variables_non_negatives()
    assert positif.objectif == (1, -4, 4)
    assert [c.coefficients for c in positif.contraintes] == [
        (1, 0, 0),
        (3, -2, 2),
        (1, 1, -1),
    ]
    assert positif.libres == frozenset()


def test_la_forme_canonique_est_celle_du_cours() -> None:
    canonique = forme_canonique(_illustration())
    assert canonique.optimisation is Optimisation.MIN
    assert canonique.objectif == (-1, 4, -4)
    assert [
        (c.coefficients, c.sens, c.second_membre) for c in canonique.contraintes
    ] == [
        ((-1, 0, 0), SUP, -7),
        ((3, -2, 2), SUP, 4),
        ((1, 1, -1), SUP, 5),
        ((-1, -1, 1), SUP, -5),
    ]


def test_la_forme_standard_est_celle_du_cours() -> None:
    """x4 et x5 sont les variables d'ecart ; l'egalite n'en recoit pas."""
    standard = forme_standard(_illustration())
    assert standard.couts == sp.Matrix([[-1, 4, -4, 0, 0]])
    assert standard.matrice == sp.Matrix(
        [[1, 0, 0, 1, 0], [3, -2, 2, 0, -1], [1, 1, -1, 0, 0]]
    )
    assert standard.seconds_membres == sp.Matrix([7, 4, 5])


def test_la_translation_n_ajoute_qu_une_variable() -> None:
    """`x_j = x_j' - xi` : une seule variable de plus, quelles que soient les libres."""
    deux_libres = ProgrammeGeneral(
        optimisation=Optimisation.MIN,
        objectif=(1, 1, 1),
        contraintes=(Contrainte((1, 2, 3), SUP, 1),),
        libres=frozenset({0, 2}),
    )
    translate = deux_libres.translater_les_libres()
    assert translate.objectif == (1, 1, 1, -2)
    assert translate.contraintes[0].coefficients == (1, 2, 3, -4)


def _point_admissible() -> tuple[int, int]:
    """`x1 = 3`, `x2 = 2` : `3 <= 7`, `9 - 4 = 5 >= 4`, `3 + 2 = 5`."""
    return 3, 2


def test_l_equivalence_canonique_par_un_point() -> None:
    x1, x2 = _point_admissible()
    traduit = (x1, x2, 0)
    canonique = forme_canonique(_illustration())
    assert all(c.est_satisfaite(traduit) for c in canonique.contraintes)
    assert canonique.valeur(traduit) == -(x1 - 4 * x2)


def test_l_equivalence_standard_par_un_point() -> None:
    """Les ecarts valent `7 - x1` et `3 x1 - 2 x2 - 4`."""
    x1, x2 = _point_admissible()
    point = sp.Matrix([x1, x2, 0, 7 - x1, 3 * x1 - 2 * x2 - 4])
    standard = forme_standard(_illustration())
    assert standard.matrice * point == standard.seconds_membres
    assert (standard.couts * point)[0, 0] == -(x1 - 4 * x2)


def test_une_contrainte_de_mauvaise_longueur_est_refusee() -> None:
    with pytest.raises(ValueError, match="coefficients pour"):
        ProgrammeGeneral(
            optimisation=Optimisation.MIN,
            objectif=(1, 2),
            contraintes=(Contrainte((1,), SUP, 0),),
        )


def test_une_variable_libre_hors_du_programme_est_refusee() -> None:
    with pytest.raises(ValueError, match="variable libre hors"):
        ProgrammeGeneral(
            optimisation=Optimisation.MIN,
            objectif=(1, 2),
            contraintes=(),
            libres=frozenset({5}),
        )


@pytest.mark.parametrize(
    ("sens", "point", "attendu"),
    [
        (INF, (1, 1), True),
        (INF, (3, 3), False),
        (SUP, (3, 3), True),
        (SUP, (1, 1), False),
        (EGA, (2, 2), True),
        (EGA, (1, 2), False),
    ],
)
def test_une_contrainte_est_satisfaite_selon_son_sens(
    sens: Sens,
    point: tuple[int, int],
    attendu: bool,  # noqa: FBT001 -- parametre de la table, pas un drapeau d'API
) -> None:
    """`x1 + x2 ? 4` : sous, sur et au-dessus de la droite."""
    assert Contrainte((1, 1), sens, 4).est_satisfaite(point) is attendu


def test_maximiser_z_equivaut_a_minimiser_son_oppose() -> None:
    minimise = _illustration().en_minimisation()
    assert minimise.optimisation is Optimisation.MIN
    assert minimise.objectif == (-1, 4)


def test_un_programme_deja_en_minimisation_ne_change_pas() -> None:
    minimise = _illustration().en_minimisation()
    assert minimise.en_minimisation() is minimise


def test_une_contrainte_retournee_change_de_sens_et_de_signe() -> None:
    assert Contrainte((1, 2), INF, 3).retournee() == Contrainte((-1, -2), SUP, -3)


def test_une_egalite_retournee_reste_une_egalite() -> None:
    assert Contrainte((1, 2), EGA, 3).retournee().sens is EGA
