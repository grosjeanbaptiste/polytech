"""Un lien entre deux cours : l'algebre lineaire, en optimisation et en maths.

Le chapitre I d'optimisation « rappelle » le rang, la matrice reguliere
et les systemes d'equations (I.3.2, I.3.3) ; le cours de mathematique
les enseigne. Les deux contextes de la bibliotheque les portent chacun
dans sa langue et par son chemin -- elimination a pivot partiel en
flottants d'un cote, rationnels exacts de l'autre. Ce test les fait
repondre aux memes questions, sur les matrices du cours d'optimisation :
s'ils divergeaient, l'un des deux cours serait mal porte.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre import rang as maths
from polytech.optimisation.domain import algebre_lineaire as optimisation

MATRICES = {
    "T de l'exemple, page 43": ((1, 1, 1, 0, 0), (-2, 3, 0, 1, 0), (1, -1, 0, 0, 1)),
    "base {1, 2, 4}": ((1, 1, 0), (-2, 3, 1), (1, -1, 0)),
    "sous-matrice {1, 3, 5} de la page 45": ((1, 1, 0), (0, 0, 0), (1, 0, 1)),
    "lignes proportionnelles": ((1, 2, 3), (2, 4, 6)),
}


@pytest.mark.parametrize("nom", sorted(MATRICES))
def test_les_deux_cours_donnent_le_meme_rang(nom: str) -> None:
    matrice = MATRICES[nom]
    assert maths.rang(matrice) == optimisation.rang(sp.Matrix(matrice))


@pytest.mark.parametrize("nom", sorted(MATRICES))
def test_reguliere_en_optimisation_veut_dire_inversible_en_mathematique(
    nom: str,
) -> None:
    matrice = MATRICES[nom]
    assert maths.est_inversible(matrice) is optimisation.est_reguliere(
        sp.Matrix(matrice)
    )


@pytest.mark.parametrize(
    ("second_membre", "compatible"),
    [((1, 2), True), ((1, 3), False)],
)
def test_le_critere_de_rouche_est_la_nature_du_systeme(
    second_membre: tuple[int, int],
    compatible: bool,  # noqa: FBT001 -- parametre de la table, pas un drapeau d'API
) -> None:
    """`x + y = 1`, `2 x + 2 y = ?` : compatible si et seulement si non impossible."""
    matrice = ((1, 1), (2, 2))
    nature = optimisation.nature_du_systeme(
        sp.Matrix(matrice), sp.Matrix(second_membre)
    )
    assert maths.systeme_compatible(matrice, second_membre) is compatible
    assert (nature is not optimisation.NatureDuSysteme.IMPOSSIBLE) is compatible


def test_le_noyau_de_t_a_la_dimension_des_variables_hors_base() -> None:
    """Theoreme du rang : `n - r(T) = n - m`, le nombre de variables hors base."""
    t = MATRICES["T de l'exemple, page 43"]
    variables, equations = len(t[0]), len(t)
    assert maths.dimension_du_noyau(t) == variables - equations
