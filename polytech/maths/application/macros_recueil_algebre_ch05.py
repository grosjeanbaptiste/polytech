r"""Macros du corrige du chapitre V du recueil d'algebre, prefixe `AlgCinq`."""

from __future__ import annotations

from polytech.maths.application import recueil_algebre_ch05 as chapitre
from polytech.maths.application.recueil_algebre import ecriture, liste, nom

PREFIXE = "AlgCinq"
ORDRES_DU_1F = (2, 3, 4, 5)
ORDRES_DU_5 = (1, 2, 3, 4)
"""Les ordres auxquels le corrige montre la valeur avant de raisonner."""


def macros_du_chapitre_cinq() -> dict[str, str]:
    """Toutes les formules que le corrige du chapitre V compose."""
    produit, determinant_2 = chapitre.exercice_2()
    determinant_7, racines_7 = chapitre.exercice_7()
    determinant_8, racines_8 = chapitre.exercice_8()
    macros = {nom(PREFIXE, 1, k): ecriture(v) for k, v in chapitre.exercice_1().items()}
    macros |= {
        nom(PREFIXE, 1, "fSuite"): liste(
            chapitre.tridiagonale(n) for n in ORDRES_DU_1F
        ),
        nom(PREFIXE, 2, "Produit"): ecriture(produit),
        nom(PREFIXE, 2, "Det"): ecriture(determinant_2),
        nom(PREFIXE, 3, "Trois"): ecriture(chapitre.vandermonde(3)[0]),
        nom(PREFIXE, 5, "Suite"): liste(map(chapitre.exercice_5, ORDRES_DU_5)),
        nom(PREFIXE, 7, "Det"): ecriture(determinant_7),
        nom(PREFIXE, 7, "Racines"): liste(sorted(racines_7)),
        nom(PREFIXE, 8, "Det"): ecriture(determinant_8),
        nom(PREFIXE, 8, "Racines"): liste(sorted(racines_8)),
        f"{PREFIXE}CourbeSept": chapitre.courbe_du_7(),
    }
    return macros
