"""Le resultat calcule doit etre confronte a la reponse publiee par le cours.

La planche d'enonce du cours (II.3, manometre multi-fluides) donne
« Rep. : p1 = 130 kPa ». Le corrige calculait 129,65 kPa sans jamais citer
cette corroboration, pourtant disponible dans le depot.
"""

import pytest

from polytech.fluides.application import enonce
from polytech.fluides.application.activite import ResultatsDeLActivite

TOLERANCE_ARRONDI_KPA = 0.5
"""La reponse du cours est donnee au kilopascal : l'ecart doit y tenir."""


def test_le_cours_publie_bien_une_reponse_pour_cet_exercice() -> None:
    """Elle est transcrite une seule fois, dans l'enonce."""
    assert enonce.REPONSE_DU_COURS.kilopascals == pytest.approx(130.0)


def test_le_resultat_calcule_concorde_avec_la_reponse_du_cours() -> None:
    """129,65 kPa arrondi au kilopascal donne bien les 130 kPa annonces."""
    calcule = ResultatsDeLActivite().executer().pression_air_kpa
    ecart = abs(calcule - enonce.REPONSE_DU_COURS.kilopascals)
    assert ecart < TOLERANCE_ARRONDI_KPA


def test_l_ecart_relatif_reste_sous_le_demi_pourcent() -> None:
    """Un ecart plus grand signalerait une erreur, pas un arrondi."""
    calcule = ResultatsDeLActivite().executer().pression_air_kpa
    relatif = abs(calcule - enonce.REPONSE_DU_COURS.kilopascals) / calcule
    assert relatif < 0.005
