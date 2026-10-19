"""Activite 2, exercice du bateau-pompe : ses hypotheses, chiffrees.

La verification independante du 2026-10-19 a trouve (H12) a (H14) non
chiffrees, le facteur de correction cinetique pris egal a un sans le dire,
et une puissance « d'environ 200 kW » tapee a la main.
"""

from __future__ import annotations

import pytest

from polytech.fluides.application import activite_2 as a2
from polytech.fluides.application import activite_2_bateau as bat
from polytech.mecanique.domain.pesanteur import PESANTEUR


def _part_cinetique() -> float:
    bateau = a2.bateau_pompe()
    return bateau.ejection**2 / (2 * PESANTEUR) / bateau.hauteur


def test_l_elasticite_du_facteur_cinetique_est_la_part_cinetique() -> None:
    """Le terme cinetique est lineaire en `alpha` : `d ln P / d ln alpha`."""
    assert bat.elasticite_du_facteur_cinetique() == pytest.approx(_part_cinetique())


def test_cinq_pour_cent_sur_alpha_se_recoupent_par_l_elasticite() -> None:
    """Refaire la puissance avec `alpha = 1,05` : l'ecart est exact, lineaire."""
    ecart = bat.ecart_pour_un_facteur_cinetique(1.05)
    assert ecart == pytest.approx(0.05 * _part_cinetique(), rel=1e-12)


def test_un_facteur_cinetique_sous_un_leve() -> None:
    """`<v^3> >= <v>^3` : un profil ne porte jamais moins que sa moyenne."""
    with pytest.raises(ValueError, match="facteur de correction cinetique sous un"):
        bat.ecart_pour_un_facteur_cinetique(0.9)


def test_une_mer_agitee_retire_sa_hauteur_cinetique() -> None:
    """(H12) : une surface a `v` apporte `v^2 / 2g` que la pompe n'a plus a donner."""
    attendu = -(1.0**2) / (2 * PESANTEUR) / a2.bateau_pompe().hauteur
    assert bat.ecart_pour_une_mer_agitee(1.0) == pytest.approx(attendu)
    assert bat.ecart_pour_une_mer_agitee(0.0) == 0.0


def test_une_vitesse_de_mer_negative_leve() -> None:
    with pytest.raises(ValueError, match="vitesse de la mer negative"):
        bat.ecart_pour_une_mer_agitee(-1.0)


def test_l_elasticite_des_pertes_est_leur_part_de_la_hauteur() -> None:
    """(H14) : `h_L` entre une fois dans la hauteur, rien d'autre ne le porte."""
    attendu = a2.PERTES / a2.bateau_pompe().hauteur
    assert bat.elasticite_des_pertes() == pytest.approx(attendu)


@pytest.mark.parametrize(
    ("valeur", "arrondi"),
    [(200.936, 200.0), (50.93, 51.0), (0.01234, 0.012), (139.2, 140.0)],
)
def test_deux_chiffres_significatifs(valeur: float, arrondi: float) -> None:
    assert bat.deux_chiffres(valeur) == pytest.approx(arrondi)


def test_les_macros_du_bateau() -> None:
    macros = bat.macros_bateau()
    assert macros["Pesanteur"] == "9.81"
    assert macros["BateauPuissanceArrondie"] == "200"
    assert macros["BateauVitesseArrondie"] == "51"
    assert macros["BateauElasticiteAlpha"] == "0.95"
    assert macros["BateauEcartAlphaPourcent"] == "4.7"
    assert macros["BateauElasticitePertes"] == "0.022"
    assert macros["BateauEcartMerPourcent"] == "-0.037"
