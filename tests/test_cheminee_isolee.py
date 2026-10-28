"""La cheminee isolee de juin 2024 : le piege de la paroi supposee a 50 degres.

Donnees : `Exam_2022_2025_Exercices_cotes_resolus.md` (corpus de thermique ;
exercice de la cheminee d'usine, question bonus). Le corrige imprime
`Nu = 454,18`, `T_fs = 147,65`, `Phi = 11947,88 W`, `T_pi = 140,86` et
`T_pe = -5,93`. La bibliotheque refait le bilan ; elle retrouve les quatre
premiers a leur arrondi pres et `T_pe` a un degre pres (voir le test dedie).
"""

from __future__ import annotations

from polytech.maths.domain.comparaison import eloignees, proches
from polytech.thermique.application.cheminee_juin_2024 import (
    CHEMINEE,
    coefficient_exterieur,
    coefficient_interieur,
)
from polytech.thermique.domain.cheminee import etat_de_la_cheminee, paroi_coherente


def test_le_nusselt_interieur_vient_du_debit_et_vaut_454() -> None:
    """`Re = 4 q / (pi D mu)`, `Nu = 0,0234 Re^0,8 Pr^(1/3)` : 454,18 imprime."""
    k, nusselt = coefficient_interieur()
    assert proches(nusselt, 454.18, 0.5)
    assert proches(k, nusselt * 0.03525 / 1.0, 0.01)


def test_les_fumees_sortent_a_147_65_avec_l_isolant() -> None:
    """`T_fs = 147,65`, `Phi = 11948 W`, `T_pi = 140,86` (paroi supposee a 50 C)."""
    etat = etat_de_la_cheminee(CHEMINEE, coefficient_exterieur(50.0))
    assert proches(etat.sortie, 147.65, 0.01)
    assert proches(etat.flux, 11947.88, 5.0)
    assert proches(etat.paroi_interne, 140.86, 0.1)


def test_la_paroi_exterieure_supposee_a_50_est_fausse() -> None:
    """Le bilan la rend bien sous zero ; le corrige imprime -5,93, la lib -4,8."""
    etat = etat_de_la_cheminee(CHEMINEE, coefficient_exterieur(50.0))
    assert etat.paroi_externe < 0.0
    assert eloignees(etat.paroi_externe, 50.0, 50.0)
    assert proches(etat.paroi_externe, -4.78, 0.05)
    assert eloignees(etat.paroi_externe, -5.93, 1.0)  # ecart au corrige officiel


def test_le_point_fixe_de_la_paroi_exterieure_est_coherent() -> None:
    """Refaire `K_ray` avec la valeur trouvee : la paroi se stabilise."""
    tpe = paroi_coherente(CHEMINEE, coefficient_exterieur, depart=50.0)
    reprise = etat_de_la_cheminee(CHEMINEE, coefficient_exterieur(tpe)).paroi_externe
    assert proches(reprise, tpe, 1e-6)
    assert tpe < 0.0
