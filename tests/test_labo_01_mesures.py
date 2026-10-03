"""Le laboratoire I d'electromagnetisme : mesures, sonde, filtres.

Chaque valeur attendue est recoupee par un chemin independant :

- la sonde compensee : la capacite qui rend le rapport independant de la
  frequence est cherchee en annulant la derivee du rapport, sans supposer
  la condition du cours `R1 C1 = R2 C2` -- puis comparee a elle ;
- la coupure de la sonde non compensee, contre le pole du rapport ;
- le passe-bande `R-L-C`, dont les deux coupures doivent avoir pour
  moyenne geometrique la frequence de resonance ;
- la methode volt-amperemetrique, moins precise que l'ohmmetre, et la
  raison : le calibre du courant.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.application.enonces_labo_01 import (
    capacite_de_compensation,
    coupure_de_la_sonde,
    coupures_du_passe_bande,
    mesure_volt_amperemetrique,
    precision_de_l_ohmmetre,
    rapport_de_la_sonde,
    resonance_du_passe_bande,
)


def test_la_sonde_au_dixieme_demande_neuf_megohms() -> None:
    rapport = rapport_de_la_sonde(capacite_d_entree=0, capacite_de_sonde=0)
    assert rapport == sp.Rational(1, 10)


def test_la_compensation_rejoint_la_condition_du_cours() -> None:
    """`C2 = R1 C1 / R2 = 11/9 pF`, trouve sans la supposer."""
    assert capacite_de_compensation() == sp.Rational(11, 9 * 10**12)


def test_compensee_la_sonde_vaut_un_dixieme_a_toute_frequence() -> None:
    rapport = rapport_de_la_sonde(
        capacite_d_entree=sp.Rational(11, 10**12),
        capacite_de_sonde=capacite_de_compensation(),
    )
    assert sp.simplify(rapport - sp.Rational(1, 10)) == 0


def test_non_compensee_la_sonde_coupe_vers_seize_kilohertz() -> None:
    assert coupure_de_la_sonde() == pytest.approx(16076, abs=1)


def test_le_passe_bande_resonne_vers_cinq_kilohertz() -> None:
    assert resonance_du_passe_bande() == pytest.approx(5032.9, abs=0.1)


def test_les_coupures_encadrent_la_resonance_en_moyenne_geometrique() -> None:
    basse, haute = coupures_du_passe_bande()
    assert (basse * haute) ** 0.5 == pytest.approx(resonance_du_passe_bande())
    assert haute - basse == pytest.approx(1e3 / (2 * 3.141592653589793 * 0.01))


def test_l_ohmmetre_est_precis_au_dixieme_d_ohm() -> None:
    assert precision_de_l_ohmmetre() == pytest.approx(0.094)


def test_la_methode_volt_amperemetrique_est_moins_precise() -> None:
    """A quatre volts, l'incertitude est dominee par le calibre du courant."""
    _, incertitude = mesure_volt_amperemetrique(4.0)
    assert incertitude == pytest.approx(1.376, abs=1e-3)
    assert incertitude > 10 * precision_de_l_ohmmetre()
