"""Planches d'electromagnetisme, chapitre I : introduction.

Source : `Electromagnetisme/SLIDES/Slides_01_Introduction.pdf`, planches 2
a 18, lues sur les pages rendues.

Le fascicule ne calcule rien, mais il affirme : des dates, un ordre de
grandeur (l'induction d'un appareil d'IRM vaut cent mille fois celle de la
terre), une relation (le courant secondaire d'un transformateur), et le
partage du cours en trois etats.
"""

from __future__ import annotations

import pytest

from polytech.electromagnetisme.domain.chronologie import apres, jalon
from polytech.electromagnetisme.domain.circuits.transformateur import (
    TransformateurParfait,
)
from polytech.electromagnetisme.domain.etats import (
    APPLICATIONS,
    Etat,
    etat_du_systeme,
)
from polytech.electromagnetisme.domain.magnetostatique.champ_terrestre import (
    COMPOSANTE_HORIZONTALE,
    COMPOSANTE_VERTICALE,
    intensite,
)
from polytech.electromagnetisme.domain.magnetostatique.milieux_aimantes import (
    induction_du_milieu,
)
from polytech.electromagnetisme.domain.premieres_applications import (
    PREMIERES_APPLICATIONS,
    application,
)


def test_les_premieres_applications_suivent_la_confirmation_de_hertz() -> None:
    """Planches 7 et 8 : Tesla en 1887, Marconi en 1901 « sur les travaux de Hertz »."""
    assert [a.annee for a in PREMIERES_APPLICATIONS] == [1887, 1901]
    assert application("Tesla").annee == jalon("Hertz").annee
    assert apres(application("Marconi"), jalon("Hertz"))
    assert "moteur" in application("Tesla").apport
    assert "radio" in application("Marconi").apport


def test_une_application_inconnue_est_refusee() -> None:
    with pytest.raises(KeyError, match="aucune application"):
        application("Edison")


def test_les_trois_etats_du_cours() -> None:
    """Planches 10, 12 et 14 : au repos, a courant constant, quelconque."""
    assert etat_du_systeme(charges_en_mouvement=False) is Etat.STATIQUE
    stationnaire = etat_du_systeme(charges_en_mouvement=True, courant_constant=True)
    assert stationnaire is Etat.STATIONNAIRE
    variable = etat_du_systeme(charges_en_mouvement=True, courant_constant=False)
    assert variable is Etat.VARIABLE


def test_chaque_etat_a_ses_quatre_applications() -> None:
    """Les quatre exemples que chaque planche enumere."""
    assert set(APPLICATIONS) == set(Etat)
    assert all(len(exemples) == 4 for exemples in APPLICATIONS.values())
    assert "Imprimantes LASER" in APPLICATIONS[Etat.STATIQUE]
    assert "Imagerie medicale (IRM)" in APPLICATIONS[Etat.STATIONNAIRE]
    assert "Transformateurs" in APPLICATIONS[Etat.VARIABLE]


def test_l_irm_vaut_cent_mille_fois_l_induction_terrestre() -> None:
    """Planche 13 : « 4,5 T = 100.000 fois l'induction magnetique terrestre ».

    Le syllabus (10.17) donne 15 et 35 A/m, soit 48 microteslas : le
    rapport vaut 94 000, que la planche arrondit.
    """
    champ = intensite(COMPOSANTE_HORIZONTALE, COMPOSANTE_VERTICALE)
    terrestre = induction_du_milieu(champ, 0.0)
    assert 4.5 / terrestre == pytest.approx(1e5, rel=0.07)


def test_le_transformateur_de_la_planche_14() -> None:
    """`i_2 = -(N_1 / N_2) i_1`, la seule relation ecrite du fascicule."""
    transformateur = TransformateurParfait(primaire=300, secondaire=100)
    assert transformateur.courant_secondaire(2) == -6
