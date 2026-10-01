"""Les familles de filtres que le fascicule compare, et le reseau en echelle.

Le fascicule enumere **cinq types** de filtres et quatre familles
d'approximation : Butterworth, Chebyshev de type I, Bessel, elliptique --
dite aussi de Cauer. Chacune optimise une chose differente, et le choix
est un compromis que le fascicule met en tableau.

Butterworth est **plat** dans la bande, Chebyshev plus **raide** au prix
d'une ondulation, Bessel a le temps de groupe le plus **constant** donc
deforme le moins un signal numerique, et l'elliptique est le plus raide de
tous au prix d'une ondulation des deux cotes.

Le **reseau en echelle LC** est la realisation passive de ces prototypes,
et le fascicule en donne trois exemples -- convertisseur alternatif-continu,
filtre LC, ligne micro-ruban.

L'ancrage est une propriete d'ordre, et elle est verifiable : a gabarit
egal, l'ordre croit de l'elliptique vers Butterworth. Aucune famille ne
peut etre a la fois la plus raide et la plus plate.
"""

from __future__ import annotations

import pytest

from polytech.electronique.domain.familles import (
    TYPES_DE_FILTRES,
    Famille,
    compromis_de_la_famille,
    ondulation_depuis_l_affaiblissement,
    ordre_par_famille,
    reseau_en_echelle,
)
from polytech.electronique.domain.gabarit import Gabarit

CLASSIQUE = Gabarit(
    affaiblissement_en_bande=3.0,
    affaiblissement_hors_bande=40.0,
    pulsation_passante=1.0,
    pulsation_bloquante=2.0,
)


def test_le_fascicule_enumere_cinq_types_de_filtres() -> None:
    """Passe-bas, passe-haut, passe-bande, coupe-bande, passe-tout."""
    attendu = 5
    assert len(TYPES_DE_FILTRES) == attendu
    assert "passe-tout" in TYPES_DE_FILTRES


def test_l_ordre_croit_de_l_elliptique_vers_butterworth() -> None:
    """A gabarit egal : la raideur se paie en ordre, et l'ordre est le cout.

    Aucune famille ne peut etre a la fois la plus raide et la plus plate,
    et c'est tout le tableau de comparaison du fascicule.
    """
    ordres = {famille: ordre_par_famille(CLASSIQUE, famille) for famille in Famille}
    assert ordres[Famille.ELLIPTIQUE] <= ordres[Famille.CHEBYSHEV]
    assert ordres[Famille.CHEBYSHEV] <= ordres[Famille.BUTTERWORTH]
    assert ordres[Famille.BUTTERWORTH] <= ordres[Famille.BESSEL]


def test_chaque_famille_declare_ce_qu_elle_optimise() -> None:
    """Le compromis est explicite, et non laisse a l'intuition."""
    assert "plat" in compromis_de_la_famille(Famille.BUTTERWORTH)
    assert "raide" in compromis_de_la_famille(Famille.CHEBYSHEV)
    assert "groupe" in compromis_de_la_famille(Famille.BESSEL)
    assert "deux" in compromis_de_la_famille(Famille.ELLIPTIQUE)


def test_l_ondulation_se_deduit_de_l_affaiblissement_en_bande() -> None:
    """`epsilon^2 = 10^(A/10) - 1` : trois decibels donnent une ondulation unite.

    C'est le point fixe du chapitre, et il explique pourquoi `3 dB` revient
    partout : c'est l'affaiblissement pour lequel l'ondulation vaut un.
    """
    assert ondulation_depuis_l_affaiblissement(3.0103) == pytest.approx(1.0, abs=1e-4)
    assert ondulation_depuis_l_affaiblissement(0.1) < 1.0


def test_un_affaiblissement_nul_ne_donne_aucune_ondulation() -> None:
    assert ondulation_depuis_l_affaiblissement(0.0) == pytest.approx(0.0)


def test_un_affaiblissement_negatif_est_refuse() -> None:
    """Un filtre ne peut pas amplifier dans sa bande passante."""
    with pytest.raises(ValueError, match="affaiblissement negatif"):
        ondulation_depuis_l_affaiblissement(-1.0)


def test_le_reseau_en_echelle_alterne_series_et_shunts() -> None:
    """La convention des tables : inductance serie d'abord, puis capacite shunt."""
    reseau = reseau_en_echelle((1.0, 2.0, 1.0))
    assert reseau == ("inductance serie", "capacite shunt", "inductance serie")


def test_un_reseau_en_echelle_vide_est_refuse() -> None:
    with pytest.raises(ValueError, match="prototype vide"):
        reseau_en_echelle(())


def test_un_element_non_positif_dans_l_echelle_est_refuse() -> None:
    with pytest.raises(ValueError, match="element non positif"):
        reseau_en_echelle((1.0, -2.0))


def test_un_ordre_impair_commence_et_finit_par_une_inductance() -> None:
    """Propriete de la topologie, et ce qui decide des terminaisons."""
    reseau = reseau_en_echelle((1.0, 2.0, 3.0, 4.0, 5.0))
    assert reseau[0] == reseau[-1] == "inductance serie"
