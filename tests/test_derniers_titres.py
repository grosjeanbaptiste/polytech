"""Les treize derniers titres, un par un.

Ce fichier clot la campagne de profondeur. Chaque test reprend un titre
precis que les planches enoncent et que rien ne portait encore, et chacun
est ancre sur une valeur exacte, une propriete d'ordre ou un recoupement.
"""

from __future__ import annotations

import math

import pytest

from polytech.electromagnetisme.domain.magnetostatique.couplage import (
    flux_a_travers_un_circuit,
)
from polytech.electronique.domain.boucle_fermee import (
    vitesse_de_balayage_minimale,
)
from polytech.electronique.domain.facteur_de_bruit import (
    TECHNOLOGIES,
    bruit_dominant_selon_la_source,
)
from polytech.electronique.domain.familles import (
    Famille,
    ondulation_des_deux_cotes,
    procedure_de_dimensionnement,
    raideur_asymptotique,
)
from polytech.signaux.domain.architecture import (
    coherence_des_unites_du_gain_de_boucle,
    etapes_du_dimensionnement,
)
from polytech.signaux.domain.controleur import contraintes_frequentielles
from polytech.signaux.domain.transformees import (
    resolution_spectrale,
    spectre_de_raies,
)
from polytech.thermique.domain.convection.forcee import couplage_vitesse_temperature


def test_la_resolution_spectrale_ne_depend_que_de_la_duree() -> None:
    """`delta f = 1 / T` : doubler la duree double la resolution.

    Le fascicule y insiste parce que c'est la seule facon de l'ameliorer --
    augmenter la cadence d'echantillonnage etend la bande observee, pas la
    finesse.
    """
    assert resolution_spectrale(duree=1.0) == pytest.approx(1.0)
    assert resolution_spectrale(duree=2.0) == pytest.approx(0.5)


def test_une_duree_nulle_n_a_pas_de_resolution() -> None:
    with pytest.raises(ValueError, match="duree nulle"):
        resolution_spectrale(duree=0.0)


def test_un_signal_periodique_a_un_spectre_de_raies() -> None:
    """Et non un spectre continu : les raies sont aux multiples du fondamental.

    « Transformee de Fourier d'une fonction periodique - series de
    Fourier » : c'est le pont que le fascicule etablit, et il change la
    nature du spectre.
    """
    raies = spectre_de_raies(fondamentale=50.0, harmoniques=3)
    assert raies == (50.0, 100.0, 150.0)


def test_un_nombre_d_harmoniques_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="nombre d'harmoniques"):
        spectre_de_raies(fondamentale=50.0, harmoniques=0)


def test_les_technologies_du_fascicule_sont_enumerees() -> None:
    """« A comparison of technologies » : bipolaire, JFET, CMOS.

    Chacune a son compromis bruit en tension / bruit en courant, et c'est
    ce compromis qui decide du choix selon l'impedance de source.
    """
    attendu = 3
    assert len(TECHNOLOGIES) == attendu
    assert "JFET" in TECHNOLOGIES


def test_une_source_resistive_fait_dominer_le_bruit_en_courant() -> None:
    """« Link between noise factor and noise voltage and current model ».

    C'est le critere de choix d'une technologie : sur une source de forte
    impedance le bruit en courant domine, donc il faut un composant a
    faible bruit en courant -- un JFET plutot qu'un bipolaire.
    """
    assert bruit_dominant_selon_la_source(5e-9, 1e-12, 1e6) == "courant"
    assert bruit_dominant_selon_la_source(5e-9, 1e-12, 100.0) == "tension"


def test_une_resistance_de_source_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="resistance de source"):
        bruit_dominant_selon_la_source(5e-9, 1e-12, -1.0)


def test_la_raideur_asymptotique_croit_avec_l_ordre() -> None:
    """`20 n` decibels par decade : c'est le seul effet de l'ordre, a l'infini.

    Les familles ne se distinguent donc **pas** par leur raideur
    asymptotique -- toutes font `20 n` dB/decade -- mais par leur
    comportement **dans** la transition. C'est le point que le fascicule
    fait sur sa carte des poles et zeros.
    """
    assert raideur_asymptotique(ordre=1) == pytest.approx(20.0)
    assert raideur_asymptotique(ordre=5) == pytest.approx(100.0)


def test_un_ordre_nul_n_a_pas_de_raideur() -> None:
    with pytest.raises(ValueError, match="ordre nul"):
        raideur_asymptotique(ordre=0)


def test_seul_l_elliptique_ondule_des_deux_cotes() -> None:
    """C'est ce qui lui donne sa raideur, et ce qui la lui fait payer."""
    assert ondulation_des_deux_cotes(Famille.ELLIPTIQUE)
    assert not ondulation_des_deux_cotes(Famille.CHEBYSHEV)
    assert not ondulation_des_deux_cotes(Famille.BUTTERWORTH)


def test_la_procedure_de_dimensionnement_a_quatre_etapes() -> None:
    """« Design procedure » : le fascicule la donne pour chaque famille.

    Elle est la meme pour toutes : gabarit, ordre, prototype, puis echelle
    et transposition. Seule la table du prototype change.
    """
    attendu = 4
    etapes = procedure_de_dimensionnement()
    assert len(etapes) == attendu
    assert "gabarit" in etapes[0]


def test_le_flux_a_travers_un_circuit_vaut_l_inductance_fois_le_courant() -> None:
    """`Phi = L i` : la definition meme de l'inductance propre.

    Le fascicule part de la pour tout le chapitre III.4, parce que c'est
    cette relation qui lie « les flux aux courants ».
    """
    assert flux_a_travers_un_circuit(inductance=1e-3, courant=2.0) == pytest.approx(
        2e-3,
    )


def test_le_gain_de_boucle_est_sans_dimension() -> None:
    """« Attention aux grandeurs physiques / unites », avertit la planche.

    Le produit de la chaine directe par la chaine de reaction doit etre
    sans dimension, sinon la comparaison a `-1` n'a aucun sens. C'est la
    verification que le fascicule demande avant tout trace de Nyquist.
    """
    assert coherence_des_unites_du_gain_de_boucle(
        unite_directe="m/V",
        unite_reaction="V/m",
    )
    assert not coherence_des_unites_du_gain_de_boucle(
        unite_directe="m/V",
        unite_reaction="m/V",
    )


def test_le_dimensionnement_d_une_boucle_a_quatre_etapes() -> None:
    """« Dimensionner une boucle de controle, c'est... » : la planche enumere."""
    attendu = 4
    etapes = etapes_du_dimensionnement()
    assert len(etapes) == attendu
    assert "capteur" in etapes[0]


def test_les_contraintes_frequentielles_viennent_des_specifications() -> None:
    """« Des specifications temporelles vers les caracteristiques frequentielles ».

    Le depassement devient une marge de phase, le temps de montee une
    bande passante : c'est la traduction que la synthese exige, parce
    qu'un correcteur se regle dans le domaine frequentiel.
    """
    marge, bande = contraintes_frequentielles(depassement=0.2, temps_de_montee=1.0)
    assert 0 < float(marge) < math.pi / 2
    assert float(bande) > 0


def test_un_depassement_nul_est_refuse_dans_les_contraintes() -> None:
    """Il exigerait une marge de phase de quatre-vingt-dix degres, inatteignable."""
    with pytest.raises(ValueError, match="depassement nul"):
        contraintes_frequentielles(depassement=0.0, temps_de_montee=1.0)


def test_la_vitesse_de_balayage_minimale_croit_avec_la_frequence() -> None:
    """`2 pi f A` : la pente maximale d'une sinusoide.

    C'est elle qu'un AOP doit suivre, et c'est la limite qui n'apparait sur
    aucun diagramme de Bode -- elle est non lineaire.
    """
    lente = vitesse_de_balayage_minimale(frequence=1e3, amplitude=1.0)
    rapide = vitesse_de_balayage_minimale(frequence=1e4, amplitude=1.0)
    assert rapide == pytest.approx(10 * lente)
    assert lente == pytest.approx(2 * math.pi * 1e3)


def test_une_amplitude_nulle_n_exige_aucune_vitesse() -> None:
    assert vitesse_de_balayage_minimale(frequence=1e3, amplitude=0.0) == 0.0


def test_le_couplage_vitesse_temperature_est_ce_qui_rend_la_convection_difficile() -> (
    None
):
    """Le champ de vitesse decide du champ de temperature, et reciproquement.

    « Couplage champ de vitesse / champ de temperature / flux a la paroi » :
    c'est pourquoi la convection se correle au lieu de se calculer, et
    c'est l'aveu que le chapitre fait d'entree.
    """
    couplage = couplage_vitesse_temperature()
    assert "vitesse" in couplage
    assert "temperature" in couplage
