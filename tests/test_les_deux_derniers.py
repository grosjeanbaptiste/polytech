"""Les deux derniers titres de la campagne de profondeur.

`Slides_4_part2` et `Slides_2_part2` sont les deux fascicules les plus
riches de l'electronique -- quarante et un et cinquante-six titres -- et
il leur manquait une entree chacun.

Pour le bruit : la **plage dynamique**, qui est ce que le chapitre sert a
calculer. Elle borne par le haut l'ecretage et par le bas le bruit, et
c'est elle qui dit combien de bits sont utiles dans une chaine.

Pour l'AOP : la **plage de sortie**, que les planches appellent « output
swing » et qui n'atteint jamais les alimentations. Elle est symetrique de
la plage d'entree en mode commun, deja portee, et c'est ce qui en fait un
test de coherence plutot qu'un ajout isole.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.couplage import (
    flux_a_travers_un_circuit,
)
from polytech.electronique.domain.boucle_fermee import (
    impedance_vue_en_boucle_ouverte,
    vitesse_de_balayage_minimale,
)
from polytech.electronique.domain.defauts_aop import (
    plage_d_entree_en_mode_commun,
    plage_de_sortie,
)
from polytech.electronique.domain.facteur_de_bruit import (
    plage_dynamique_en_decibels,
)
from polytech.signaux.domain.controleur import pulsation_de_correction
from polytech.signaux.domain.reponse_indicielle import minimum_de_resonance
from polytech.signaux.domain.transmittance import Transmittance


def test_la_plage_dynamique_compare_l_ecretage_au_bruit() -> None:
    """`20 log10(V_max / V_bruit)` : ce que le chapitre sert a calculer.

    Un volt de pleine echelle sur un microvolt de bruit fait exactement
    cent vingt decibels -- point fixe exact, et le chiffre qu'un
    dimensionnement de chaine vise.
    """
    assert plage_dynamique_en_decibels(1.0, 1e-6) == pytest.approx(120.0)


def test_la_plage_dynamique_dit_combien_de_bits_sont_utiles() -> None:
    """Recoupement avec la regle des six decibels par bit.

    Cent vingt decibels de plage dynamique autorisent au plus vingt bits
    utiles : au-dela, les bits de poids faible ne codent que du bruit. Les
    deux formules sont independantes, et leur accord n'est pas acquis.
    """
    plage = plage_dynamique_en_decibels(1.0, 1e-6)
    bits_utiles = math.floor((plage - 1.7609) / 6.0206)
    attendu = 19
    assert bits_utiles == attendu


def test_un_bruit_nul_donnerait_une_plage_infinie() -> None:
    with pytest.raises(ValueError, match="tension de bruit"):
        plage_dynamique_en_decibels(1.0, 0.0)


def test_une_pleine_echelle_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="pleine echelle"):
        plage_dynamique_en_decibels(0.0, 1e-6)


def test_la_plage_de_sortie_n_atteint_jamais_les_alimentations() -> None:
    """« Output swing » : il reste toujours une chute dans l'etage de sortie.

    Un AOP alimente en plus ou moins quinze volts ne sort pas quinze
    volts, et un montage dimensionne sans cette marge ecrete.
    """
    basse, haute = plage_de_sortie(-15.0, 15.0, chute=1.2)
    assert basse == pytest.approx(-13.8)
    assert haute == pytest.approx(13.8)


def test_la_plage_de_sortie_est_plus_etroite_que_les_alimentations() -> None:
    """Propriete, et non coincidence : la chute est positive par construction."""
    basse, haute = plage_de_sortie(-15.0, 15.0, chute=1.2)
    assert -15.0 < basse < haute < 15.0


def test_la_plage_de_sortie_et_celle_d_entree_sont_de_meme_nature() -> None:
    """Recoupement : les deux retranchent une marge aux alimentations.

    Les porter dans le meme module et les verifier l'une par l'autre evite
    de traiter l'une comme une loi et l'autre comme un detail -- ce sont
    deux fois la meme limitation d'etage.
    """
    entree = plage_d_entree_en_mode_commun(-15.0, 15.0, marge=1.2)
    sortie = plage_de_sortie(-15.0, 15.0, chute=1.2)
    assert entree == sortie


def test_une_chute_plus_large_que_l_alimentation_est_refusee() -> None:
    with pytest.raises(ValueError, match="retrait"):
        plage_de_sortie(-1.0, 1.0, chute=5.0)


def test_une_impedance_negative_en_boucle_ouverte_est_refusee() -> None:
    """Une impedance est positive : un signe est une faute de releve."""
    with pytest.raises(ValueError, match="impedance negative"):
        impedance_vue_en_boucle_ouverte(-1.0, 1e6)


def test_une_frequence_negative_est_refusee_dans_la_vitesse_de_balayage() -> None:
    """La vitesse de balayage se calcule sur un module, pas sur un signe."""
    with pytest.raises(ValueError, match="negative"):
        vitesse_de_balayage_minimale(frequence=-1e3, amplitude=1.0)


def test_une_chaine_sans_pulsation_de_correction_est_nommee() -> None:
    """Certaines chaines n'atteignent jamais l'argument vise.

    Un premier ordre pur ne depasse jamais -90 degres : lui demander une
    pulsation ou l'argument vaut -135 est sans solution, et le dire vaut
    mieux que de rendre une liste vide.
    """
    variable = sp.Symbol("p")
    with pytest.raises(ValueError, match="aucune pulsation"):
        pulsation_de_correction(
            Transmittance(1 / (1 + variable)),
            marge_visee=sp.pi / 4,
        )


def test_une_inductance_negative_est_refusee_dans_le_flux() -> None:
    """Elle inverserait le flux pour un courant positif."""
    with pytest.raises(ValueError, match="inductance negative"):
        flux_a_travers_un_circuit(inductance=-1e-3, courant=2.0)


def test_un_creux_ne_se_cherche_pas_sur_moins_de_trois_points() -> None:
    """Un minimum interieur demande deux voisins : c'est sa definition."""
    with pytest.raises(ValueError, match="moins de trois points"):
        minimum_de_resonance((1.2, 0.8))
