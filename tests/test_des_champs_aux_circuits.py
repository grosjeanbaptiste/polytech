"""Le chapitre 1 du cours d'electronique, et le critere qui fonde tout le reste.

Second aveu. L'inventaire faisait venir ce chapitre de `Slides_1_part3` et
le declarait porte par le module `laplace` -- alors que `Slides_1_part1`
traite « de l'electromagnetisme aux circuits » : Maxwell microscopique et
macroscopique, Jefimenko, Biot-Savart, la force de Lorentz, le modele de
Drude, le ferromagnetisme. J'avais attribue les sources d'apres la
numerotation des fascicules, sans les lire. C'est exactement ce que la
consigne du depot interdit, et deux chapitres en ont ete mal declares.

Ce que ce chapitre apporte vraiment est **un critere** : la longueur
electrique. `theta = 2 pi l / lambda`, et la planche donne la regle en
clair -- « a common rule of thumb for propagation delays to start being
negligible is that l_c < lambda/20 ». C'est lui qui autorise les lois de
Kirchhoff, donc tout le reste du cours. Un corrige qui appliquerait
l'analyse nodale au-dela serait faux sans le paraitre.

Et un recoupement que je n'attendais pas : la conductivite de Drude du
cuivre, calculee avec les valeurs usuelles de densite electronique et de
temps de vol libre, retombe sur la conductivite que
`polytech.electromagnetisme` declare pour l'effet pelliculaire -- a un
demi pour cent. Deux contextes bornes, deux routes, un seul nombre.
"""

from __future__ import annotations

import math

import pytest

from polytech.electromagnetisme.domain.effet_pelliculaire import (
    CONDUCTIVITE_DU_CUIVRE,
)
from polytech.electronique.domain.des_champs_aux_circuits import (
    FRACTION_DE_LONGUEUR_D_ONDE,
    capacite_parasite,
    conductivite_de_drude,
    frequence_limite,
    les_constantes_localisees_valent,
    longueur_d_onde,
    longueur_electrique,
)

LUMIERE = 3e8
DENSITE_DU_CUIVRE = 8.5e28
"""Electrons libres par metre cube. Valeur usuelle de la litterature."""

TEMPS_DE_VOL_DU_CUIVRE = 2.5e-14
"""Temps de vol libre moyen, en secondes. Meme provenance."""


def test_la_longueur_d_onde_decroit_avec_la_frequence() -> None:
    assert longueur_d_onde(LUMIERE, 1e9) == pytest.approx(0.3)


def test_la_longueur_electrique_est_un_angle_et_non_une_longueur() -> None:
    """Dix centimetres a un gigahertz font un tiers de longueur d'onde.

    Soit `2 pi / 3` radians. La planche insiste sur l'unite parce que
    l'oubli est courant -- et parce qu'un angle de 2 radians n'a rien a
    voir avec une longueur de 2 metres.
    """
    assert longueur_electrique(0.1, LUMIERE, 1e9) == pytest.approx(2 * math.pi / 3)


def test_une_longueur_nulle_ne_dephase_rien() -> None:
    assert longueur_electrique(0.0, LUMIERE, 1e9) == pytest.approx(0.0)


def test_le_critere_de_la_planche_est_celui_du_vingtieme() -> None:
    """« l_c < lambda / 20 », ecrit en clair sur la planche.

    Vingt, et non dix : la valeur decide de la frequence a laquelle il
    faut abandonner Kirchhoff, et la changer changerait le cours.
    """
    assert FRACTION_DE_LONGUEUR_D_ONDE == 20


def test_un_circuit_court_a_basse_frequence_tient_en_constantes_localisees() -> None:
    """Dix centimetres a cent megahertz : la longueur d'onde fait trois metres."""
    assert les_constantes_localisees_valent(0.1, LUMIERE, 1e8)


def test_le_meme_circuit_a_un_gigahertz_ne_tient_plus() -> None:
    """La longueur d'onde tombe a trente centimetres : le critere est viole.

    C'est le moment ou l'analyse nodale cesse de valoir, et ou un corrige
    qui l'appliquerait quand meme serait faux.
    """
    assert not les_constantes_localisees_valent(0.1, LUMIERE, 1e9)


def test_la_frequence_limite_est_l_inverse_exact_du_critere() -> None:
    """A la frequence limite, le critere est a la limite d'etre viole."""
    limite = frequence_limite(0.1, LUMIERE)
    assert limite == pytest.approx(1.5e8)
    assert not les_constantes_localisees_valent(0.1, LUMIERE, limite)
    assert les_constantes_localisees_valent(0.1, LUMIERE, limite * 0.99)


def test_la_conductivite_de_drude_du_cuivre_recoupe_l_autre_contexte() -> None:
    """Le recoupement inattendu, et le plus solide de ce fichier.

    `sigma = n q^2 tau / m` avec les valeurs usuelles du cuivre retombe sur
    la conductivite que `polytech.electromagnetisme` declare pour l'effet
    pelliculaire. Les deux contextes sont etanches -- aucun n'importe
    l'autre -- et pourtant ils s'accordent a un demi pour cent.
    """
    calculee = conductivite_de_drude(DENSITE_DU_CUIVRE, TEMPS_DE_VOL_DU_CUIVRE)
    assert calculee == pytest.approx(CONDUCTIVITE_DU_CUIVRE, rel=0.01)


def test_la_conductivite_croit_avec_le_temps_de_vol_libre() -> None:
    """Moins de collisions, plus de conductivite : c'est tout le modele."""
    lent = conductivite_de_drude(DENSITE_DU_CUIVRE, 1e-14)
    rapide = conductivite_de_drude(DENSITE_DU_CUIVRE, 2e-14)
    assert rapide == pytest.approx(2 * lent)


def test_la_capacite_parasite_croit_quand_les_pistes_se_rapprochent() -> None:
    """Et c'est pourquoi un circuit reel a des poles que son schema n'a pas."""
    eloignee = capacite_parasite(8.854e-12, 1e-4, 1e-3)
    proche = capacite_parasite(8.854e-12, 1e-4, 1e-4)
    assert proche == pytest.approx(10 * eloignee, rel=1e-9, abs=0)


@pytest.mark.parametrize("frequence", [0.0, -1.0])
def test_une_frequence_non_positive_est_refusee(frequence: float) -> None:
    with pytest.raises(ValueError, match="frequence nulle ou negative"):
        longueur_d_onde(LUMIERE, frequence)


def test_une_vitesse_de_propagation_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="vitesse de propagation"):
        longueur_d_onde(0.0, 1e9)


def test_une_longueur_de_chemin_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="longueur de chemin negative"):
        longueur_electrique(-0.1, LUMIERE, 1e9)


def test_un_critere_sur_une_longueur_negative_est_refuse() -> None:
    with pytest.raises(ValueError, match="longueur de chemin negative"):
        les_constantes_localisees_valent(-0.1, LUMIERE, 1e9)


def test_une_frequence_limite_sur_une_longueur_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="longueur de chemin nulle"):
        frequence_limite(0.0, LUMIERE)


def test_une_frequence_limite_sans_vitesse_est_refusee() -> None:
    with pytest.raises(ValueError, match="vitesse de propagation"):
        frequence_limite(0.1, 0.0)


def test_une_densite_electronique_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="densite electronique"):
        conductivite_de_drude(0.0, TEMPS_DE_VOL_DU_CUIVRE)


def test_une_geometrie_de_capacite_parasite_invalide_est_refusee() -> None:
    with pytest.raises(ValueError, match="geometrie invalide"):
        capacite_parasite(8.854e-12, 0.0, 1e-3)


def test_une_permittivite_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="permittivite nulle"):
        capacite_parasite(0.0, 1e-4, 1e-3)
