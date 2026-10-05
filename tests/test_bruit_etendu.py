"""Le chapitre 4 du cours d'electronique, cote bruit, et ce qu'il enseigne.

Un aveu prealable, parce qu'il explique ce fichier. L'inventaire de
couverture declarait ce chapitre « couvert » sur la foi d'un module de
dix-neuf lignes portant **une** fonction -- la densite spectrale du bruit
thermique -- alors que les quarante et un titres de contenu de la planche
enseignent aussi le bruit de grenaille, le bruit en 1/f, le modele en
tension et courant d'un quadripole bruyant, le bruit ramene a l'entree et
le facteur de bruit. La declaration etait fausse, et elle l'etait parce
que la granularite retenue -- un module par chapitre -- permettait de
declarer couvert un chapitre dont un sixieme etait porte.

Les ancrages sont de deux natures. Trois valeurs sont **exactes par
definition** et servent de points fixes : un facteur de bruit de 2 fait
exactement `3,01 dB`, et exactement `290 K` de temperature de bruit
equivalente avec la reference de l'IEEE. Le reste est verifie par ses
proprietes : le bruit en courant pese d'autant plus que la resistance de
source est grande, le bruit blanc croit comme la largeur de bande et le
bruit rose comme son logarithme.
"""

from __future__ import annotations

import math

import pytest

from polytech.electromagnetisme.domain.constantes import CHARGE_ELEMENTAIRE
from polytech.electronique.domain.bruit import (
    densite_spectrale_de_grenaille,
    densite_spectrale_en_un_sur_f,
    puissance_integree_blanc_et_rose,
    tension_efficace_ramenee_a_l_entree,
)
from polytech.electronique.domain.facteur_de_bruit import (
    facteur_de_bruit,
    figure_de_bruit,
    temperature_de_bruit_equivalente,
)


def test_le_bruit_de_grenaille_est_proportionnel_au_courant() -> None:
    """`2 q I` : doubler le courant double la densite, et non son carre."""
    simple = densite_spectrale_de_grenaille(1e-3)
    double = densite_spectrale_de_grenaille(2e-3)
    assert double == pytest.approx(2 * simple, rel=1e-9, abs=0)
    assert simple == pytest.approx(2 * CHARGE_ELEMENTAIRE * 1e-3, rel=1e-9, abs=0)


def test_un_courant_nul_ne_produit_aucun_bruit_de_grenaille() -> None:
    """Sans porteur franchissant la barriere, il n'y a rien a compter."""
    assert densite_spectrale_de_grenaille(0.0) == 0.0


def test_un_courant_negatif_est_refuse() -> None:
    """Le sens du courant ne change pas son bruit : un signe est une faute."""
    with pytest.raises(ValueError, match="courant negatif"):
        densite_spectrale_de_grenaille(-1e-3)


def test_le_bruit_en_un_sur_f_decroit_comme_la_frequence() -> None:
    """D'ou son autre nom : bruit rose, par opposition au blanc."""
    basse = densite_spectrale_en_un_sur_f(1e-12, 1.0)
    haute = densite_spectrale_en_un_sur_f(1e-12, 100.0)
    assert basse == pytest.approx(100 * haute, rel=1e-9, abs=0)


def test_une_frequence_nulle_est_refusee_parce_que_la_densite_y_diverge() -> None:
    """Ce n'est pas un defaut du modele : c'est pourquoi on integre depuis f_basse."""
    with pytest.raises(ValueError, match="frequence nulle"):
        densite_spectrale_en_un_sur_f(1e-12, 0.0)


def test_le_blanc_croit_comme_la_bande_et_le_rose_comme_son_logarithme() -> None:
    """La distinction decide du dimensionnement d'une mesure.

    Elargir la bande vers le haut coute cher en blanc ; allonger la duree
    de mesure, donc abaisser `f_basse`, coute cher en rose.
    """
    blanc_seul_etroit = puissance_integree_blanc_et_rose(1e-16, 0.0, 1.0, 101.0)
    blanc_seul_large = puissance_integree_blanc_et_rose(1e-16, 0.0, 1.0, 201.0)
    assert blanc_seul_large == pytest.approx(2 * blanc_seul_etroit, rel=1e-9, abs=0)

    rose_seul = puissance_integree_blanc_et_rose(0.0, 1e-14, 1.0, 100.0)
    rose_decade_de_plus = puissance_integree_blanc_et_rose(0.0, 1e-14, 1.0, 10_000.0)
    assert rose_decade_de_plus == pytest.approx(2 * rose_seul, rel=1e-9, abs=0)


def test_une_bande_inversee_est_refusee() -> None:
    with pytest.raises(ValueError, match="bande invalide"):
        puissance_integree_blanc_et_rose(1e-16, 1e-14, 100.0, 1.0)


def test_une_bande_commencant_a_zero_est_refusee() -> None:
    with pytest.raises(ValueError, match="bande invalide"):
        puissance_integree_blanc_et_rose(1e-16, 1e-14, 0.0, 100.0)


def test_les_deux_sources_de_bruit_s_ajoutent_en_puissance() -> None:
    """Et non en amplitude : elles sont decorrelees.

    `racine(v^2 + (i R)^2)` et non `v + i R`. L'erreur surestime le bruit
    d'un facteur qui peut atteindre racine de deux.
    """
    tension, courant, resistance = 5e-9, 1e-12, 5e3
    attendu = math.hypot(tension, courant * resistance)
    obtenu = tension_efficace_ramenee_a_l_entree(tension, courant, resistance)
    assert obtenu == pytest.approx(attendu, rel=1e-9, abs=0)
    assert obtenu < tension + courant * resistance


def test_le_bruit_en_courant_pese_d_autant_plus_que_la_source_est_resistive() -> None:
    """C'est ce qui decide, sur une datasheet, lequel des deux domine."""
    faible = tension_efficace_ramenee_a_l_entree(5e-9, 1e-12, 100.0)
    forte = tension_efficace_ramenee_a_l_entree(5e-9, 1e-12, 1e6)
    assert forte > 100 * faible


def test_une_source_sans_resistance_ne_laisse_que_le_bruit_en_tension() -> None:
    assert tension_efficace_ramenee_a_l_entree(5e-9, 1e-12, 0.0) == pytest.approx(
        5e-9, rel=1e-9, abs=0
    )


def test_une_resistance_de_source_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="resistance de source negative"):
        tension_efficace_ramenee_a_l_entree(5e-9, 1e-12, -1.0)


def test_le_facteur_de_bruit_ne_descend_jamais_sous_l_unite() -> None:
    """La planche l'ecrit : « noise factor should always be >= 1 ».

    Un quadripole n'ameliore jamais un rapport signal sur bruit ; au mieux
    il le preserve. Un facteur calcule sous l'unite signale donc une faute
    de mesure ou de calcul, et c'est a ce moment-la qu'il faut le dire.
    """
    assert facteur_de_bruit(100.0, 50.0) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="facteur de bruit sous l'unite"):
        facteur_de_bruit(50.0, 100.0)


def test_un_rapport_signal_sur_bruit_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="rapport signal sur bruit"):
        facteur_de_bruit(100.0, 0.0)


def test_un_facteur_de_deux_fait_exactement_trois_decibels() -> None:
    """Point fixe exact : `10 log10 2 = 3,0103`.

    Dix et non vingt -- le facteur de bruit est un rapport de puissances.
    L'erreur double la figure, et elle est courante.
    """
    assert figure_de_bruit(2.0) == pytest.approx(3.0103, abs=0.0001)


def test_un_quadripole_parfait_a_une_figure_de_bruit_nulle() -> None:
    assert figure_de_bruit(1.0) == pytest.approx(0.0)


def test_une_figure_de_bruit_sous_l_unite_est_refusee() -> None:
    with pytest.raises(ValueError, match="facteur de bruit sous l'unite"):
        figure_de_bruit(0.5)


def test_un_facteur_de_deux_vaut_exactement_la_temperature_de_reference() -> None:
    """Second point fixe exact : `T_e = (F - 1) T_0` vaut `T_0` pour `F = 2`."""
    assert temperature_de_bruit_equivalente(2.0) == pytest.approx(290.0)


def test_un_quadripole_parfait_a_une_temperature_de_bruit_nulle() -> None:
    assert temperature_de_bruit_equivalente(1.0) == pytest.approx(0.0)


def test_une_temperature_de_bruit_sous_l_unite_est_refusee() -> None:
    with pytest.raises(ValueError, match="facteur de bruit sous l'unite"):
        temperature_de_bruit_equivalente(0.5)
