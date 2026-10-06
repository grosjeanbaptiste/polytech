"""Le budget de bruit d'une chaine d'acquisition, etage par etage.

Source : le cours d'electronique, chapitre 4 (bruit rapporte a l'entree,
chaines d'acquisition), et le corrige officiel du laboratoire 3
(`ElectroniqueInstrumentation/LABOS/Lab_03_Instrumentation_Solution.pdf`,
page 3), qui ecrit la regle : la puissance de bruit d'un etage, en sortie
de chaine, vaut sa densite au carre, fois la bande, fois le carre du gain
qui reste a traverser -- le sien compris.

L'oracle chiffre est l'exercice 1.1 de ce corrige : `5 10^-7`, `2 10^-6`,
`2 10^-8`, `2 10^-10` et `2 10^-8` volts carres.
"""

from __future__ import annotations

import math

import pytest

from polytech.electronique.domain.chaine_d_acquisition import (
    Etage,
    bruit_efficace,
    gain_total,
    puissances_a_l_entree,
    puissances_en_sortie,
)

CHAINE = (
    Etage("ampli A", 5.0, 1e-9),
    Etage("passe-haut", 1.0, 10e-9),
    Etage("ampli B", 1000.0, 1e-9),
    Etage("passe-bas", 1.0, 100e-9),
    Etage("CAN", 1.0, 1000e-9),
)


def test_le_gain_total_est_le_produit_des_gains() -> None:
    assert gain_total(CHAINE) == pytest.approx(5000.0)


def test_les_puissances_en_sortie_de_l_exercice_1_1() -> None:
    """Page 3 du corrige : chaque bruit traverse son etage et tous les suivants."""
    puissances = puissances_en_sortie(CHAINE, 20e3)
    assert puissances == pytest.approx([5e-7, 2e-6, 2e-8, 2e-10, 2e-8], rel=1e-9, abs=0)


def test_les_puissances_a_l_entree_sont_celles_de_sortie_divisees() -> None:
    """Rapporter a l'entree, c'est diviser par le carre du gain total."""
    sortie = puissances_en_sortie(CHAINE, 20e3)
    entree = puissances_a_l_entree(CHAINE, 20e3)
    assert entree == pytest.approx([p / 5000**2 for p in sortie], rel=1e-9, abs=0)
    assert entree[0] == pytest.approx((1e-9) ** 2 * 20e3, rel=1e-9, abs=0)
    assert entree[1] == pytest.approx((10e-9) ** 2 * 20e3 / 5**2, rel=1e-9, abs=0)


def test_le_bruit_efficace_est_la_racine_de_la_somme() -> None:
    """Des bruits independants s'ajoutent en puissance, non en tension."""
    assert bruit_efficace([9.0, 16.0]) == pytest.approx(5.0)
    total = bruit_efficace(puissances_en_sortie(CHAINE, 20e3))
    assert total == pytest.approx(math.sqrt(2.5402e-6))


def test_un_etage_ou_une_chaine_sans_sens_sont_refuses() -> None:
    with pytest.raises(ValueError, match="gain nul ou negatif"):
        Etage("x", 0.0, 1e-9)
    with pytest.raises(ValueError, match="densite negative"):
        Etage("x", 1.0, -1e-9)
    with pytest.raises(ValueError, match="chaine vide"):
        gain_total(())
    with pytest.raises(ValueError, match="chaine vide"):
        puissances_en_sortie((), 1.0)
    with pytest.raises(ValueError, match="bande nulle ou negative"):
        puissances_en_sortie(CHAINE, 0.0)
    with pytest.raises(ValueError, match="aucune puissance"):
        bruit_efficace([])
    with pytest.raises(ValueError, match="puissance negative"):
        bruit_efficace([1.0, -1.0])
    assert Etage("fil", 1.0, 0.0).densite == 0.0
