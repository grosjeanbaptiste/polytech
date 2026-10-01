"""Le chapitre 2 partie II : les montages, et ce que l'AOP reel leur fait.

Deux moities. Les **montages** d'abord -- additionneur, soustracteur,
convertisseur courant-tension, derivateur, integrateur -- dont le
fascicule donne le transfert pour chacun. Puis les **defauts** qui les
limitent : taux de rejection du mode commun, plage d'entree en mode
commun, courant de polarisation, courant et tension de decalage, et la
capacite de Miller.

L'ancrage le plus net est celui de Miller, parce qu'il surprend : la
capacite vue a l'entree d'un inverseur de gain `-G` vaut `C (1 + G)`, donc
un gain de cent multiplie par cent une capacite parasite de quelques
picofarads. C'est ce qui limite la bande d'un montage bien avant l'AOP
lui-meme, et c'est pourquoi le fascicule lui consacre une planche.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.defauts_aop import (
    capacite_de_miller,
    erreur_de_courant_de_decalage,
    erreur_de_courant_de_polarisation,
    erreur_de_tension_de_decalage,
    plage_d_entree_en_mode_commun,
    rejection_du_mode_commun,
    rejection_en_decibels,
    resistance_de_compensation,
)
from polytech.electronique.domain.montages import (
    convertisseur_courant_tension,
    derivateur,
    integrateur,
    somme_ponderee,
    soustracteur,
)

P = sp.Symbol("p")


def test_l_additionneur_pondere_chaque_entree_par_sa_resistance() -> None:
    """`-R_f (V_1/R_1 + V_2/R_2)` : chaque voie a son propre poids."""
    poids = somme_ponderee(contre_reaction=10e3, entrees=(1e3, 2e3))
    assert poids == pytest.approx((-10.0, -5.0))


def test_un_additionneur_sans_entree_est_refuse() -> None:
    with pytest.raises(ValueError, match="aucune entree"):
        somme_ponderee(contre_reaction=10e3, entrees=())


def test_une_resistance_d_entree_nulle_est_refusee() -> None:
    """Elle donnerait un poids infini : un court-circuit sur la sortie."""
    with pytest.raises(ValueError, match="resistance d'entree"):
        somme_ponderee(contre_reaction=10e3, entrees=(0.0,))


def test_le_soustracteur_rejette_le_mode_commun_quand_les_ponts_sont_egaux() -> None:
    """C'est la condition d'equilibre, et elle est exacte.

    `R_1/R_2 = R_3/R_4` annule le gain de mode commun. Un desequilibre de
    un pour cent le fait remonter, et c'est ce qui limite le CMRR reel d'un
    amplificateur differentiel a base d'AOP.
    """
    equilibre, commun = soustracteur(1e3, 10e3, 1e3, 10e3)
    assert equilibre == pytest.approx(10.0)
    assert commun == pytest.approx(0.0, abs=1e-12)


def test_un_soustracteur_desequilibre_laisse_passer_du_mode_commun() -> None:
    _, commun = soustracteur(1e3, 10e3, 1.01e3, 10e3)
    assert abs(commun) > 1e-3


def test_le_convertisseur_courant_tension_rend_moins_r_i() -> None:
    """Un seul parametre : la resistance de transimpedance."""
    assert convertisseur_courant_tension(1e6) == pytest.approx(-1e6)


def test_le_derivateur_a_un_zero_a_l_origine() -> None:
    """`-R C p` : il amplifie les hautes frequences, donc le bruit.

    C'est pourquoi le fascicule le montre toujours avec une resistance en
    serie -- le derivateur pur est inutilisable.
    """
    transfert = derivateur(1e3, 1e-6, P)
    assert sp.simplify(transfert.subs(P, 0)) == 0


def test_l_integrateur_a_un_pole_a_l_origine() -> None:
    """`-1 / (R C p)` : il diverge en continu, d'ou la resistance de fuite."""
    transfert = integrateur(1e3, 1e-6, P)
    assert sp.limit(sp.Abs(transfert), P, 0) == sp.oo


def test_le_derivateur_et_l_integrateur_sont_inverses_au_signe_pres() -> None:
    """Recoupement : leur produit vaut l'unite, au signe des deux inversions."""
    produit = sp.simplify(derivateur(1e3, 1e-6, P) * integrateur(1e3, 1e-6, P))
    assert sp.simplify(produit - 1) == 0


@pytest.mark.parametrize(
    ("fonction", "message"),
    [
        (derivateur, "resistance nulle"),
        (integrateur, "resistance nulle"),
    ],
)
def test_un_derivateur_ou_integrateur_sans_resistance_est_refuse(
    fonction: object,
    message: str,
) -> None:
    assert callable(fonction)
    with pytest.raises(ValueError, match=message):
        fonction(0.0, 1e-6, P)


def test_le_taux_de_rejection_est_le_rapport_des_deux_gains() -> None:
    """`CMRR = |A_d / A_c|` : differentiel sur mode commun."""
    assert rejection_du_mode_commun(1e3, 0.01) == pytest.approx(1e5)


def test_un_gain_de_mode_commun_nul_donne_une_rejection_infinie() -> None:
    """Et le dire vaut mieux que de rendre `inf` que le document composerait."""
    with pytest.raises(ValueError, match="gain de mode commun nul"):
        rejection_du_mode_commun(1e3, 0.0)


def test_cent_mille_de_rejection_font_cent_decibels() -> None:
    """Point fixe exact : `20 log10(10^5) = 100`.

    Vingt et non dix : le CMRR est un rapport de **tensions**.
    """
    assert rejection_en_decibels(1e5) == pytest.approx(100.0)


def test_la_plage_d_entree_en_mode_commun_est_bornee_par_les_alimentations() -> None:
    """Elle ne les atteint jamais : la marge est une donnee du composant."""
    basse, haute = plage_d_entree_en_mode_commun(-15.0, 15.0, marge=1.5)
    assert basse == pytest.approx(-13.5)
    assert haute == pytest.approx(13.5)


def test_une_marge_plus_large_que_l_alimentation_est_refusee() -> None:
    with pytest.raises(ValueError, match="marge"):
        plage_d_entree_en_mode_commun(-1.0, 1.0, marge=5.0)


def test_le_courant_de_polarisation_cree_une_tension_d_erreur() -> None:
    """`V_err = I_b R` : d'ou l'interet de resistances faibles."""
    assert erreur_de_courant_de_polarisation(20e-9, 1e6) == pytest.approx(20e-3)


def test_la_resistance_de_compensation_egalise_les_deux_entrees() -> None:
    """`R_c = R_1 // R_f` : c'est ce qui annule l'effet du courant de polarisation.

    Il ne reste alors que le courant de **decalage**, qui est bien plus
    petit -- c'est tout l'interet de la compensation.
    """
    compensation = resistance_de_compensation(1e3, 99e3)
    assert compensation == pytest.approx(1e3 * 99e3 / 100e3)


def test_le_courant_de_decalage_reste_apres_compensation() -> None:
    """Et il est d'un ordre de grandeur plus faible que la polarisation."""
    residu = erreur_de_courant_de_decalage(2e-9, 1e6)
    assert residu == pytest.approx(2e-3)
    assert residu < erreur_de_courant_de_polarisation(20e-9, 1e6)


def test_la_tension_de_decalage_est_amplifiee_comme_un_signal() -> None:
    """`V_err = V_os (1 + R_f/R_1)` : elle subit le gain du montage.

    C'est ce qui la rend genante : un millivolt de decalage devient cent
    millivolts a la sortie d'un montage de gain cent.
    """
    assert erreur_de_tension_de_decalage(1e-3, 100.0) == pytest.approx(0.1)


def test_la_capacite_de_miller_multiplie_la_capacite_parasite() -> None:
    """`C (1 + G)` : un gain de cent multiplie par cent une capacite de 5 pF.

    C'est ce qui limite la bande d'un montage bien avant l'AOP lui-meme.
    """
    assert capacite_de_miller(5e-12, 100.0) == pytest.approx(5e-12 * 101)


def test_un_gain_nul_laisse_la_capacite_inchangee() -> None:
    """Sans amplification il n'y a pas d'effet Miller."""
    assert capacite_de_miller(5e-12, 0.0) == pytest.approx(5e-12)


def test_un_gain_negatif_est_refuse_dans_miller() -> None:
    """La formule attend le **module** du gain d'un inverseur."""
    with pytest.raises(ValueError, match="module du gain"):
        capacite_de_miller(5e-12, -100.0)


def test_la_rejection_en_decibels_refuse_un_taux_nul() -> None:
    with pytest.raises(ValueError, match="taux de rejection"):
        rejection_en_decibels(0.0)


def test_le_logarithme_de_la_rejection_est_bien_en_base_dix() -> None:
    """Recoupement elementaire mais utile : une decade fait vingt decibels."""
    assert rejection_en_decibels(10.0) == pytest.approx(20 * math.log10(10.0))
