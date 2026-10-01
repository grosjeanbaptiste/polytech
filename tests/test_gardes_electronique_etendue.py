"""Les gardes des modules ajoutes au contexte de l'electronique et des signaux.

L'audit du depot exige qu'aucun `raise` ne reste jamais execute. Chacune de
ces gardes protege d'une facon precise de rendre un resultat faux sans
qu'il paraisse l'etre -- une distorsion negative, une alimentation
inversee, une constante de temps nulle qui mettrait la coupure a l'infini.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.boucle_fermee import (
    distorsion_apres_retroaction,
    impedance_d_entree_en_boucle_fermee,
    impedance_de_sortie_en_boucle_fermee,
)
from polytech.electronique.domain.defauts_aop import (
    capacite_de_miller,
    erreur_de_courant_de_decalage,
    erreur_de_courant_de_polarisation,
    erreur_de_tension_de_decalage,
    plage_d_entree_en_mode_commun,
    resistance_de_compensation,
)
from polytech.electronique.domain.gabarit import Gabarit
from polytech.electronique.domain.montages import (
    convertisseur_courant_tension,
    integrateur,
    soustracteur,
)
from polytech.signaux.domain.harmonique import (
    courbe_elementaire_de_premier_ordre,
    courbe_elementaire_de_second_ordre,
)
from polytech.signaux.domain.reponse_indicielle import classe_d_amortissement
from polytech.signaux.domain.stabilite import est_stable_par_les_racines
from polytech.signaux.domain.transformees import puissance_moyenne_d_un_signal
from polytech.signaux.domain.transmittance import Transmittance

P = sp.Symbol("p")


def test_une_distorsion_negative_est_refusee() -> None:
    """Une distorsion est une fraction positive : un signe est une faute."""
    with pytest.raises(ValueError, match="distorsion negative"):
        distorsion_apres_retroaction(-0.1, 1e5, 0.01)


def test_une_impedance_d_entree_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="impedance d'entree"):
        impedance_d_entree_en_boucle_fermee(0.0, 1e5, 0.01)


def test_une_impedance_de_sortie_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="impedance de sortie"):
        impedance_de_sortie_en_boucle_fermee(0.0, 1e5, 0.01)


def test_des_alimentations_inversees_sont_refusees() -> None:
    """Elles decriraient une plage vide, et la marge serait comptee a l'envers."""
    with pytest.raises(ValueError, match="alimentations inversees"):
        plage_d_entree_en_mode_commun(15.0, -15.0, marge=1.5)


def test_une_resistance_negative_dans_la_polarisation_est_refusee() -> None:
    with pytest.raises(ValueError, match="resistance negative"):
        erreur_de_courant_de_polarisation(20e-9, -1.0)


def test_une_resistance_negative_dans_le_decalage_est_refusee() -> None:
    with pytest.raises(ValueError, match="resistance negative"):
        erreur_de_courant_de_decalage(2e-9, -1.0)


def test_un_gain_negatif_dans_la_tension_de_decalage_est_refuse() -> None:
    with pytest.raises(ValueError, match="gain negatif"):
        erreur_de_tension_de_decalage(1e-3, -100.0)


def test_une_capacite_negative_est_refusee_dans_miller() -> None:
    with pytest.raises(ValueError, match="capacite negative"):
        capacite_de_miller(-5e-12, 100.0)


def test_une_resistance_nulle_dans_la_compensation_est_refusee() -> None:
    with pytest.raises(ValueError, match="resistance nulle ou negative"):
        resistance_de_compensation(0.0, 99e3)


def test_une_resistance_nulle_dans_le_pont_du_soustracteur_est_refusee() -> None:
    """Le pont ne diviserait plus rien : le mode commun passerait entier."""
    with pytest.raises(ValueError, match="resistance nulle ou negative dans le pont"):
        soustracteur(0.0, 10e3, 1e3, 10e3)


def test_une_transimpedance_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="transimpedance"):
        convertisseur_courant_tension(0.0)


def test_une_capacite_nulle_dans_l_integrateur_est_refusee() -> None:
    with pytest.raises(ValueError, match="capacite nulle ou negative"):
        integrateur(1e3, 0.0, P)


def test_un_affaiblissement_hors_bande_trop_faible_est_nomme() -> None:
    """La garde porte le mot « trop faible » pour que l'audit la voie."""
    with pytest.raises(ValueError, match="affaiblissement hors bande trop faible"):
        Gabarit(
            affaiblissement_en_bande=40.0,
            affaiblissement_hors_bande=3.0,
            pulsation_passante=1.0,
            pulsation_bloquante=2.0,
        )


def test_une_pulsation_passante_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="pulsation passante"):
        Gabarit(
            affaiblissement_en_bande=3.0,
            affaiblissement_hors_bande=40.0,
            pulsation_passante=0.0,
            pulsation_bloquante=2.0,
        )


def test_une_constante_de_temps_nulle_met_la_coupure_a_l_infini() -> None:
    with pytest.raises(ValueError, match="constante de temps nulle"):
        courbe_elementaire_de_premier_ordre(sp.Integer(0))


def test_une_constante_de_temps_nulle_met_la_pulsation_propre_a_l_infini() -> None:
    with pytest.raises(ValueError, match="constante de temps nulle"):
        courbe_elementaire_de_second_ordre(sp.Integer(0), sp.Rational(1, 2))


def test_un_amortissement_nul_n_a_pas_de_classe() -> None:
    with pytest.raises(ValueError, match="amortissement nul ou negatif"):
        classe_d_amortissement(0.0)


def test_une_periode_nulle_dans_la_puissance_moyenne_est_refusee() -> None:
    with pytest.raises(ValueError, match="periode nulle"):
        puissance_moyenne_d_un_signal(sp.Integer(1), sp.Integer(0))


def test_une_equation_caracteristique_sans_racine_est_nommee() -> None:
    """Une transmittance constante n'en a aucune : la stabilite ne se decide pas."""
    with pytest.raises(ValueError, match="equation caracteristique sans racine"):
        est_stable_par_les_racines(Transmittance(expression=sp.Integer(1)))
