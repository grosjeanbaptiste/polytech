"""Les fascicules 6, 7, 8 et le plan du cours, repris en profondeur.

Ce que les planches enseignent et que les modules ne portaient pas.

**Stabilite** : la condition necessaire et suffisante par les racines de
l'equation caracteristique, les deux marges, et surtout l'**influence
d'un retard** -- auquel le fascicule consacre plusieurs planches, parce
qu'un retard ne change pas le module mais mange la phase, donc consomme
la marge sans qu'on le voie sur un diagramme de gain.

**Performances** : les erreurs de regime, de consigne et de perturbation,
la lecture de `|M(0)|`, du pic de resonance et de la bande passante, et la
classification en amorti, moderement amorti, faiblement amorti que le
fascicule illustre en trois planches.

**Analyse harmonique** : les courbes elementaires dont tout diagramme se
compose, et l'influence du gain -- qui translate le module sans toucher a
l'argument.

**Signaux** : le produit scalaire, le rapport signal sur bruit et la
quantification. Le point fixe exact y est la regle des six decibels par
bit, `SNR = 6,02 n + 1,76` dB, que le fascicule emploie sans la demontrer.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.signaux.domain.elementaires import (
    Signal,
    pas_de_quantification,
    produit_scalaire,
    rapport_signal_sur_bruit_d_un_quantificateur,
    rapport_signal_sur_bruit_en_decibels,
)
from polytech.signaux.domain.harmonique import (
    courbe_elementaire_d_integrateur,
    courbe_elementaire_de_gain,
    courbe_elementaire_de_premier_ordre,
    courbe_elementaire_de_second_ordre,
    influence_du_gain,
)
from polytech.signaux.domain.performances import (
    bande_passante_du_second_ordre,
    classe_d_amortissement,
    erreur_de_perturbation,
    erreur_statique_de_consigne,
    module_en_continu,
)
from polytech.signaux.domain.stabilite import (
    Stabilite,
    equation_caracteristique,
    est_stable_par_les_racines,
    marge_de_phase_avec_retard,
    phase_ajoutee_par_un_retard,
    retard_critique,
)
from polytech.signaux.domain.symboles import GAIN, IMAGE, PULSATION, TEMPS
from polytech.signaux.domain.transmittance import Transmittance

P = IMAGE
OMEGA = PULSATION
PREMIER = Transmittance(expression=1 / (1 + P))


def test_l_equation_caracteristique_est_un_plus_la_boucle_ouverte() -> None:
    """`1 + G(p) = 0` : ses racines sont les poles de la boucle fermee."""
    equation = equation_caracteristique(PREMIER)
    assert sp.simplify(equation - (2 + P)) == 0


def test_les_racines_de_l_equation_caracteristique_decident_de_la_stabilite() -> None:
    """Condition necessaire **et** suffisante : toutes a partie reelle negative."""
    assert est_stable_par_les_racines(PREMIER)
    assert not est_stable_par_les_racines(Transmittance(expression=1 / (P - 2)))


def test_une_boucle_peut_stabiliser_un_systeme_instable() -> None:
    """`G = 2/(p-1)` est instable en ouvert et **stable** en ferme.

    Ce test est ne d'une erreur : je l'avais ecrit en attendant
    l'instabilite. `1 + G = (p+1)/(p-1)`, dont la racine est `-1`. C'est
    la raison d'etre du bouclage, et c'est ce que l'equation
    caracteristique montre sans qu'on ait a le supposer -- le pole de la
    boucle ouverte n'est pas un pole de la boucle fermee.
    """
    instable_en_ouvert = Transmittance(expression=2 / (P - 1))
    assert instable_en_ouvert.poles() == (sp.Integer(1),)
    assert est_stable_par_les_racines(instable_en_ouvert)


def test_un_retard_mange_la_phase_sans_toucher_au_module() -> None:
    """`e^{-j omega tau}` : module un, argument `-omega tau`.

    C'est tout le danger d'un retard : il ne se voit pas sur un diagramme
    de gain, et il consomme la marge de phase proportionnellement a la
    pulsation.
    """
    duree = sp.Rational(1, 10)
    phase = phase_ajoutee_par_un_retard(duree, OMEGA)
    assert sp.simplify(phase + duree * OMEGA) == 0


def test_la_marge_de_phase_diminue_avec_le_retard() -> None:
    """Et c'est une soustraction directe, en radians."""
    sans = marge_de_phase_avec_retard(sp.pi / 3, sp.Integer(0), sp.Integer(1))
    avec = marge_de_phase_avec_retard(sp.pi / 3, sp.Rational(1, 10), sp.Integer(1))
    assert sp.simplify(sans - sp.pi / 3) == 0
    assert avec < sans


def test_le_retard_critique_annule_exactement_la_marge_de_phase() -> None:
    """`tau_c = marge / omega` : au-dela, la boucle devient instable.

    Verification croisee : appliquer ce retard doit rendre une marge nulle.
    """
    marge, pulsation = sp.pi / 3, sp.Integer(2)
    critique = retard_critique(marge, pulsation)
    assert sp.simplify(critique - sp.pi / 6) == 0
    restante = marge_de_phase_avec_retard(marge, critique, pulsation)
    assert sp.simplify(restante) == 0


def test_un_retard_critique_sur_une_pulsation_nulle_est_refuse() -> None:
    with pytest.raises(ValueError, match="pulsation nulle"):
        retard_critique(sp.pi / 3, sp.Integer(0))


def test_la_marge_de_gain_se_lit_au_croisement_de_l_axe_reel() -> None:
    """Recoupement avec le module existant, sur l'exemple du cours."""
    ouverte = Transmittance(expression=10 * GAIN / (P * (1 + P) * (1 + 2 * P)))
    stabilite = Stabilite(boucle_ouverte=ouverte)
    reel = stabilite.reel_au_croisement({GAIN: sp.Integer(1)})
    assert float(abs(reel)) == pytest.approx(6.667, abs=0.001)


def test_l_erreur_statique_de_consigne_decroit_comme_le_gain() -> None:
    """`1 / (1 + K)` : doubler le gain divise presque par deux l'erreur."""
    assert erreur_statique_de_consigne(9.0) == pytest.approx(0.1)
    assert erreur_statique_de_consigne(99.0) == pytest.approx(0.01)


def test_un_integrateur_dans_la_boucle_annule_l_erreur_statique() -> None:
    """C'est le resultat que le fascicule met en avant, et il est exact."""
    assert erreur_statique_de_consigne(math.inf) == pytest.approx(0.0)


def test_un_gain_negatif_est_refuse_dans_l_erreur_statique() -> None:
    with pytest.raises(ValueError, match="gain de boucle negatif"):
        erreur_statique_de_consigne(-2.0)


def test_l_erreur_de_perturbation_depend_du_gain_amont_seul() -> None:
    """`1 / (1 + K)` fois le gain **aval** de la perturbation.

    Le fascicule insiste : une perturbation qui entre en aval du
    correcteur n'est pas rejetee comme une erreur de consigne.
    """
    assert erreur_de_perturbation(amplitude=1.0, gain_aval=2.0, gain_de_boucle=9.0) == (
        pytest.approx(0.2)
    )


def test_le_module_en_continu_tend_vers_un_quand_le_gain_croit() -> None:
    """`K / (1 + K)` : la boucle fermee suit sa consigne d'autant mieux."""
    assert module_en_continu(9.0) == pytest.approx(0.9)
    assert module_en_continu(999.0) == pytest.approx(0.999)


def test_la_bande_passante_du_second_ordre_croit_avec_la_pulsation_propre() -> None:
    """A amortissement fixe, elle est proportionnelle a `omega_0`."""
    etroite = bande_passante_du_second_ordre(1.0, 0.7)
    large = bande_passante_du_second_ordre(2.0, 0.7)
    assert large == pytest.approx(2 * etroite)


def test_un_amortissement_nul_est_refuse_dans_la_bande_passante() -> None:
    with pytest.raises(ValueError, match="amortissement"):
        bande_passante_du_second_ordre(1.0, 0.0)


@pytest.mark.parametrize(
    ("amortissement", "attendu"),
    [(0.1, "faiblement amorti"), (0.5, "moderement amorti"), (1.5, "amorti")],
)
def test_les_trois_classes_d_amortissement_du_fascicule(
    amortissement: float,
    attendu: str,
) -> None:
    """Trois planches d'illustration, trois classes, et les seuils sont dits."""
    assert classe_d_amortissement(amortissement) == attendu


def test_la_courbe_elementaire_d_un_gain_est_plate() -> None:
    """Module constant, argument nul : c'est la brique la plus simple."""
    module, phase = courbe_elementaire_de_gain(sp.Integer(10))
    assert sp.simplify(module - 10) == 0
    assert sp.simplify(phase) == 0


def test_la_courbe_elementaire_d_un_integrateur_chute_de_vingt_db_par_decade() -> None:
    """`1 / (j omega)` : module en `1/omega`, argument constant a -90 degres."""
    module, phase = courbe_elementaire_d_integrateur()
    assert sp.simplify(module - 1 / OMEGA) == 0
    assert sp.simplify(phase + sp.pi / 2) == 0


def test_la_courbe_elementaire_du_premier_ordre_dephase_de_moins_quarante_cinq() -> (
    None
):
    """A la pulsation de coupure, exactement : c'est le point fixe du trace."""
    constante = sp.Integer(1)
    _, phase = courbe_elementaire_de_premier_ordre(constante)
    assert sp.simplify(phase.subs(OMEGA, 1) + sp.pi / 4) == 0


def test_la_courbe_elementaire_du_second_ordre_dephase_de_moins_quatre_vingt_dix() -> (
    None
):
    """A la pulsation propre, quel que soit l'amortissement."""
    _, phase = courbe_elementaire_de_second_ordre(sp.Integer(1), sp.Rational(1, 2))
    assert sp.simplify(phase.subs(OMEGA, 1) + sp.pi / 2) == 0


def test_le_gain_translate_le_module_sans_toucher_a_l_argument() -> None:
    """« |.|, Re et Im multiplies par K ; argument inchange », dit la planche."""
    module, phase = courbe_elementaire_de_premier_ordre(sp.Integer(1))
    module_amplifie, phase_amplifiee = influence_du_gain(
        (module, phase),
        sp.Integer(100),
    )
    assert sp.simplify(module_amplifie - 100 * module) == 0
    assert sp.simplify(phase_amplifiee - phase) == 0


def test_le_produit_scalaire_de_deux_signaux_orthogonaux_est_nul() -> None:
    """Sinus et cosinus sur une periode : le resultat exact du fascicule."""
    periode = 2 * sp.pi
    produit = produit_scalaire(sp.sin(TEMPS), sp.cos(TEMPS), periode)
    assert sp.simplify(produit) == 0


def test_le_produit_scalaire_d_un_signal_avec_lui_meme_est_son_energie() -> None:
    periode = 2 * sp.pi
    propre = produit_scalaire(sp.sin(TEMPS), sp.sin(TEMPS), periode)
    assert sp.simplify(propre - sp.pi) == 0


def test_un_produit_scalaire_sur_une_duree_nulle_est_refuse() -> None:
    with pytest.raises(ValueError, match="duree nulle"):
        produit_scalaire(sp.sin(TEMPS), sp.cos(TEMPS), sp.Integer(0))


def test_le_rapport_signal_sur_bruit_en_decibels() -> None:
    """Dix log : c'est un rapport de **puissances**."""
    assert rapport_signal_sur_bruit_en_decibels(100.0, 1.0) == pytest.approx(20.0)


def test_un_bruit_nul_donne_un_rapport_infini_et_est_refuse() -> None:
    with pytest.raises(ValueError, match="puissance de bruit"):
        rapport_signal_sur_bruit_en_decibels(100.0, 0.0)


def test_le_pas_de_quantification_se_divise_par_deux_a_chaque_bit() -> None:
    assert pas_de_quantification(10.0, 8) == pytest.approx(10.0 / 256)
    assert pas_de_quantification(10.0, 9) == pytest.approx(10.0 / 512)


def test_un_nombre_de_bits_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="nombre de bits"):
        pas_de_quantification(10.0, 0)


def test_la_regle_des_six_decibels_par_bit() -> None:
    """`SNR = 6,02 n + 1,76` dB : point fixe exact du chapitre.

    Douze bits donnent 74 dB, et c'est le nombre qu'un dimensionnement de
    chaine d'acquisition emploie directement.
    """
    assert rapport_signal_sur_bruit_d_un_quantificateur(12) == pytest.approx(
        74.0,
        abs=0.05,
    )
    un_bit_de_plus = rapport_signal_sur_bruit_d_un_quantificateur(13)
    assert un_bit_de_plus - rapport_signal_sur_bruit_d_un_quantificateur(12) == (
        pytest.approx(6.02, abs=0.01)
    )


def test_un_signal_garde_sa_classe_apres_les_ajouts() -> None:
    """Non-regression : les ajouts ne doivent rien changer a l'existant."""
    assert Signal(expression=sp.exp(-TEMPS)).energie_verifiee() == pytest.approx(0.5)
