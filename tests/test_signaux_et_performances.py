"""Signaux, systemes, reponses et performances, ancres sur le cours.

Deux ancrages exacts viennent de `Exs_08_Performances_Reponses`, feuille
transcrite a la main dans le depot depuis un scan manuscrit :

* `|M(0)| = -3 dB` et `y(infini) = 0,707` -- et `10^(-3/20)` vaut
  bien `0,70795` ;
* `D = 87 %` et `y_max = 1,32` -- et `0,707 x 1,87` vaut `1,3221`.

Les deux se recoupent a la troisieme decimale sans qu'aucune tolerance
ait ete choisie pour les faire passer.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.signaux.domain.controleur import ProportionnelIntegral
from polytech.signaux.domain.elementaires import Classe, Signal, echelon, rampe
from polytech.signaux.domain.harmonique import AnalyseHarmonique, en_decibels
from polytech.signaux.domain.performances import (
    gain_statique_depuis_decibels,
)
from polytech.signaux.domain.reponse_indicielle import (
    amortissement_du_second_ordre,
    depassement_du_second_ordre,
    en_pourcent,
    maximum_de_la_reponse,
)
from polytech.signaux.domain.symboles import IMAGE, TEMPS
from polytech.signaux.domain.systemes import Systeme, convolution
from polytech.signaux.domain.temporel import ReponsesTemporelles
from polytech.signaux.domain.transformees import (
    derivation,
    fourier_depuis_laplace,
    retard,
    transformee_de_laplace,
)
from polytech.signaux.domain.transmittance import Transmittance

P = IMAGE
PREMIER_ORDRE = Transmittance(expression=1 / (1 + P))


def test_une_exponentielle_decroissante_est_d_energie_finie() -> None:
    """La classe decide de ce que l'on peut calculer : energie ou puissance."""
    signal = Signal(expression=sp.exp(-TEMPS))
    assert signal.classe() is Classe.ENERGIE_FINIE
    assert sp.simplify(signal.energie_verifiee() - sp.Rational(1, 2)) == 0


def test_l_echelon_est_de_puissance_finie_et_non_d_energie_finie() -> None:
    """Son energie diverge : la confondre ferait composer un infini."""
    signal = Signal(expression=echelon())
    assert signal.classe() is Classe.PUISSANCE_FINIE
    with pytest.raises(ValueError, match="energie infinie"):
        signal.energie_verifiee()


def test_la_rampe_n_est_ni_l_un_ni_l_autre() -> None:
    """Son energie et sa puissance divergent toutes deux."""
    assert Signal(expression=rampe()).classe() is Classe.NI_L_UN_NI_L_AUTRE


def test_un_systeme_a_gain_constant_est_lineaire_et_invariant() -> None:
    systeme = Systeme(operateur=lambda entree: 3 * entree)
    assert systeme.est_lineaire()
    assert systeme.est_invariant()


def test_un_systeme_quadratique_n_est_pas_lineaire() -> None:
    """Le dire avant de calculer : il sort du cadre du cours."""
    assert not Systeme(operateur=lambda entree: entree**2).est_lineaire()


def test_la_convolution_de_deux_echelons_donne_la_rampe() -> None:
    """Resultat d'ecole, et controle de la borne superieure `t`."""
    produit = convolution(sp.Integer(1), sp.Integer(1))
    assert sp.simplify(produit - TEMPS) == 0


def test_la_reponse_impulsionnelle_du_premier_ordre() -> None:
    """`1/(1+p)` a pour original `e^{-t}`."""
    reponses = ReponsesTemporelles(transmittance=PREMIER_ORDRE)
    assert sp.simplify(reponses.impulsionnelle() - sp.exp(-TEMPS)) == 0


def test_la_reponse_indicielle_du_premier_ordre() -> None:
    """`1 - e^{-t}` : c'est `T(p)/p` qu'il faut inverser, pas `T(p)`."""
    reponses = ReponsesTemporelles(transmittance=PREMIER_ORDRE)
    assert sp.simplify(reponses.indicielle() - (1 - sp.exp(-TEMPS))) == 0


def test_la_valeur_finale_se_lit_sans_inverser() -> None:
    """`T(0)` suffit, et la reponse indicielle doit y tendre."""
    reponses = ReponsesTemporelles(transmittance=PREMIER_ORDRE)
    assert sp.simplify(reponses.valeur_finale() - 1) == 0
    assert sp.limit(reponses.indicielle(), TEMPS, sp.oo) == 1


def test_un_systeme_sans_regime_final_est_refuse() -> None:
    """Un pole a l'origine : `T(0)` serait un nombre sans signification."""
    integrateur = ReponsesTemporelles(transmittance=Transmittance(expression=1 / P))
    with pytest.raises(ValueError, match="pas de regime final"):
        integrateur.valeur_finale()


def test_le_retard_devient_un_facteur_exponentiel() -> None:
    """La propriete que les exercices emploient le plus."""
    duree = sp.Symbol("tau", positive=True)
    image = transformee_de_laplace(sp.exp(-TEMPS))
    assert sp.simplify(retard(image, duree) - sp.exp(-P * duree) / (P + 1)) == 0


def test_la_derivation_fait_entrer_la_condition_initiale() -> None:
    initiale = sp.Symbol("x_0")
    image = sp.Symbol("X")
    assert sp.simplify(derivation(image, initiale) - (P * image - initiale)) == 0


def test_fourier_coincide_avec_laplace_sur_un_signal_causal() -> None:
    """Et seulement la : les deux integrales n'ont pas les memes bornes."""
    image = transformee_de_laplace(sp.exp(-TEMPS))
    pulsation = sp.Symbol("omega", positive=True)
    attendu = 1 / (1 + sp.I * pulsation)
    assert sp.simplify(fourier_depuis_laplace(image) - attendu) == 0


def test_les_trois_lectures_portent_les_memes_nombres() -> None:
    """Bode, Nyquist et Black ne sont que trois lectures du meme G(j omega)."""
    analyse = AnalyseHarmonique(transmittance=PREMIER_ORDRE)
    gain, phase = analyse.point_de_bode(sp.Integer(1))
    reel, imaginaire = analyse.point_de_nyquist(sp.Integer(1))
    phase_black, gain_black = analyse.point_de_black(sp.Integer(1))
    assert (phase_black, gain_black) == (phase, gain)
    module = sp.sqrt(reel**2 + imaginaire**2)
    assert sp.simplify(en_decibels(module) - gain) == 0


def test_le_gain_statique_du_corrige_donne_bien_sa_valeur_finale() -> None:
    """Feuille de reponses : `|M(0)| = -3 dB` et `y(infini) = 0,707`."""
    valeur = gain_statique_depuis_decibels(sp.Integer(-3))
    assert float(valeur) == pytest.approx(0.707, abs=0.001)


def test_le_maximum_du_corrige_se_retrouve_a_partir_du_depassement() -> None:
    """Feuille de reponses : `D = 87 %` et `y_max = 1,32`."""
    valeur_finale = gain_statique_depuis_decibels(sp.Integer(-3))
    maximum = maximum_de_la_reponse(valeur_finale, sp.Rational(87, 100))
    assert float(maximum) == pytest.approx(1.32, abs=0.005)


def test_le_depassement_et_l_amortissement_sont_reciproques() -> None:
    """Aller et revenir doit redonner le point de depart, sans table."""
    amortissement = sp.Rational(1, 5)
    depassement = depassement_du_second_ordre(amortissement)
    retour = amortissement_du_second_ordre(depassement)
    assert float(retour) == pytest.approx(float(amortissement), rel=1e-9)


def test_un_amortissement_hors_de_l_intervalle_est_refuse() -> None:
    """Au-dela de un il n'y a plus d'oscillation, donc plus de depassement."""
    with pytest.raises(ValueError, match="hors de"):
        depassement_du_second_ordre(sp.Integer(2))


def test_le_depassement_s_exprime_en_pourcent_comme_la_feuille_l_ecrit() -> None:
    assert sp.simplify(en_pourcent(sp.Rational(87, 100)) - 87) == 0


def test_un_correcteur_integral_annule_l_erreur_statique() -> None:
    """La verification intrinseque de la synthese, au lieu de l'esperer."""
    correcteur = ProportionnelIntegral(gain=sp.Integer(2), temps_integral=sp.Integer(1))
    assert sp.simplify(correcteur.erreur_statique(PREMIER_ORDRE)) == 0


def test_un_temps_integral_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="temps integral nul"):
        ProportionnelIntegral(gain=sp.Integer(1), temps_integral=sp.Integer(0))


def test_un_echelon_retarde_n_est_pas_reduit_sur_le_domaine() -> None:
    """La reduction ne doit pas effacer l'instant de debut du signal.

    `Heaviside(t)` est une convention de domaine ; `Heaviside(t - tau)`
    est une donnee de l'enonce. Les reduire tous les deux ferait d'un
    signal retarde un signal immediat, en silence.
    """
    retard_signal = sp.Symbol("tau", positive=True)
    signal = Signal(expression=sp.Heaviside(TEMPS - retard_signal))
    assert signal.sur_le_domaine().has(sp.Heaviside)
