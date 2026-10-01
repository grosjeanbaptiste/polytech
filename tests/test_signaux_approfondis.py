"""Trois fascicules de signaux, repris en profondeur.

Les transformees, les systemes et l'analyse temporelle. Ce que les
planches enseignent et que les modules ne portaient pas :

* la **dualite** de la transformee de Fourier, les transformees usuelles,
  le passage aux series de Fourier pour un signal periodique, et
  l'estimation numerique -- qui est ce que `numpy.fft` calcule vraiment ;
* la **linearisation** par developpement de Taylor au premier terme, qui
  est la seule facon de ramener un systeme physique au cadre LTI du cours,
  et la distinction entre modele global et modele local ;
* les reponses a l'echelon, a l'impulsion et a la **rampe**, et la
  classification par ordre que le fascicule structure en trois parties.

Les ancrages sont exacts. La transformee de `e^{-at} u(t)` vaut
`1 / (a + j omega)` ; la dualite appliquee deux fois rend le signal de
depart ; la reponse a une rampe d'un premier ordre vaut
`t - tau (1 - e^{-t/tau})`, donc son retard asymptotique est exactement
`tau` -- ce qui est le resultat que le fascicule met en avant.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.signaux.domain.symboles import IMAGE, PULSATION, TEMPS
from polytech.signaux.domain.systemes import (
    Systeme,
    est_causal,
    linearisation_de_taylor,
    point_de_fonctionnement,
)
from polytech.signaux.domain.temporel import (
    ReponsesTemporelles,
    ordre_du_systeme,
    reponse_a_une_rampe,
    reponse_harmonique,
)
from polytech.signaux.domain.transformees import (
    amplitude_et_phase,
    coefficient_de_serie_de_fourier,
    dualite,
    fourier_d_une_exponentielle_causale,
    puissance_moyenne_d_un_signal,
    pulsation_de_l_estimation_numerique,
)
from polytech.signaux.domain.transmittance import Transmittance

P = IMAGE
OMEGA = PULSATION


def test_la_transformee_de_l_exponentielle_causale() -> None:
    """`1 / (a + j omega)` : la transformee la plus utilisee du cours."""
    taux = sp.Symbol("a", positive=True)
    attendu = 1 / (taux + sp.I * OMEGA)
    assert sp.simplify(fourier_d_une_exponentielle_causale(taux) - attendu) == 0


def test_l_amplitude_et_la_phase_se_separent_du_spectre() -> None:
    """Le fascicule les trace separement : ce sont deux lectures du spectre."""
    taux = sp.Integer(1)
    amplitude, phase = amplitude_et_phase(fourier_d_une_exponentielle_causale(taux))
    assert sp.simplify(amplitude - 1 / sp.sqrt(1 + OMEGA**2)) == 0
    assert sp.simplify(phase + sp.atan(OMEGA)) == 0


def test_la_dualite_appliquee_deux_fois_rend_le_signal_de_depart() -> None:
    """C'est le controle le plus severe possible sur la dualite.

    Echanger temps et pulsation deux fois doit etre l'identite -- a la
    reflexion que la convention de signe impose, et le test la porte.
    """
    signal = sp.exp(-(TEMPS**2))
    aller = dualite(signal)
    retour = dualite(aller)
    assert sp.simplify(retour - signal.subs(TEMPS, -TEMPS)) == 0


def test_le_coefficient_de_serie_de_fourier_d_un_signal_constant() -> None:
    """Un signal constant n'a que son terme continu : `c_0 = A`, les autres nuls."""
    assert sp.simplify(coefficient_de_serie_de_fourier(sp.Integer(5), 1, 0) - 5) == 0
    assert sp.simplify(coefficient_de_serie_de_fourier(sp.Integer(5), 1, 1)) == 0


def test_une_periode_nulle_est_refusee_dans_la_serie() -> None:
    with pytest.raises(ValueError, match="periode nulle"):
        coefficient_de_serie_de_fourier(sp.Integer(1), 0, 0)


def test_la_pulsation_de_l_estimation_numerique_depend_de_la_cadence() -> None:
    """`omega_k = 2 pi k F_e / N` : ce que `numpy.fft` calcule vraiment.

    Le fascicule y insiste parce que c'est la source de toutes les erreurs
    d'interpretation d'un spectre calcule : la resolution n'est pas libre,
    elle vaut `F_e / N`.
    """
    pulsation = pulsation_de_l_estimation_numerique(
        indice=1, points=100, cadence=1000.0
    )
    assert float(pulsation) == pytest.approx(2 * 3.141592653589793 * 10.0)


def test_un_nombre_de_points_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="nombre de points"):
        pulsation_de_l_estimation_numerique(indice=1, points=0, cadence=1000.0)


def test_la_puissance_moyenne_d_une_sinusoide_vaut_la_moitie_du_carre() -> None:
    """`A^2 / 2` : resultat exact, et point fixe du chapitre."""
    amplitude = sp.Symbol("A", positive=True)
    signal = amplitude * sp.sin(OMEGA * TEMPS)
    periode = 2 * sp.pi / OMEGA
    assert (
        sp.simplify(
            puissance_moyenne_d_un_signal(signal, periode) - amplitude**2 / 2,
        )
        == 0
    )


def test_la_linearisation_au_premier_terme_passe_par_le_point() -> None:
    """`f(x) ~ f(x_0) + f'(x_0) (x - x_0)` : la tangente, et rien de plus.

    C'est la seule facon de ramener un systeme physique au cadre LTI du
    cours, et le fascicule le dit : « developpement en serie de Taylor
    limite au premier terme ».
    """
    variable = sp.Symbol("x", real=True)
    fonction = variable**2
    tangente = linearisation_de_taylor(fonction, variable, sp.Integer(3))
    assert sp.simplify(tangente.subs(variable, 3) - 9) == 0
    assert sp.simplify(sp.diff(tangente, variable) - 6) == 0


def test_la_linearisation_est_exacte_sur_une_fonction_deja_affine() -> None:
    """Controle de coherence : lineariser un lineaire ne change rien."""
    variable = sp.Symbol("x", real=True)
    affine = 3 * variable + 1
    assert (
        sp.simplify(linearisation_de_taylor(affine, variable, sp.Integer(0)) - affine)
        == 0
    )


def test_le_point_de_fonctionnement_annule_la_derivee_du_modele() -> None:
    """Un equilibre est un point ou le systeme ne bouge plus de lui-meme."""
    variable = sp.Symbol("x", real=True)
    modele = variable * (1 - variable)
    assert point_de_fonctionnement(modele, variable) == (sp.Integer(0), sp.Integer(1))


def test_un_modele_sans_equilibre_est_nomme() -> None:
    """Le dire plutot que de rendre une liste vide qu'un appelant ignorerait."""
    variable = sp.Symbol("x", real=True)
    with pytest.raises(ValueError, match="aucun point de fonctionnement"):
        point_de_fonctionnement(sp.Integer(1), variable)


def test_un_systeme_sans_avance_est_causal() -> None:
    """Il ne depend pas de son entree future : condition du cadre du cours."""
    assert est_causal(Systeme(operateur=lambda entree: 2 * entree))


def test_un_systeme_qui_anticipe_n_est_pas_causal() -> None:
    """Et le dire avant de lui chercher une transmittance."""
    avance = sp.Symbol("tau", positive=True)
    anticipateur = Systeme(operateur=lambda entree: entree.subs(TEMPS, TEMPS + avance))
    assert not est_causal(anticipateur)


def test_la_reponse_a_une_rampe_d_un_premier_ordre_retarde_de_tau() -> None:
    """`t - tau (1 - e^{-t/tau})` : l'ecart asymptotique vaut exactement tau.

    C'est le resultat que le fascicule met en avant, et il est exact : un
    premier ordre suit une rampe avec un retard permanent egal a sa
    constante de temps.
    """
    premier = Transmittance(expression=1 / (1 + P))
    reponse = reponse_a_une_rampe(premier)
    ecart = sp.limit(TEMPS - reponse, TEMPS, sp.oo)
    assert sp.simplify(ecart - 1) == 0


def test_la_reponse_harmonique_est_la_transmittance_en_j_omega() -> None:
    premier = Transmittance(expression=1 / (1 + P))
    attendu = 1 / (1 + sp.I * OMEGA)
    assert sp.simplify(reponse_harmonique(premier) - attendu) == 0


@pytest.mark.parametrize(
    ("expression", "attendu"),
    [(1 / (1 + P), 1), (1 / (1 + P + P**2), 2), (1 / (1 + P**3), 3)],
)
def test_l_ordre_du_systeme_est_le_degre_de_son_denominateur(
    expression: sp.Expr,
    attendu: int,
) -> None:
    """Le fascicule structure tout le chapitre sur cette classification."""
    assert ordre_du_systeme(Transmittance(expression=expression)) == attendu


def test_un_second_ordre_a_poles_doubles_est_distingue_des_poles_simples() -> None:
    """Le fascicule traite les deux cas separement, et leurs reponses different.

    A poles doubles la reponse porte un facteur `t` que les poles simples
    n'ont pas -- c'est ce qui distingue l'amortissement critique.
    """
    double = ReponsesTemporelles(
        transmittance=Transmittance(expression=1 / (1 + P) ** 2)
    )
    simple = ReponsesTemporelles(
        transmittance=Transmittance(expression=1 / ((1 + P) * (1 + 2 * P))),
    )
    assert double.impulsionnelle().has(TEMPS)
    assert sp.simplify(double.impulsionnelle() - TEMPS * sp.exp(-TEMPS)) == 0
    assert sp.simplify(simple.impulsionnelle().coeff(TEMPS, 1)) == 0
