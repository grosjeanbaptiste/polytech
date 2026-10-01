"""Quatre fascicules de plus : acquisition, conversion, circuits, controleur.

**Acquisition** (`Slides_4_part1`) : la densite spectrale de puissance et
sa convention unilaterale, l'autocorrelation et l'autocovariance, le bruit
blanc gaussien additif, et les processus stationnaires au sens large. Le
fascicule les introduit par **trois defis** nommes -- faibles tensions du
transducteur, bruits d'origine externe, relations non lineaires -- et
c'est cette structure qui est reprise.

**Conversion** (`Slides_4_part3`) : le temps d'ouverture, dont le
fascicule donne l'analogie photographique, et ses trois cas
d'echantillonnage -- correct, pleine echelle depassee, pleine echelle non
couverte. Plus la formule de Friis en cascade, deja portee.

**Des champs aux circuits** (`Slides_1_part2`) : les lois de Kirchhoff et
leur **preuve**, les composants idealises, l'inductance et la capacite
cachees, et le schema equivalent.

**Controleur** (plan du cours) : le PID, ses trois parametres, et la
demarche de synthese -- des specifications temporelles aux contraintes
frequentielles, puis le calcul.

L'ancrage le plus net est l'autocorrelation d'un bruit blanc : elle vaut
une impulsion a l'origine et **zero partout ailleurs**, ce qui est la
definition meme de la blancheur, et ce qui rend sa densite spectrale
plate.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.acquisition import (
    autocorrelation_d_un_bruit_blanc,
    autocovariance,
    densite_spectrale_bilaterale,
    densite_spectrale_unilaterale_depuis_bilaterale,
    variance_d_un_bruit_borne,
)
from polytech.electronique.domain.conversion import (
    EchantillonnageDeLEtendue,
    erreur_de_temps_d_ouverture,
    etat_de_l_echantillonnage,
    temps_d_ouverture_maximal,
)
from polytech.electronique.domain.des_champs_aux_circuits import (
    inductance_cachee_d_une_boucle,
    kirchhoff_des_courants,
    kirchhoff_des_tensions,
)
from polytech.signaux.domain.controleur import (
    ProportionnelIntegralDerive,
    pulsation_de_correction,
)
from polytech.signaux.domain.transmittance import Transmittance


def test_l_autocorrelation_d_un_bruit_blanc_est_nulle_hors_de_l_origine() -> None:
    """C'est la definition de la blancheur, et ce qui rend sa DSP plate.

    Deux echantillons distincts d'un bruit blanc sont decorrelees, quelle
    que soit leur proximite -- ce qui est une idealisation, puisqu'aucun
    bruit reel n'a de bande infinie.
    """
    assert autocorrelation_d_un_bruit_blanc(densite=1e-16, retard=0.0) > 0
    assert autocorrelation_d_un_bruit_blanc(densite=1e-16, retard=1e-9) == 0.0


def test_l_autocovariance_retire_la_moyenne_de_l_autocorrelation() -> None:
    """`C(tau) = R(tau) - m^2` : les deux coincident pour un signal centre."""
    assert autocovariance(correlation=5.0, moyenne=2.0) == pytest.approx(1.0)
    assert autocovariance(correlation=5.0, moyenne=0.0) == pytest.approx(5.0)


def test_la_densite_unilaterale_vaut_le_double_de_la_bilaterale() -> None:
    """C'est la seule difference entre les deux conventions, et elle vaut deux.

    Le fascicule consacre une planche a la convention unilaterale parce
    que la confusion double ou divise par deux toute puissance de bruit.
    """
    bilaterale = densite_spectrale_bilaterale(unilaterale=4e-18)
    assert bilaterale == pytest.approx(2e-18)
    retour = densite_spectrale_unilaterale_depuis_bilaterale(bilaterale)
    assert retour == pytest.approx(4e-18)


def test_la_variance_d_un_bruit_borne_croit_avec_la_bande() -> None:
    """`sigma^2 = S B` : doubler la bande double la puissance de bruit."""
    etroite = variance_d_un_bruit_borne(densite=1e-16, bande=100.0)
    large = variance_d_un_bruit_borne(densite=1e-16, bande=200.0)
    assert large == pytest.approx(2 * etroite)


def test_une_bande_nulle_ne_porte_aucun_bruit() -> None:
    with pytest.raises(ValueError, match="bande nulle"):
        variance_d_un_bruit_borne(densite=1e-16, bande=0.0)


def test_l_erreur_de_temps_d_ouverture_croit_avec_la_frequence() -> None:
    """`2 pi f t_a` en fraction de pleine echelle : l'analogie photographique.

    Le fascicule la file jusqu'au bout -- un temps de pose trop long floute
    un sujet rapide, et un temps d'ouverture trop long floute un signal
    rapide.
    """
    lente = erreur_de_temps_d_ouverture(frequence=1e3, temps=1e-9)
    rapide = erreur_de_temps_d_ouverture(frequence=1e6, temps=1e-9)
    assert rapide == pytest.approx(1000 * lente)
    assert lente == pytest.approx(2 * math.pi * 1e3 * 1e-9)


def test_le_temps_d_ouverture_maximal_tient_l_erreur_sous_un_demi_bit() -> None:
    """Critere de dimensionnement : l'erreur ne doit pas depasser le demi-pas.

    Verification croisee : appliquer ce temps doit rendre exactement
    l'erreur visee.
    """
    maximal = temps_d_ouverture_maximal(frequence=1e6, bits=12)
    erreur = erreur_de_temps_d_ouverture(frequence=1e6, temps=maximal)
    assert erreur == pytest.approx(1 / 2**13, rel=1e-9)


def test_un_nombre_de_bits_nul_est_refuse_dans_le_temps_d_ouverture() -> None:
    with pytest.raises(ValueError, match="nombre de bits"):
        temps_d_ouverture_maximal(frequence=1e6, bits=0)


@pytest.mark.parametrize(
    ("amplitude", "attendu"),
    [
        (0.9, EchantillonnageDeLEtendue.CORRECT),
        (1.5, EchantillonnageDeLEtendue.DEPASSEE),
        (0.1, EchantillonnageDeLEtendue.SOUS_EXPLOITEE),
    ],
)
def test_les_trois_cas_d_echantillonnage_du_fascicule(
    amplitude: float,
    attendu: EchantillonnageDeLEtendue,
) -> None:
    """Le fascicule montre les trois sur un convertisseur quatre bits.

    Depasser l'etendue ecrete ; ne pas la couvrir gaspille des bits. Les
    deux fautes sont symetriques et le dimensionnement doit les eviter
    toutes deux.
    """
    assert etat_de_l_echantillonnage(amplitude, etendue=1.0) == attendu


def test_une_etendue_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="etendue nulle"):
        etat_de_l_echantillonnage(0.5, etendue=0.0)


def test_la_loi_des_courants_de_kirchhoff_somme_a_zero() -> None:
    """`somme I = 0` a un noeud : c'est la conservation de la charge."""
    assert kirchhoff_des_courants((1.0, -0.6, -0.4)) == pytest.approx(0.0)
    assert kirchhoff_des_courants((1.0, -0.5)) != pytest.approx(0.0)


def test_la_loi_des_tensions_de_kirchhoff_somme_a_zero() -> None:
    """`somme U = 0` sur une maille : c'est l'existence du potentiel."""
    assert kirchhoff_des_tensions((12.0, -5.0, -7.0)) == pytest.approx(0.0)


def test_une_maille_vide_est_refusee() -> None:
    with pytest.raises(ValueError, match="aucune branche"):
        kirchhoff_des_tensions(())


def test_l_inductance_cachee_croit_avec_l_aire_de_la_boucle() -> None:
    """C'est pourquoi on reduit les boucles de courant sur un circuit imprime.

    Le fascicule l'appelle « hidden inductance » : elle n'est sur aucun
    schema, et elle fixe la vitesse a laquelle un courant peut changer.
    """
    petite = inductance_cachee_d_une_boucle(aire=1e-4, perimetre=0.04)
    grande = inductance_cachee_d_une_boucle(aire=4e-4, perimetre=0.08)
    assert grande > petite


def test_une_boucle_d_aire_nulle_n_a_pas_d_inductance() -> None:
    with pytest.raises(ValueError, match="aire nulle"):
        inductance_cachee_d_une_boucle(aire=0.0, perimetre=0.04)


def test_le_pid_a_exactement_trois_parametres() -> None:
    """« Comme seuls trois parametres Gc, Ti, Td disponibles », dit la planche.

    C'est ce qui borne la synthese : trois parametres ne peuvent satisfaire
    que trois contraintes.
    """
    controleur = ProportionnelIntegralDerive(
        gain=sp.Integer(2),
        temps_integral=sp.Integer(1),
        temps_derive=sp.Rational(1, 10),
    )
    assert controleur.parametres() == 3


def test_le_pid_annule_l_erreur_statique_comme_le_pi() -> None:
    """Le terme derive n'ajoute rien en continu : c'est l'integral qui agit."""
    controleur = ProportionnelIntegralDerive(
        gain=sp.Integer(2),
        temps_integral=sp.Integer(1),
        temps_derive=sp.Rational(1, 10),
    )
    chaine = sp.Symbol("p")
    ferme = controleur.erreur_statique(Transmittance(1 / (1 + chaine)))
    assert sp.simplify(ferme) == 0


def test_un_temps_derive_negatif_est_refuse() -> None:
    with pytest.raises(ValueError, match="temps derive negatif"):
        ProportionnelIntegralDerive(
            gain=sp.Integer(1),
            temps_integral=sp.Integer(1),
            temps_derive=sp.Integer(-1),
        )


def test_la_pulsation_de_correction_se_choisit_sur_l_argument() -> None:
    """La planche la lie a la marge de phase visee.

    « Comment choisir la pulsation de correction ? Comme -l'argument... »
    C'est la pulsation ou l'argument de la chaine vaut `-pi + marge`.
    """
    chaine = sp.Symbol("p")
    pulsation = pulsation_de_correction(
        Transmittance(1 / (chaine * (1 + chaine))),
        marge_visee=sp.pi / 4,
    )
    assert pulsation > 0
