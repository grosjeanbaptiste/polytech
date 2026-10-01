"""Transmittances et stabilite, ancrees sur l'exercice 1 du cours.

La feuille `Exs_07_Boucle_fermee_Stabilite_Reponses.md` publie, pour la
boucle ouverte `10 K / (p (1+p) (1+2p))`, quatre resultats que ces tests
reprennent un par un :

* les poles en boucle ouverte sont `0`, `-1` et `-0,5` ;
* la transmittance en boucle fermee vaut `10K / (2p^3 + 3p^2 + p + 10K)` ;
* la courbe de Nyquist croise l'axe reel vers `-6,67` ;
* le systeme est instable pour tout `K >= 0,15`.

Le dernier est le plus interessant : il se retrouve **deux fois**, par
Routh-Hurwitz sur le denominateur et par le gain de Nyquist au croisement.
Les deux chemins ne partagent aucune formule.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.signaux.domain.stabilite import Stabilite
from polytech.signaux.domain.symboles import GAIN, IMAGE
from polytech.signaux.domain.transmittance import Transmittance

P = IMAGE
OUVERTE = Transmittance(expression=10 * GAIN / (P * (1 + P) * (1 + 2 * P)))


def test_les_poles_de_la_boucle_ouverte_sont_ceux_du_corrige() -> None:
    """« les poles en boucle ouverte sont en 0 ; -1 ; -0,5 »."""
    attendus = (sp.Integer(-1), sp.Rational(-1, 2), sp.Integer(0))
    assert OUVERTE.poles() == attendus


def test_la_boucle_fermee_est_celle_que_le_corrige_publie() -> None:
    """`T_bf = 10K / (2p^3 + 3p^2 + p + 10K)`."""
    attendue = 10 * GAIN / (2 * P**3 + 3 * P**2 + P + 10 * GAIN)
    assert sp.simplify(OUVERTE.en_boucle_fermee().expression - attendue) == 0


def test_le_denominateur_de_la_boucle_fermee_est_du_troisieme_degre() -> None:
    """Trois poles en ouverte, trois en fermee : le degre ne change pas."""
    degre = 3
    assert OUVERTE.en_boucle_fermee().degre() == degre


def test_routh_hurwitz_donne_la_limite_de_stabilite_du_corrige() -> None:
    """« instable pour tout K >= 0,15 »."""
    limite = Stabilite(boucle_ouverte=OUVERTE).gain_critique(GAIN)
    assert sp.simplify(limite - sp.Rational(3, 20)) == 0


def test_le_systeme_est_stable_en_dessous_et_instable_au_dessus() -> None:
    """Le critere doit trancher des deux cotes, pas seulement a la limite."""
    stabilite = Stabilite(boucle_ouverte=OUVERTE)
    assert stabilite.est_stable({GAIN: sp.Rational(1, 10)})
    assert not stabilite.est_stable({GAIN: sp.Integer(1)})


def test_la_pulsation_du_croisement_de_l_axe_reel() -> None:
    """La partie imaginaire s'annule en `2 omega^2 = 1`, soit `omega = 1/racine(2)`."""
    croisement = Stabilite(boucle_ouverte=OUVERTE).pulsation_du_croisement()
    assert sp.simplify(croisement - 1 / sp.sqrt(2)) == 0


def test_le_gain_au_croisement_est_celui_du_corrige() -> None:
    """« remonte en croisant l'axe reel vers -6,67 » -- pour K = 1."""
    stabilite = Stabilite(boucle_ouverte=OUVERTE)
    reel = stabilite.reel_au_croisement({GAIN: sp.Integer(1)})
    assert float(reel) == pytest.approx(-6.667, abs=0.001)


def test_les_deux_chemins_donnent_la_meme_limite_de_stabilite() -> None:
    """Le recoupement : Routh-Hurwitz d'un cote, Nyquist de l'autre.

    Nyquist dit que la boucle est instable des que le gain au croisement
    depasse l'unite en module. Pour `K = 1` ce gain vaut `6,667`, donc la
    limite est `1 / 6,667`. Routh-Hurwitz, qui ne regarde que le
    denominateur, doit tomber dessus.
    """
    stabilite = Stabilite(boucle_ouverte=OUVERTE)
    par_nyquist = stabilite.gain_critique_par_nyquist(GAIN)
    par_routh = stabilite.gain_critique(GAIN)
    assert sp.simplify(par_nyquist - par_routh) == 0


def test_une_transmittance_sans_pole_est_refusee() -> None:
    """Un gain pur n'est pas un systeme : il n'a ni dynamique ni stabilite."""
    with pytest.raises(ValueError, match="aucun pole"):
        Transmittance(expression=sp.Integer(5)).poles()


def test_un_denominateur_sans_terme_constant_n_a_pas_de_gain_critique() -> None:
    """Le critere de Routh suppose tous les coefficients presents.

    Le dire en levant vaut mieux que de rendre une limite tiree d'un
    tableau incomplet, qui serait fausse sans le paraitre.
    """
    degenere = Transmittance(expression=1 / P**3)
    with pytest.raises(ValueError, match="coefficient nul"):
        Stabilite(boucle_ouverte=degenere).gain_critique(GAIN)
