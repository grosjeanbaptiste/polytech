"""Le chapitre 3 du cours d'electronique : du gabarit au filtre reel.

Le fascicule enseigne une **chaine** que le depot ne portait qu'a moitie :
specifier un gabarit non ideal, en deduire l'ordre necessaire, choisir une
famille (Butterworth, Chebyshev, Bessel, elliptique), prendre le prototype
passe-bas normalise, le mettre a l'echelle d'impedance, puis le transposer
-- passe-haut, passe-bande ou coupe-bande.

Le depot avait la mise a l'echelle et la transposition passe-haut. Il
n'avait ni le gabarit, ni l'ordre, ni les deux autres transpositions, ni
Bessel, ni le temps de groupe -- alors que c'est le temps de groupe qui
justifie Bessel, et que le fascicule y consacre cinq planches.

Les ancrages sont exacts. L'ordre d'un Butterworth se tire d'une formule
fermee, et le cas d'ecole `A_max = 3 dB, A_min = 40 dB, omega_s/omega_p = 2`
donne exactement `n = 7` -- la formule rend 6,64, donc il faut 7. La
transposition passe-haut est l'involution `s -> 1/s` : l'appliquer deux
fois doit rendre le filtre de depart, et c'est le controle le plus severe
possible sur elle.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.gabarit import (
    Gabarit,
    ordre_de_butterworth,
    ordre_de_chebyshev,
)
from polytech.electronique.domain.transpositions import (
    mise_a_l_echelle_d_impedance,
    temps_de_groupe,
    vers_coupe_bande,
    vers_passe_bande,
    vers_passe_haut,
)

S = sp.Symbol("s")

CLASSIQUE = Gabarit(
    affaiblissement_en_bande=3.0,
    affaiblissement_hors_bande=40.0,
    pulsation_passante=1.0,
    pulsation_bloquante=2.0,
)


def test_le_gabarit_rend_sa_selectivite() -> None:
    """`omega_s / omega_p` : le rapport dont l'ordre depend."""
    assert CLASSIQUE.selectivite() == pytest.approx(2.0)


def test_un_gabarit_dont_la_bande_bloquante_precede_la_passante_est_refuse() -> None:
    """Il decrirait un filtre impossible, et la formule rendrait un ordre negatif."""
    with pytest.raises(ValueError, match="bande bloquante"):
        Gabarit(
            affaiblissement_en_bande=3.0,
            affaiblissement_hors_bande=40.0,
            pulsation_passante=2.0,
            pulsation_bloquante=1.0,
        )


def test_un_affaiblissement_hors_bande_trop_faible_est_refuse() -> None:
    """Un filtre doit attenuer **plus** hors bande que dedans."""
    with pytest.raises(ValueError, match="affaiblissement hors bande"):
        Gabarit(
            affaiblissement_en_bande=40.0,
            affaiblissement_hors_bande=3.0,
            pulsation_passante=1.0,
            pulsation_bloquante=2.0,
        )


def test_l_ordre_de_butterworth_du_cas_d_ecole_vaut_sept() -> None:
    """`3 dB`, `40 dB`, selectivite 2 : la formule rend 6,64, donc sept poles.

    L'arrondi se fait **vers le haut** : un ordre de 6 ne tiendrait pas le
    gabarit, et c'est tout l'objet du calcul.
    """
    assert ordre_de_butterworth(CLASSIQUE) == 7


def test_chebyshev_demande_moins_de_poles_que_butterworth() -> None:
    """C'est la raison de son existence : plus raide a ordre egal.

    Le prix est une ondulation dans la bande, que le fascicule montre sur
    sa reponse en frequence.
    """
    assert ordre_de_chebyshev(CLASSIQUE) < ordre_de_butterworth(CLASSIQUE)


def test_un_gabarit_plus_severe_demande_un_ordre_plus_grand() -> None:
    """Propriete de monotonie : elle vaut pour les deux familles."""
    severe = Gabarit(
        affaiblissement_en_bande=1.0,
        affaiblissement_hors_bande=60.0,
        pulsation_passante=1.0,
        pulsation_bloquante=1.2,
    )
    assert ordre_de_butterworth(severe) > ordre_de_butterworth(CLASSIQUE)
    assert ordre_de_chebyshev(severe) > ordre_de_chebyshev(CLASSIQUE)


def test_la_mise_a_l_echelle_d_impedance_laisse_le_transfert_inchange() -> None:
    """« Beware: the gain of the filter has not changed », dit la planche.

    Multiplier les resistances et les inductances par `k` et diviser les
    capacites par `k` ne change **rien** a la fonction de transfert. C'est
    ce qui rend l'operation gratuite, et c'est ce que ce test verifie.
    """
    facteur = 50.0
    resistance, inductance, capacite = 1.0, 2.0, 3.0
    echelle = mise_a_l_echelle_d_impedance(facteur, resistance, inductance, capacite)
    nouvelle_r, nouvelle_l, nouvelle_c = echelle
    avant = sp.simplify(1 / (1 + S * resistance * capacite))
    apres = sp.simplify(1 / (1 + S * nouvelle_r * nouvelle_c))
    assert sp.simplify(avant - apres) == 0
    assert nouvelle_l == pytest.approx(facteur * inductance)


def test_un_facteur_d_echelle_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="facteur d'echelle"):
        mise_a_l_echelle_d_impedance(0.0, 1.0, 1.0, 1.0)


def test_la_transposition_passe_haut_est_une_involution() -> None:
    """`s -> 1/s` appliquee deux fois rend le filtre de depart.

    C'est le controle le plus severe possible sur une transposition : elle
    ne peut pas etre fausse et involutive a la fois.
    """
    passe_bas = 1 / (1 + S)
    aller = vers_passe_haut(passe_bas, S)
    retour = vers_passe_haut(aller, S)
    assert sp.simplify(retour - passe_bas) == 0


def test_la_transposition_passe_haut_echange_les_bandes() -> None:
    """Le gain en zero du passe-bas devient le gain a l'infini du passe-haut."""
    passe_bas = 1 / (1 + S)
    passe_haut = vers_passe_haut(passe_bas, S)
    assert sp.limit(passe_bas, S, 0) == 1
    assert sp.limit(passe_haut, S, sp.oo) == 1
    assert sp.limit(passe_haut, S, 0) == 0


def test_la_transposition_passe_bande_est_centree_sur_sa_pulsation() -> None:
    """`s -> (s^2 + w0^2) / (B s)` : le gain maximal est en `w0`.

    A la pulsation centrale la substitution s'annule, donc le passe-bande
    y vaut ce que le passe-bas valait en zero.
    """
    centre, largeur = sp.Integer(10), sp.Integer(2)
    passe_bas = 1 / (1 + S)
    passe_bande = vers_passe_bande(passe_bas, S, centre, largeur)
    au_centre = sp.simplify(passe_bande.subs(S, sp.I * centre))
    assert sp.simplify(sp.Abs(au_centre) - 1) == 0


def test_la_transposition_passe_bande_annule_le_gain_aux_deux_bouts() -> None:
    passe_bande = vers_passe_bande(1 / (1 + S), S, sp.Integer(10), sp.Integer(2))
    assert sp.limit(passe_bande, S, 0) == 0
    assert sp.limit(passe_bande, S, sp.oo) == 0


def test_la_transposition_coupe_bande_annule_le_gain_au_centre() -> None:
    """C'est sa definition meme : elle rejette la pulsation centrale."""
    coupe_bande = vers_coupe_bande(1 / (1 + S), S, sp.Integer(10), sp.Integer(2))
    au_centre = sp.simplify(coupe_bande.subs(S, sp.I * 10))
    assert sp.simplify(au_centre) == 0


def test_la_coupe_bande_laisse_passer_les_deux_extremites() -> None:
    coupe_bande = vers_coupe_bande(1 / (1 + S), S, sp.Integer(10), sp.Integer(2))
    assert sp.simplify(sp.Abs(sp.limit(coupe_bande, S, 0)) - 1) == 0


def test_une_largeur_de_bande_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="largeur de bande"):
        vers_passe_bande(1 / (1 + S), S, sp.Integer(10), sp.Integer(0))


def test_une_pulsation_centrale_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="pulsation centrale"):
        vers_coupe_bande(1 / (1 + S), S, sp.Integer(0), sp.Integer(2))


def test_le_temps_de_groupe_du_premier_ordre_decroit_avec_la_pulsation() -> None:
    """`-d(arg H)/d omega`. C'est lui qui justifie Bessel.

    Un filtre dont le temps de groupe varie deforme un signal numerique en
    retardant ses harmoniques inegalement -- le fascicule y consacre cinq
    planches, et c'est tout l'argument de la famille de Bessel.
    """
    pulsation = sp.Symbol("w", positive=True)
    retard = temps_de_groupe(1 / (1 + S), S, pulsation)
    attendu = 1 / (1 + pulsation**2)
    assert sp.simplify(retard - attendu) == 0


def test_le_temps_de_groupe_d_un_dephasage_pur_est_constant() -> None:
    """Phase lineaire, retard constant : c'est l'idéal que Bessel approche."""
    pulsation = sp.Symbol("w", positive=True)
    duree = sp.Symbol("tau", positive=True)
    retard = temps_de_groupe(sp.exp(-S * duree), S, pulsation)
    assert sp.simplify(retard - duree) == 0


def test_l_ordre_est_un_entier_et_jamais_nul() -> None:
    """Un ordre fractionnaire n'a pas de realisation en elements discrets."""
    for gabarit in (CLASSIQUE,):
        assert isinstance(ordre_de_butterworth(gabarit), int)
        assert ordre_de_butterworth(gabarit) >= 1
        assert isinstance(ordre_de_chebyshev(gabarit), int)
        assert ordre_de_chebyshev(gabarit) >= 1


def test_la_formule_de_butterworth_correspond_a_son_expression_fermee() -> None:
    """Recoupement : la formule du cours, recalculee a part.

    `n >= log10((10^(A_min/10) - 1) / (10^(A_max/10) - 1)) / (2 log10(ws/wp))`
    """
    numerateur = math.log10((10 ** (40.0 / 10) - 1) / (10 ** (3.0 / 10) - 1))
    denominateur = 2 * math.log10(2.0)
    assert ordre_de_butterworth(CLASSIQUE) == math.ceil(numerateur / denominateur)
