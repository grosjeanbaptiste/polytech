"""Les planches d'electronique : les cellules de Sallen-Key (chapitre 3, partie II).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_3_part2.pdf`, planches
9 a 43, lues sur les pages rendues -- leurs formules sont des images.

Les oracles : la cascade de la planche 10, le Butterworth d'ordre six de
la planche 11, les deux sections du premier ordre (planches 15 et 17), la
fonction de transfert du passe-bas (planche 22), ses quatre methodes
(planches 24 a 27), l'exemple chiffre des planches 28 a 33, et les
variantes passe-haut, passe-bande et coupe-bande (planches 35 a 43).

Un ecart : la planche 41 annonce un gain « ~ 27.3551 » pour le passe-bande
d'exemple ; `mu Q / racine(2)` avec `mu = 3,8586` et `Q = 10` vaut 27,28.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.butterworth import poles_de_butterworth
from polytech.electronique.domain.methodes_sallen_key import (
    Methode,
    cellule_normalisee,
    gain_de_la_methode,
    methodes_a_gain_fixe,
    qualite_des_elements_egaux,
    resistances_de_gain,
)
from polytech.electronique.domain.sallen_key_a_gain import SallenKeyPasseBas
from polytech.electronique.domain.sallen_key_variantes import (
    passe_bande_a_elements_egaux,
    resistances_du_passe_haut,
    transfert_coupe_bande,
    transfert_passe_bande,
    transfert_passe_haut,
)
from polytech.electronique.domain.sections_actives import (
    biquads_de_butterworth,
    section_inverseuse,
    section_non_inverseuse,
)

S = sp.Symbol("s")
R1, R2, R3, C1, C2, MU = sp.symbols("R_1 R_2 R_3 C_1 C_2 mu", positive=True)


def test_le_butterworth_d_ordre_six_se_decompose_en_trois_cellules() -> None:
    """Planche 11 : `s^2 + 0,518 s + 1`, `s^2 + 1,414 s + 1`, `s^2 + 1,932 s + 1`."""
    amortissements = biquads_de_butterworth(6)
    assert amortissements == pytest.approx((0.518, 1.414, 1.932), abs=5e-4)
    produit = sp.prod([S**2 + b * S + 1 for b in amortissements])
    coefficients = sp.Poly(sp.expand(produit), S).all_coeffs()
    attendus = (1, 3.864, 7.464, 9.142, 7.464, 3.864, 1)
    assert [float(c) for c in coefficients] == pytest.approx(attendus, abs=5e-4)
    racines = sorted(-2 * p.real for p in poles_de_butterworth(6) if p.imag > 0)
    assert racines == pytest.approx(amortissements)


def test_le_butterworth_d_ordre_quatre_de_l_exemple() -> None:
    """Planche 28 : `(s^2 + 0,765 s + 1)(s^2 + 1,848 s + 1)`."""
    assert biquads_de_butterworth(4) == pytest.approx((0.765, 1.848), abs=5e-4)
    with pytest.raises(ValueError, match="ordre pair"):
        biquads_de_butterworth(3)


def test_la_section_inverseuse_du_premier_ordre() -> None:
    """Planche 15 : `-(a_1 s + a_0) / (s + b_0)`, soit `-Y_1 / Y_2`."""
    a1, a0, b0 = sp.symbols("a_1 a_0 b_0", positive=True)
    transfert = section_inverseuse(a1 * S + a0, S + b0)
    assert sp.simplify(transfert + (a1 * S + a0) / (S + b0)) == 0
    echelle = section_inverseuse((a1 * S + a0) / 1000, (S + b0) / 1000)
    assert sp.simplify(echelle - transfert) == 0


def test_la_section_non_inverseuse_du_premier_ordre() -> None:
    """Planche 17 : `1 + Y_1 / Y_2`, avec `Y_1 = (a_1 - 1) s + (a_0 - b_0)`."""
    a1, a0, b0 = sp.symbols("a_1 a_0 b_0", positive=True)
    transfert = section_non_inverseuse((a1 - 1) * S + (a0 - b0), S + b0)
    assert sp.simplify(transfert - (a1 * S + a0) / (S + b0)) == 0


CELLULE = SallenKeyPasseBas(R1, R2, C1, C2, MU)


def test_la_fonction_de_transfert_du_passe_bas() -> None:
    """Planche 22 : l'analyse nodale redonne la formule de Kendall Su."""
    numerateur = MU / (R1 * R2 * C1 * C2)
    lineaire = 1 / (R1 * C1) + 1 / (R2 * C1) + (1 - MU) / (R2 * C2)
    planche = numerateur / (S**2 + lineaire * S + 1 / (R1 * R2 * C1 * C2))
    assert sp.simplify(CELLULE.transfert(S) - planche) == 0


def test_la_pulsation_le_facteur_de_qualite_et_le_gain() -> None:
    """Planche 23 : `w_0 = 1 / racine(R_1 R_2 C_1 C_2)`, `Q`, et `G = mu`."""
    pulsation, qualite = CELLULE.pulsation(), CELLULE.qualite()
    canonique = MU * pulsation**2 / (S**2 + pulsation / qualite * S + pulsation**2)
    assert sp.simplify(CELLULE.transfert(S) - canonique) == 0
    assert CELLULE.transfert(S).subs(S, 0) == MU


@pytest.mark.parametrize("methode", list(Methode))
def test_chaque_methode_realise_la_pulsation_et_la_qualite_demandees(
    methode: Methode,
) -> None:
    """Planches 24 a 27 : quatre choix de composants pour les memes `w_0` et `Q`."""
    qualite, pulsation = sp.Rational(3, 2), sp.Integer(7)
    cellule = cellule_normalisee(methode, qualite, pulsation)
    assert sp.simplify(cellule.pulsation() - pulsation) == 0
    assert sp.simplify(cellule.qualite() - qualite) == 0
    assert sp.simplify(cellule.gain - gain_de_la_methode(methode, qualite)) == 0


def test_les_quatre_methodes_de_dimensionnement() -> None:
    """Planches 24 a 27 : les equations de chaque methode, telles qu'imprimees."""
    q, w = sp.symbols("Q omega_0", positive=True)
    un = cellule_normalisee(Methode.ELEMENTS_EGAUX, q, w)
    assert (un.r1, un.r2, un.c1, un.c2, un.gain) == (1 / w, 1 / w, 1, 1, 3 - 1 / q)
    deux = cellule_normalisee(Methode.CAPACITES_EGALES, q, w)
    assert (deux.r1, deux.r2, deux.c1, deux.c2, deux.gain) == (
        q / w,
        1 / (q * w),
        1,
        1,
        2,
    )
    assert sp.simplify(deux.r1 / deux.r2 - q**2) == 0
    trois = cellule_normalisee(Methode.SENSIBILITE_MODEREE, q, w)
    assert (trois.c1, trois.c2, trois.gain) == (sp.sqrt(3) * q, 1, sp.Rational(4, 3))
    assert (trois.r1, trois.r2) == (1 / (w * q), 1 / (sp.sqrt(3) * w))
    quatre = cellule_normalisee(Methode.SENSIBILITE_MINIMALE, q, w)
    assert (quatre.r1, quatre.r2, quatre.gain) == (1, 1, 1)
    assert (quatre.c1, quatre.c2) == (2 * q / w, 1 / (2 * q * w))
    assert sp.simplify(quatre.c1 / quatre.c2 - 4 * q**2) == 0


def test_le_gain_que_chaque_methode_impose() -> None:
    """Planches 24 a 27 : `3 - 1/Q`, 2, 4/3 et 1 ; seule la premiere se regle."""
    assert qualite_des_elements_egaux(2.5) == pytest.approx(2.0)
    assert qualite_des_elements_egaux(1.0) == pytest.approx(0.5)
    for qualite in (0.6, 1.0, 7.0):
        gain = float(gain_de_la_methode(Methode.ELEMENTS_EGAUX, qualite))
        assert qualite_des_elements_egaux(gain) == pytest.approx(qualite)
    for trop in (3.0, 3.5):
        with pytest.raises(ValueError, match="gain de trois ou plus"):
            qualite_des_elements_egaux(trop)
    assert methodes_a_gain_fixe(2.5) == ()
    assert methodes_a_gain_fixe(2.0) == (Methode.CAPACITES_EGALES,)
    assert methodes_a_gain_fixe(4 / 3) == (Methode.SENSIBILITE_MODEREE,)
    assert methodes_a_gain_fixe(1.0) == (Methode.SENSIBILITE_MINIMALE,)


def test_les_resistances_de_gain_compensent_les_courants_de_polarisation() -> None:
    """Planche 30 : `mu = 2,235`, `R_a // R_b = 2` ; 3,620 et 4,469 ohm."""
    vers_la_masse, retour = resistances_de_gain(3 - 0.765, 2.0)
    assert (vers_la_masse, retour) == pytest.approx((3.620, 4.469), abs=1.5e-3)
    assert 1 + retour / vers_la_masse == pytest.approx(2.235)
    parallele = vers_la_masse * retour / (vers_la_masse + retour)
    assert parallele == pytest.approx(2.0)
    egales = resistances_de_gain(2.0, 0.541 + 1.848)
    assert egales == pytest.approx((4.778, 4.778), abs=1e-3)
    with pytest.raises(ValueError, match="gain inferieur ou egal a un"):
        resistances_de_gain(1.0, 2.0)


def test_l_exemple_du_butterworth_d_ordre_quatre() -> None:
    """Planches 30 a 33 : 100 nF, 1 kHz ; 1,591 kohm, 861 ohm, 2,94 kohm."""
    premier_b, second_b = biquads_de_butterworth(4)
    coupure = 2 * math.pi * 1000
    assert 1 / coupure == pytest.approx(1.591e-4, rel=1e-3)
    echelle = (1 / coupure) / 100e-9
    assert echelle == pytest.approx(1591.5, rel=1e-4)
    premier = cellule_normalisee(Methode.ELEMENTS_EGAUX, 1 / premier_b, 1)
    reel = premier.denormalisee(coupure=coupure, impedance=echelle)
    assert float(reel.c1) == pytest.approx(100e-9)
    assert float(reel.r1) == pytest.approx(1591.5, rel=1e-4)
    assert float(reel.gain) == pytest.approx(2.235, abs=5e-4)
    second = cellule_normalisee(Methode.CAPACITES_EGALES, 1 / second_b, 1)
    assert (float(second.r1), float(second.r2)) == pytest.approx(
        (0.541, 1.848), abs=5e-4
    )
    second_reel = second.denormalisee(coupure=coupure, impedance=echelle)
    assert float(second_reel.r1) == pytest.approx(861.3, rel=1e-3)
    assert float(second_reel.r2) == pytest.approx(2941.0, rel=1e-3)
    pulsation = float(second_reel.pulsation())
    assert pulsation == pytest.approx(coupure)
    assert float(second_reel.qualite()) == pytest.approx(1 / second_b)


def test_la_cascade_des_deux_cellules_est_un_butterworth() -> None:
    """Planche 10 : `H = H_1 H_2`, chaque cellule isolee de la suivante."""
    premier_b, second_b = biquads_de_butterworth(4)
    premier = cellule_normalisee(Methode.ELEMENTS_EGAUX, 1 / premier_b, 1)
    second = cellule_normalisee(Methode.CAPACITES_EGALES, 1 / second_b, 1)
    cascade = premier.transfert(S) * second.transfert(S)
    continu = float(cascade.subs(S, 0))
    assert continu == pytest.approx(2.235 * 2, abs=1e-3)
    for pulsation in (0.5, 1.0, 2.0):
        module = abs(complex(cascade.subs(S, 1j * pulsation))) / continu
        assert module == pytest.approx(1 / math.sqrt(1 + pulsation**8))


def test_la_fonction_de_transfert_du_passe_haut() -> None:
    """Planche 35 : le terme en `s` est `1/R_2C_2 + 1/R_2C_1 + (1-mu)/R_1C_1`."""
    lineaire = 1 / (R2 * C2) + 1 / (R2 * C1) + (1 - MU) / (R1 * C1)
    planche = MU * S**2 / (S**2 + lineaire * S + 1 / (R1 * R2 * C1 * C2))
    assert sp.simplify(transfert_passe_haut((R1, R2), (C1, C2), MU, S) - planche) == 0


def test_le_passe_haut_a_capacites_unitaires() -> None:
    """Planche 37 : `R_2` racine positive, et `R_1 = 1 / (b_0 R_2)`."""
    b0, b1, mu = 4.0, 1.2, 1.8
    premiere, seconde = resistances_du_passe_haut(b0, b1, mu)
    assert premiere * seconde == pytest.approx(1 / b0)
    assert 2 / seconde + (1 - mu) / premiere == pytest.approx(b1)
    transfert = transfert_passe_haut((premiere, seconde), (1, 1), mu, S)
    attendu = mu * S**2 / (S**2 + b1 * S + b0)
    assert abs(complex((transfert - attendu).subs(S, 2j))) < 1e-12
    with pytest.raises(ValueError, match="gain inferieur ou egal a un"):
        resistances_du_passe_haut(b0, b1, 1.0)


def test_la_fonction_de_transfert_du_passe_bande() -> None:
    """Planche 38 : numerateur `mu s / (R_1 C_1)`, cinq elements passifs."""
    lineaire = 1 / (R1 * C1) + 1 / (R3 * C2) + 1 / (R3 * C1) + (1 - MU) / (R2 * C1)
    constant = (R1 + R2) / (R1 * R2 * R3 * C1 * C2)
    planche = MU / (R1 * C1) * S / (S**2 + lineaire * S + constant)
    transfert = transfert_passe_bande((R1, R2, R3), (C1, C2), MU, S)
    assert sp.simplify(transfert - planche) == 0


def test_l_exemple_du_passe_bande() -> None:
    """Planches 40 et 41 : `R = racine(2) / w_0`, `mu = 4 - racine(2) / Q`."""
    resistance, gain_de_l_etage = passe_bande_a_elements_egaux(10.0, 1.0)
    assert resistance == pytest.approx(math.sqrt(2))
    assert gain_de_l_etage == pytest.approx(3.8586, abs=5e-5)
    symbolique = transfert_passe_bande((R1, R1, R1), (1, 1), MU, S)
    transfert = symbolique.subs({R1: sp.sqrt(2), MU: 4 - sp.sqrt(2) / 10})
    assert sp.simplify(sp.denom(sp.cancel(transfert)) / (10 * S**2 + S + 10)).is_number
    au_centre = abs(complex(transfert.subs(S, 1j)))
    assert au_centre == pytest.approx(gain_de_l_etage * 10 / math.sqrt(2))
    assert au_centre == pytest.approx(27.28, abs=5e-3)
    assert abs(au_centre - 27.3551) > 0.05


def test_le_coupe_bande_en_double_t() -> None:
    """Planche 42, a `alpha = 1` : zero en `1 / RC`, `Q = 1 / (4 (1 - mu))`."""
    resistance, capacite, mu = 2, sp.Rational(1, 2), sp.Rational(9, 10)
    transfert = transfert_coupe_bande(resistance, capacite, mu, S)
    assert transfert.subs(S, sp.I) == 0
    assert sp.simplify(transfert.subs(S, 0) - mu) == 0
    qualite = 1 / (4 * (1 - mu))
    attendu = mu * (S**2 + 1) / (S**2 + S / qualite + 1)
    assert sp.simplify(transfert - attendu) == 0
    partiel = transfert_coupe_bande(
        resistance, capacite, mu, S, alpha=sp.Rational(1, 4)
    )
    assert partiel.subs(S, 2 * sp.I) == 0
    assert sp.simplify(partiel.subs(S, 0) - mu) == 0


def test_le_coupe_bande_pour_tout_alpha() -> None:
    """Planche 42, formule entiere : le zero recule en `1 / racine(alpha)`."""
    r, c, alpha = sp.symbols("R C alpha", positive=True)
    planche = (
        MU
        * alpha
        * (S**2 + 1 / (alpha * r**2 * c**2))
        / (S**2 + 4 * (1 - MU) / (r * c) * S + 1 / (r**2 * c**2))
    )
    assert sp.simplify(transfert_coupe_bande(r, c, MU, S, alpha=alpha) - planche) == 0


def test_un_gain_trop_grand_fait_osciller_la_cellule() -> None:
    """Planche 24 : `mu = 3 - 1/Q` ; a `mu = 3`, le terme en `s` s'annule."""
    limite = SallenKeyPasseBas(1, 1, 1, 1, 3)
    assert limite.amortissement() == 0
    assert SallenKeyPasseBas(1, 1, 1, 1, 2).amortissement() == 1
    assert SallenKeyPasseBas(1, 1, 1, 1, 4).amortissement() == -1
    assert SallenKeyPasseBas(2, 3, 5, 7, 1).amortissement() == sp.Rational(1, 6)


@pytest.mark.parametrize("rang", range(5))
def test_la_cellule_refuse_un_composant_non_positif(rang: int) -> None:
    """Une resistance, une capacite ou un gain nul n'est pas un montage."""
    for fautif in (0, -1):
        valeurs = [1, 1, 1, 1, 1]
        valeurs[rang] = fautif
        with pytest.raises(ValueError, match="nul ou negatif"):
            SallenKeyPasseBas(*valeurs)


def test_la_denormalisation_refuse_un_facteur_non_positif() -> None:
    """Ni la coupure ni le facteur d'echelle ne peuvent etre nuls."""
    cellule = SallenKeyPasseBas(1, 1, 1, 1, 1)
    for coupure, impedance in ((0, 1), (1, 0), (-1, 1), (1, -1)):
        with pytest.raises(ValueError, match="non positif"):
            cellule.denormalisee(coupure=coupure, impedance=impedance)
    reelle = cellule.denormalisee(coupure=4, impedance=5)
    vingtieme = sp.Rational(1, 20)
    assert (reelle.r1, reelle.r2, reelle.c1, reelle.c2) == (5, 5, vingtieme, vingtieme)


def test_les_gardes_des_dimensionnements() -> None:
    """Chaque dimensionnement nomme la grandeur qu'il refuse."""
    with pytest.raises(ValueError, match="resistance vue nulle"):
        resistances_de_gain(2.0, 0.0)
    assert resistances_de_gain(1.5, 1.0) == pytest.approx((3.0, 1.5))
    for b0, b1 in ((0.0, 1.0), (1.0, 0.0)):
        with pytest.raises(ValueError, match="coefficient nul ou negatif"):
            resistances_du_passe_haut(b0, b1, 2.0)
    for qualite, pulsation in ((0.0, 1.0), (1.0, 0.0)):
        with pytest.raises(ValueError, match="non positif"):
            passe_bande_a_elements_egaux(qualite, pulsation)
    assert passe_bande_a_elements_egaux(2.0, 4.0) == pytest.approx(
        (math.sqrt(2) / 4, 4 - math.sqrt(2) / 2)
    )
