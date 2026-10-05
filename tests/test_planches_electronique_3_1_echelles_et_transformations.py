"""Les planches d'electronique : echelles LC et transformations (chapitre 3, I).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_3_part1.pdf`, planches
59 a 132, lues sur leur couche de texte et, pour les tables de
conversion, sur les pages rendues.

Les oracles : les deux tables de coefficients `g_i` (planches 67 et 68),
la preuve de la mise a l'echelle d'impedance (planches 78 a 84), l'exemple
du passe-haut a 100 rad/s (planches 101 a 108) et les trois tables de
conversion de composants (planches 110, 125 et 131).

Un ecart, lu sur la page rendue : la planche 69 ecrit `g_(N+1) =
coth(beta / 4)` aux ordres pairs, quand sa propre table (1,3554) est le
carre de cette valeur. Et la planche 124 ecrit « L omega_c / B » pour
l'inductance du passe-bande : c'est `L Omega_c / B`, que la table 125 donne.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.abcd import impedance_d_entree
from polytech.electronique.domain.butterworth import module_de_butterworth
from polytech.electronique.domain.chebyshev import (
    charge_de_chebyshev,
    elements_de_chebyshev,
)
from polytech.electronique.domain.conversions_d_elements import (
    Branche,
    changer_la_coupure,
    en_coupe_bande,
    en_passe_bande,
    en_passe_haut,
)
from polytech.electronique.domain.echelle_lc import (
    Topologie,
    echelle,
    elements_de_butterworth,
    transfert_de_l_echelle,
)
from polytech.electronique.domain.netliste import (
    Admittance,
    SourceDeTension,
    resoudre,
)
from polytech.electronique.domain.passe_bande import (
    bords_du_passe_bande,
    facteur_de_qualite_de_bande,
    poles_du_passe_bande,
)
from polytech.electronique.domain.pole_zero import carte_des_poles_et_zeros
from polytech.electronique.domain.specifications_de_filtre import perte_d_insertion
from polytech.electronique.domain.transpositions import (
    mise_a_l_echelle_d_impedance,
    vers_coupe_bande,
    vers_passe_bande,
    vers_passe_haut,
)

S = sp.Symbol("s")

BUTTERWORTH = {
    1: (2.000,),
    2: (1.4142, 1.4142),
    3: (1.0000, 2.0000, 1.0000),
    4: (0.7654, 1.8478, 1.8478, 0.7654),
    5: (0.6180, 1.6180, 2.000, 1.6180, 0.6180),
}
"""Planche 67 : la table des `g_i` de Butterworth."""

CHEBYSHEV = {
    1: ((0.3052,), 1.0),
    2: ((0.8431, 0.6220), 1.3554),
    3: ((1.0316, 1.1474, 1.0316), 1.0),
    4: ((1.1088, 1.3062, 1.7704, 0.8181), 1.3554),
    5: ((1.1468, 1.3712, 1.9750, 1.3712, 1.1468), 1.0),
}
"""Planche 68 : les `g_i` de Chebyshev pour `L_Ar = 0,1 dB`, puis `g_(N+1)`."""


def test_la_table_de_butterworth() -> None:
    """Planche 67 : `g_i = 2 sin((2i - 1) pi / 2N)`, symetriques."""
    for ordre, ligne in BUTTERWORTH.items():
        elements = elements_de_butterworth(ordre)
        assert elements == pytest.approx(ligne, abs=5e-5)
        assert elements == pytest.approx(elements[::-1])
    with pytest.raises(ValueError, match="ordre inferieur a un"):
        elements_de_butterworth(0)


def test_la_table_de_chebyshev() -> None:
    """Planches 68 et 69 : la recurrence redonne la table, a 1e-4 pres."""
    for ordre, (ligne, charge) in CHEBYSHEV.items():
        assert elements_de_chebyshev(0.1, ordre) == pytest.approx(ligne, abs=1.5e-4)
        assert charge_de_chebyshev(0.1, ordre) == pytest.approx(charge, abs=5e-5)


def test_la_planche_69_oublie_le_carre_de_la_cotangente() -> None:
    """`coth(beta / 4)` vaut 1,1642 ; la table porte 1,3554, son carre."""
    beta = math.log(1 / math.tanh(0.1 / 17.37))
    imprimee = 1 / math.tanh(beta / 4)
    assert imprimee == pytest.approx(1.1642, abs=5e-5)
    assert imprimee**2 == pytest.approx(CHEBYSHEV[2][1], abs=5e-5)


@pytest.mark.parametrize("topologie", list(Topologie))
def test_l_echelle_de_butterworth_realise_le_module_de_butterworth(
    topologie: Topologie,
) -> None:
    """Planches 62 a 65 : les deux topologies, 3,01 dB a 1 rad/s."""
    for ordre in (2, 3, 4):
        elements = elements_de_butterworth(ordre)
        transfert = transfert_de_l_echelle(elements, S, topologie=topologie)
        for pulsation in (0.0, 0.5, 1.0, 3.0):
            module = abs(complex(transfert.subs(S, 1j * pulsation)))
            assert module == pytest.approx(module_de_butterworth(pulsation, ordre) / 2)
        a_la_coupure = 2 * abs(complex(transfert.subs(S, 1j)))
        assert perte_d_insertion(a_la_coupure) == pytest.approx(3.01, abs=5e-3)


def _extremes(topologie: Topologie, charge: float) -> tuple[float, float]:
    """Le creux et le sommet du module dans la bande, rapportes au continu."""
    elements = elements_de_chebyshev(0.1, 2)
    transfert = transfert_de_l_echelle(elements, S, topologie=topologie, charge=charge)
    continu = abs(complex(transfert.subs(S, 0)))
    modules = [abs(complex(transfert.subs(S, 1j * k / 200))) for k in range(201)]
    return min(modules) / continu, max(modules) / continu


def test_la_charge_d_un_chebyshev_pair_est_une_resistance_ou_une_conductance() -> None:
    """Planche 64 : resistance apres une capacite, conductance apres une inductance.

    A l'ordre deux, 0,1 dB d'ondulation : le module part du creux et monte
    de 0,1 dB. Sur une charge de 1 ohm, « the filter does not work ».
    """
    charge = charge_de_chebyshev(0.1, 2)
    ondulation = 10 ** (0.1 / 20)
    resistance = _extremes(Topologie.INDUCTANCE_D_ABORD, charge)
    assert resistance == pytest.approx((1.0, ondulation), abs=1e-4)
    conductance = _extremes(Topologie.CAPACITE_D_ABORD, 1 / charge)
    assert conductance == pytest.approx((1.0, ondulation), abs=1e-4)
    for topologie in Topologie:
        creux, sommet = _extremes(topologie, 1.0)
        assert sommet == pytest.approx(1.0, abs=1e-4)
        assert creux < 0.97


def test_la_mise_a_l_echelle_d_impedance_ne_touche_pas_aux_tensions() -> None:
    """Planches 73 a 84 : `V' = V`, `I_S' = I_S / gamma_0`, `Z_in' = gamma_0 Z_in`."""
    gamma = sp.Integer(50)
    mis = mise_a_l_echelle_d_impedance(50.0, 1.0, 2.0, 1.0)
    assert mis == pytest.approx((50.0, 100.0, 0.02))

    def circuit(facteur: sp.Expr) -> tuple[sp.Expr, ...]:
        elements = (
            SourceDeTension(1, 0, sp.Integer(1), sp.Symbol("I_S")),
            Admittance(1, 2, 1 / facteur),
            Admittance(2, 0, S / facteur),
            Admittance(2, 3, 1 / (2 * S * facteur)),
            Admittance(3, 0, S / facteur),
            Admittance(3, 0, 1 / facteur),
        )
        return tuple(resoudre(elements, 3))

    avant, apres = circuit(sp.Integer(1)), circuit(gamma)
    for noeud in range(3):
        assert sp.simplify(avant[noeud] - apres[noeud]) == 0
    assert sp.simplify(apres[3] - avant[3] / gamma) == 0
    butterworth = sp.Rational(1, 2) / (S**3 + 2 * S**2 + 2 * S + 1)
    assert sp.simplify(avant[2] - butterworth) == 0
    reference = impedance_d_entree(echelle(BUTTERWORTH[3], S), sp.Integer(1))
    mise = [50 * g if rang % 2 else g / 50 for rang, g in enumerate(BUTTERWORTH[3])]
    echelonnee = impedance_d_entree(echelle(tuple(mise), S), sp.Integer(50))
    assert sp.simplify(echelonnee - 50 * reference) == 0


def test_la_transformation_passe_bas() -> None:
    """Planches 89 a 93 : `L, C -> (Omega_c / w_c) L, C` equivaut a `s -> s / w_c`."""
    coupure = 100.0
    assert changer_la_coupure(Branche(inductance=2.0), coupure) == Branche(
        inductance=0.02
    )
    assert changer_la_coupure(Branche(capacite=1.0), coupure) == Branche(capacite=0.01)
    prototype = transfert_de_l_echelle(BUTTERWORTH[3], S)
    deplaces = tuple(g / 100 for g in BUTTERWORTH[3])
    deplace = transfert_de_l_echelle(deplaces, S)
    assert sp.simplify(deplace - prototype.subs(S, S / 100)) == 0
    poles, _ = carte_des_poles_et_zeros(deplace, S)
    origine, _ = carte_des_poles_et_zeros(prototype, S)
    assert sorted(abs(complex(p)) for p in poles) == pytest.approx(
        sorted(100 * abs(complex(p)) for p in origine)
    )
    assert deplace.subs(S, 0) == prototype.subs(S, 0)


def test_la_table_du_passe_haut() -> None:
    """Planche 110 : `C -> L = 1 / (w_c Omega_c C)`, `L -> C = 1 / (w_c Omega_c L)`."""
    coupure = 100.0
    nouvelle = en_passe_haut(Branche(capacite=2.0), coupure)
    assert nouvelle == Branche(inductance=1 / 200)
    assert en_passe_haut(Branche(inductance=4.0), coupure) == Branche(capacite=1 / 400)
    ancienne = Branche(capacite=2.0).impedance(coupure / S)
    assert sp.simplify(nouvelle.impedance(S) - ancienne) == 0


def test_l_exemple_du_passe_haut_a_cent_radians_par_seconde() -> None:
    """Planches 101 a 108 : `T_2 -> T_3 -> T_4`, zeros en `25 z`, poles en `100 p`."""
    poles = [sp.exp(2 * sp.I * sp.pi / 3), sp.Integer(-1), sp.exp(4 * sp.I * sp.pi / 3)]
    zeros = [2 * sp.I, -2 * sp.I]
    gain = sp.prod(poles) / sp.prod(zeros)
    t2 = gain * sp.prod([S - z for z in zeros]) / sp.prod([S - p for p in poles])
    assert sp.simplify(sp.Abs(t2.subs(S, 0))) == 1
    t3 = vers_passe_haut(t2, S)
    planche_3 = -S * (S - zeros[1] / 4) * (S - zeros[0] / 4)
    planche_3 /= sp.prod([S - p for p in poles])
    assert sp.simplify(t3 - planche_3) == 0
    t4 = sp.cancel(t3.subs(S, S / 100))
    planche_4 = -S * (S - 25 * zeros[1]) * (S - 25 * zeros[0])
    planche_4 /= sp.prod([S - 100 * p for p in poles])
    assert sp.simplify(t4 - planche_4) == 0
    assert abs(complex(t4.subs(S, 1e7j))) == pytest.approx(1.0, abs=1e-6)
    assert t4.subs(S, 0) == 0


def test_les_bords_du_passe_bande() -> None:
    """Planches 116 et 117 : `w_2' - w_1' = B` et `racine(w_1' w_2') = w_0`."""
    bas, haut = bords_du_passe_bande(1000.0, 200.0)
    assert haut - bas == pytest.approx(200.0)
    assert math.sqrt(bas * haut) == pytest.approx(1000.0)
    assert (bas + haut) / 2 > 1000.0
    for bord in (bas, haut):
        image = (1000.0 / 200.0) * (bord / 1000.0 - 1000.0 / bord)
        assert abs(image) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="largeur de bande nulle ou negative"):
        bords_du_passe_bande(1000.0, 0.0)


def _imaginaire(point: complex) -> float:
    return point.imag


def test_le_passe_bande_dedouble_chaque_pole() -> None:
    """Planches 118 a 121 : `s ~ (B / 2) A +- j w_0` des que `Q > 5`."""
    pole = complex(-0.5, 0.8)
    exacts = sorted(poles_du_passe_bande(pole, 1000.0, 200.0), key=_imaginaire)
    for racine in exacts:
        residu = racine**2 + 1000.0**2 - pole * 200.0 * racine
        assert residu == pytest.approx(0, abs=1e-6)
    for exact, signe in zip(exacts, (-1, 1), strict=True):
        assert abs(exact - (100 * pole + signe * 1000j)) < 0.01 * 1000.0
    normalise = poles_du_passe_bande(2 * pole, 1000.0, 200.0, prototype=2.0)
    assert sorted(normalise, key=_imaginaire) == pytest.approx(exacts)
    assert facteur_de_qualite_de_bande(1000.0, 200.0) == 5.0
    assert facteur_de_qualite_de_bande(250.0, 200.0) == 1.25
    bas_q = sorted(poles_du_passe_bande(pole, 250.0, 200.0), key=_imaginaire)
    ecarts = [
        abs(exact - (100 * pole + signe * 250j))
        for exact, signe in zip(bas_q, (-1, 1), strict=True)
    ]
    assert max(ecarts) > 0.05 * 250.0


def test_la_table_du_passe_bande() -> None:
    """Planche 125 : `C` devient `L // C`, `L` devient `L` en serie avec `C`."""
    centre, largeur = 1000.0, 200.0
    shunt = en_passe_bande(Branche(capacite=2.0), centre, largeur)
    assert shunt == Branche(
        inductance=largeur / (centre**2 * 2.0), capacite=2.0 / largeur, serie=False
    )
    serie = en_passe_bande(Branche(inductance=4.0), centre, largeur)
    assert serie == Branche(
        inductance=4.0 / largeur, capacite=largeur / (centre**2 * 4.0), serie=True
    )
    image = (S**2 + centre**2) / (largeur * S)
    assert sp.simplify(shunt.impedance(S) - Branche(capacite=2.0).impedance(image)) == 0
    assert (
        sp.simplify(serie.impedance(S) - Branche(inductance=4.0).impedance(image)) == 0
    )
    prototype = 1 / (S + 1)
    bande = vers_passe_bande(prototype, S, sp.Integer(1000), sp.Integer(200))
    assert bande.subs(S, 1000 * sp.I) == prototype.subs(S, 0)
    assert bande.subs(S, 0) == 0


def test_la_table_du_coupe_bande() -> None:
    """Planche 131 : `C` devient `L` en serie avec `C`, `L` devient `L // C`."""
    centre, largeur = 1000.0, 200.0
    shunt = en_coupe_bande(Branche(capacite=2.0), centre, largeur)
    assert shunt == Branche(
        inductance=1 / (largeur * 2.0), capacite=largeur * 2.0 / centre**2, serie=True
    )
    serie = en_coupe_bande(Branche(inductance=4.0), centre, largeur)
    assert serie == Branche(
        inductance=largeur * 4.0 / centre**2, capacite=1 / (largeur * 4.0), serie=False
    )
    image = largeur * S / (S**2 + centre**2)
    assert sp.simplify(shunt.impedance(S) - Branche(capacite=2.0).impedance(image)) == 0
    assert (
        sp.simplify(serie.impedance(S) - Branche(inductance=4.0).impedance(image)) == 0
    )
    prototype = 1 / (S + 1)
    coupe = vers_coupe_bande(prototype, S, sp.Integer(1000), sp.Integer(200))
    assert coupe.subs(S, 1000 * sp.I) == 0
    assert coupe.subs(S, 0) == prototype.subs(S, 0)


def test_une_branche_refuse_d_etre_vide() -> None:
    with pytest.raises(ValueError, match="branche sans element"):
        Branche()
    with pytest.raises(ValueError, match="un seul element reactif"):
        en_passe_haut(Branche(inductance=1.0, capacite=1.0, serie=True), 100.0)


def test_les_tables_valent_pour_un_prototype_qui_ne_coupe_pas_a_un() -> None:
    """Planches 110, 125 et 131 : `Omega_c` y figure, et compte s'il ne vaut pas 1."""
    centre, largeur, omega = 1000.0, 200.0, 2.0
    images = {
        "bas": omega * S / 100,
        "haut": omega * 100 / S,
        "bande": omega * (S**2 + centre**2) / (largeur * S),
        "coupe": omega * largeur * S / (S**2 + centre**2),
    }
    for ancienne in (Branche(capacite=2.0), Branche(inductance=4.0)):
        nouvelles = {
            "bas": changer_la_coupure(ancienne, 100.0, prototype=omega),
            "haut": en_passe_haut(ancienne, 100.0, prototype=omega),
            "bande": en_passe_bande(ancienne, centre, largeur, prototype=omega),
            "coupe": en_coupe_bande(ancienne, centre, largeur, prototype=omega),
        }
        for nom, nouvelle in nouvelles.items():
            ecart = nouvelle.impedance(S) - ancienne.impedance(images[nom])
            assert sp.simplify(ecart) == 0, nom


def test_les_gardes_des_echelles_et_des_bandes() -> None:
    """Une echelle vide, une charge nulle, un element negatif, un centre nul."""
    with pytest.raises(ValueError, match="echelle vide"):
        echelle((), S)
    with pytest.raises(ValueError, match="terminaisons invalides"):
        transfert_de_l_echelle(BUTTERWORTH[3], S, charge=0.0)
    with pytest.raises(ValueError, match="terminaisons invalides"):
        transfert_de_l_echelle(BUTTERWORTH[3], S, source=-1.0)
    with pytest.raises(ValueError, match="ondulation non positive"):
        charge_de_chebyshev(0.0, 2)
    with pytest.raises(ValueError, match="ordre inferieur a un"):
        charge_de_chebyshev(0.1, 0)
    with pytest.raises(ValueError, match="element nul ou negatif"):
        Branche(inductance=-1.0)
    with pytest.raises(ValueError, match="pulsation centrale nulle ou negative"):
        bords_du_passe_bande(0.0, 200.0)
