"""La note d'electronique : fonctions de transfert et plan de Bode.

Source : `Slides_Note_Fonctions_de_transfert_et_plan_de_Bode.pdf`, dans
`ElectroniqueInstrumentation/SLIDES/`, neuf pages lues sur leur couche de
texte ; la page 5 sur son rendu.

Les oracles : le circuit RC passe-bas (page 1), les proprietes de l'axe
logarithmique (page 2), les six expressions approchees de la page 5 et
son exemple trace -- zeros en 1e2 et 1e8, poles en 1e4, 1e6 et 1e10 --,
le pole double de la page 7 et le circuit RLC des pages 7 a 9.

Un ecart, lu sur la page rendue : la cinquieme expression de la page 5,
« pour z_1 << w << p_2, H = A_0 p_0 p_1 p_2 / (-w^2 z_0 z_1) », est fausse.
Entre `z_1` et `p_2` deux zeros et deux poles ont agi : `H = A_0 p_0 p_1 /
(z_0 z_1)`, un palier -- celui que la figure de la meme page trace a 1.
La page 4 annonce d'ailleurs l'ordre « p_2 < z_1 » et decoupe ensuite ses
intervalles avec `z_1 < p_2`.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.bode_asymptotique import (
    phase_asymptotique,
    pic_du_second_ordre,
)
from polytech.electronique.domain.laplace import (
    diviseur_de_tension,
    impedance_de_bobine,
    impedance_de_condensateur,
    impedance_de_resistance,
)
from polytech.electronique.domain.second_ordre import PasseBasCanonique
from polytech.signaux.domain.bode_par_intervalles import (
    Factorisation,
    distance_logarithmique,
    intervalles,
    parametres_du_rlc,
)

P = sp.Symbol("p")
W = sp.Symbol("omega", positive=True)
R, L, C = sp.symbols("R L C", positive=True)
A0, Z0, Z1, P0, P1, P2 = sp.symbols("A_0 z_0 z_1 p_0 p_1 p_2", positive=True)

LITTERALE = Factorisation(A0, zeros=(Z0, Z1), poles=(P0, P1, P2))
ORDRE = (Z0, P0, P1, Z1, P2)
"""Page 4 : l'ordre que les intervalles de la note emploient."""

ZEROS, POLES = (1e2, 1e8), (1e4, 1e6, 1e10)
EXEMPLE = Factorisation(sp.Integer(1), zeros=ZEROS, poles=POLES)
"""Page 5 : l'exemple trace."""

MILIEUX = (1e1, 1e3, 1e5, 1e7, 1e9, 1e11)


def test_la_fonction_de_transfert_du_rc_passe_bas() -> None:
    """Page 1 : `V_out = V_in / (1 + p R C)`, et `p -> j w` pour Fourier."""
    transfert = diviseur_de_tension(
        impedance_de_resistance(R, P), impedance_de_condensateur(C, P)
    )
    assert sp.simplify(transfert - 1 / (1 + P * R * C)) == 0
    fourier = transfert.subs(P, sp.I * W)
    assert sp.simplify(fourier - 1 / (1 + sp.I * W * R * C)) == 0
    t = sp.Symbol("t", positive=True)
    sortie = 1 - sp.exp(-t / (R * C))
    assert sp.simplify(R * C * sp.diff(sortie, t) + sortie - 1) == 0


def test_l_axe_logarithmique_mesure_des_rapports() -> None:
    """Page 2 : la distance de 0,1 a 1 vaut celle de 1 a 10, et 1-2, 2-4, 4-8."""
    decade = distance_logarithmique(1.0, 10.0)
    assert distance_logarithmique(0.1, 1.0) == pytest.approx(decade)
    octave = distance_logarithmique(1.0, 2.0)
    assert distance_logarithmique(2.0, 4.0) == pytest.approx(octave)
    assert distance_logarithmique(4.0, 8.0) == pytest.approx(octave)
    assert distance_logarithmique(1.0, 0.2) == pytest.approx(-0.699, abs=5e-4)
    with pytest.raises(ValueError, match="un axe logarithmique n'a pas de zero"):
        distance_logarithmique(0.0, 1.0)


def test_les_cassures_decoupent_l_axe_en_intervalles() -> None:
    """Page 4 : chaque zero et chaque pole est une frontiere ; ici six intervalles."""
    assert intervalles(ZEROS, POLES) == (1e2, 1e4, 1e6, 1e8, 1e10)
    assert len(intervalles(ZEROS, POLES)) + 1 == len(MILIEUX)


def test_les_expressions_approchees_de_la_page_5() -> None:
    """Page 5 : `A_0`, puis `A_0 j w / z_0`, `A_0 p_0 / z_0`, et les suivantes."""
    jw = sp.I * W
    attendues = {
        0: A0,
        1: A0 * jw / Z0,
        2: A0 * P0 / Z0,
        3: A0 * P0 * P1 / (jw * Z0),
        5: A0 * P0 * P1 * P2 / (jw * Z0 * Z1),
    }
    for rang, attendue in attendues.items():
        approchee = LITTERALE.approchee(ORDRE, rang, jw)
        assert sp.simplify(approchee - attendue) == 0
    with pytest.raises(ValueError, match="l'ordre doit nommer"):
        LITTERALE.approchee((Z0, P0), 1, jw)
    with pytest.raises(ValueError, match="rang hors de"):
        LITTERALE.approchee(ORDRE, 6, jw)


def test_la_cinquieme_expression_de_la_page_5_est_un_palier() -> None:
    """Entre `z_1` et `p_2` : `A_0 p_0 p_1 / (z_0 z_1)`, et non la forme imprimee."""
    palier = LITTERALE.approchee(ORDRE, 4, sp.I * W)
    assert sp.simplify(palier - A0 * P0 * P1 / (Z0 * Z1)) == 0
    imprimee = A0 * P0 * P1 * P2 / (-(W**2) * Z0 * Z1)
    assert sp.simplify(palier - imprimee) != 0
    valeurs = {A0: 1, Z0: 1e2, Z1: 1e8, P0: 1e4, P1: 1e6, P2: 1e10, W: 1e9}
    assert float(palier.subs(valeurs)) == pytest.approx(1.0)
    assert abs(complex(imprimee.subs(valeurs))) == pytest.approx(1e20 / (1e18 * 1e10))
    au_milieu = abs(complex(EXEMPLE.exacte(P).subs(P, 1e9j)))
    assert au_milieu == pytest.approx(1.0, rel=0.02)


def test_le_trace_reel_colle_au_trace_asymptotique_loin_des_cassures() -> None:
    """Page 5 : le module de l'exemple, au milieu de chaque intervalle."""
    ordre = intervalles(ZEROS, POLES)
    for rang, pulsation in enumerate(MILIEUX):
        approchee = EXEMPLE.approchee(ordre, rang, P).subs(P, 1j * pulsation)
        exacte = EXEMPLE.exacte(P).subs(P, 1j * pulsation)
        assert abs(complex(exacte)) == pytest.approx(abs(complex(approchee)), rel=0.03)


def test_un_zero_releve_la_pente_et_un_pole_l_abaisse() -> None:
    """Pages 5 et 6 : `H ~ k (j w)^n`, pente `n` et phase `n` fois 90 degres."""
    ordre = intervalles(ZEROS, POLES)
    pentes = [EXEMPLE.pente(ordre, rang) for rang in range(len(MILIEUX))]
    assert pentes == [0, 1, 0, -1, 0, -1]
    for rang, pulsation in enumerate(MILIEUX):
        approchee = EXEMPLE.approchee(ordre, rang, P)
        assert sp.simplify(approchee / P ** pentes[rang]).is_constant()
        phase = phase_asymptotique(pulsation, zeros=ZEROS, poles=POLES)
        assert phase == 90.0 * pentes[rang]


def test_un_pole_double_compte_double() -> None:
    """Page 7 : `1 / (1 + j w / 1000)^2`, pente -2 et phase de -180 degres."""
    double = Factorisation(sp.Integer(1), zeros=(), poles=(1e3, 1e3))
    approchee = double.approchee((1e3, 1e3), 2, P)
    assert sp.simplify(approchee - 1e6 / P**2) == 0
    assert double.pente((1e3, 1e3), 2) == -2
    assert phase_asymptotique(1e5, zeros=(), poles=(1e3, 1e3)) == -180.0
    exacte = abs(complex(double.exacte(P).subs(P, 1e5j)))
    assert exacte == pytest.approx(abs(complex(approchee.subs(P, 1e5j))), rel=1e-3)


def test_la_forme_canonique_du_rlc() -> None:
    """Page 7 : `p_0 = 1 / racine(L C)`, `zeta = (R / 2) racine(C / L)`."""
    transfert = diviseur_de_tension(
        impedance_de_resistance(R, P) + impedance_de_bobine(L, P),
        impedance_de_condensateur(C, P),
    )
    assert sp.simplify(transfert - 1 / (1 + P * R * C + P**2 * L * C)) == 0
    propre, amortissement = parametres_du_rlc(R, L, C)
    assert propre == 1 / sp.sqrt(L * C)
    assert sp.simplify(amortissement - R / 2 * sp.sqrt(C / L)) == 0
    canonique = 1 / (1 + 2 * amortissement * P / propre + (P / propre) ** 2)
    assert sp.simplify(transfert - canonique) == 0
    lie = PasseBasCanonique(sp.Integer(1), propre, 1 / (2 * amortissement))
    assert sp.simplify(lie.transfert(P) - canonique) == 0


def _canonique(amortissement: sp.Expr) -> PasseBasCanonique:
    """`1 / (1 + 2 zeta p / p_0 + (p / p_0)^2)` pour `p_0 = 1000` : `Q = 1 / 2 zeta`."""
    qualite = 1 / (2 * sp.sympify(amortissement))
    return PasseBasCanonique(sp.Integer(1), sp.Integer(1000), qualite)


def test_les_poles_du_second_ordre_selon_l_amortissement() -> None:
    """Page 8 : `p_0 (-zeta +- racine(zeta^2 - 1))` ; reels, double, ou conjugues."""
    for amortissement, reels in ((2, True), (1, True), (sp.Rational(1, 2), False)):
        racine = sp.sqrt(sp.sympify(amortissement) ** 2 - 1)
        attendus = {
            sp.simplify(1000 * (-amortissement + signe * racine)) for signe in (-1, 1)
        }
        filtre = _canonique(amortissement)
        assert {sp.simplify(pole) for pole in filtre.poles()} == attendus
        assert filtre.poles_sont_reels() is reels
    conjugues = _canonique(sp.Rational(1, 2)).poles()
    assert {sp.Abs(pole) for pole in conjugues} == {1000}


def test_la_resonance_a_la_pulsation_propre() -> None:
    """Pages 8 et 9 : `H(j w_0) = 1 / (2 j zeta)` ; 1/2 au pole double."""
    for amortissement in (sp.Rational(1, 2), sp.Rational(1, 10), sp.Rational(1, 100)):
        a_la_resonance = _canonique(amortissement).reponse_a_la_pulsation_propre()
        assert sp.simplify(a_la_resonance - 1 / (2 * sp.I * amortissement)) == 0
        assert sp.arg(a_la_resonance) == -sp.pi / 2
        en_decibels = 20 * math.log10(float(sp.Abs(a_la_resonance)))
        assert en_decibels == pytest.approx(pic_du_second_ordre(float(amortissement)))
    double = _canonique(1).reponse_a_la_pulsation_propre()
    assert sp.Abs(double) == sp.Rational(1, 2)


def test_une_paire_conjuguee_vaut_un_pole_double_loin_d_elle() -> None:
    """Page 8 : `H ~ 1` sous `w_0`, `H ~ -(w_0 / w)^2` au-dessus."""
    filtre = _canonique(sp.Rational(1, 10))
    basse = complex(filtre.transfert(P).subs(P, 10j))
    haute = complex(filtre.transfert(P).subs(P, 1e5j))
    assert basse == pytest.approx(1.0, abs=3e-3)
    assert haute == pytest.approx(-((1000 / 1e5) ** 2), rel=3e-3)
