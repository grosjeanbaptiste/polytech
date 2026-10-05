"""Les planches d'electronique : l'amplificateur operationnel et la retroaction (2, I).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_2_part1.pdf`, 64
planches lues sur leur couche de texte et, pour les schemas, sur les
pages rendues.

Les oracles : les 50 uV de la planche 12, l'impedance de sortie de 1 ohm
(planche 23), les matrices `H` de l'amplificateur et de son reseau de
retour (planche 57), les grandeurs en boucle fermee (planches 45 a 48),
le `g_12 = -1e-8` de la planche 48 et l'encadrement de la planche 52.
Chaque montage est aussi resolu par l'analyse nodale, sans masse
virtuelle : c'est le second chemin.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.amplification import (
    gain_inverseur,
    gain_non_inverseur,
)
from polytech.electronique.domain.aop import saturer
from polytech.electronique.domain.aop_reel import (
    ecart_differentiel_maximal,
    impedance_d_entree_chargee,
    impedance_de_sortie_du_non_inverseur,
)
from polytech.electronique.domain.boucle_fermee import (
    coefficient_de_retour,
    gain_en_boucle_fermee,
    gain_vu_de_la_charge,
    gain_vu_de_la_source,
    impedance_d_entree_en_boucle_fermee,
    impedance_de_sortie_en_boucle_fermee,
    retroaction_efficace,
)
from polytech.electronique.domain.conversions_de_quadripole import (
    Jeu,
    associer,
    convertir,
)
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeTension,
    SourceDeTensionCommandee,
    resoudre,
)
from polytech.electronique.domain.retroaction import (
    parametres_h_amplificateur,
    parametres_h_reseau_resistif,
)

RIN, RO, AV = 1e6, 100.0, 1e4
ENTREE, MOINS, SORTIE, INTERNE = 1, 2, 3, 4


def _amplificateur(plus: int, moins: int, gain: float = AV) -> tuple[Element, ...]:
    """L'AOP reel : `R_in` entre ses entrees, `A_v (V+ - V-)` derriere `R_o`."""
    commandee = SourceDeTensionCommandee(
        INTERNE, 0, sp.Float(gain), (plus, moins), sp.Symbol("i_a")
    )
    interne = (commandee, Admittance(INTERNE, SORTIE, sp.Float(1 / RO)))
    return (*interne, Admittance(plus, moins, sp.Float(1 / RIN)))


def _non_inverseur(r_1: float, r_2: float, tension: float) -> tuple[Element, ...]:
    return (
        SourceDeTension(ENTREE, 0, sp.Float(tension), sp.Symbol("i_e")),
        *_amplificateur(ENTREE, MOINS),
        Admittance(MOINS, 0, sp.Float(1 / r_1)),
        Admittance(MOINS, SORTIE, sp.Float(1 / r_2)),
    )


def test_la_saturation_borne_la_tension_differentielle() -> None:
    """Planche 12 : `|V+ - V-| <= V_pos / A_v`, soit 50 uV pour 5 V et `1e5`."""
    assert ecart_differentiel_maximal(5.0, 1e5) == pytest.approx(50e-6)
    assert saturer(1e5 * 40e-6, 5.0) == pytest.approx(4.0)
    assert saturer(1e5 * 80e-6, 5.0) == pytest.approx(5.0)
    assert saturer(-1e5 * 80e-6, 5.0) == pytest.approx(-5.0)
    with pytest.raises(ValueError, match="gain nul ou negatif"):
        ecart_differentiel_maximal(5.0, 0.0)


def test_le_gain_du_non_inverseur() -> None:
    """Planches 16 et 17 : `V_out = (1 + R_2 / R_1) V_in`, jamais sous l'unite."""
    assert gain_non_inverseur(1e3, 99e3) == pytest.approx(100.0)
    tensions = resoudre(_non_inverseur(1e3, 99e3, 0.01), 4)
    assert float(tensions[SORTIE - 1]) == pytest.approx(1.0, rel=0.011)
    ecart = float(tensions[ENTREE - 1] - tensions[MOINS - 1])
    assert abs(ecart) < ecart_differentiel_maximal(5.0, AV)


def _impedance_de_sortie(circuit: tuple[Element, ...]) -> float:
    """Une source d'essai a la sortie : `Z_out = V / I`, `I` entrant dans le montage."""
    essai = SourceDeTension(SORTIE, 0, sp.Float(1.0), sp.Symbol("i_t"))
    solution = resoudre((*circuit, essai), 4)
    return 1.0 / float(solution[-1])


def test_l_impedance_de_sortie_du_non_inverseur() -> None:
    """Planches 20 a 23 : `Z_out ~ R_o (1 + R_2 / R_1) / A_v`, soit 1 ohm.

    La planche 20 ecrit `I_OA = (V_out - A_v V_-) / R_o`. La source
    interne vaut `A_v (V+ - V-) = -A_v V_-` : le courant est `(V_out + A_v
    V_-) / R_o`. Avec le signe imprime, l'impedance serait negative ; la
    planche 23 conclut pourtant par la valeur positive, qui est la bonne.
    """
    circuit = (
        Admittance(ENTREE, 0, sp.Float(1e9)),
        *_amplificateur(ENTREE, MOINS),
        Admittance(MOINS, 0, sp.Float(1e-3)),
        Admittance(MOINS, SORTIE, sp.Float(1 / 99e3)),
    )
    nodale = _impedance_de_sortie(circuit)
    exacte = impedance_de_sortie_du_non_inverseur(
        RO, AV, vers_la_masse=1e3, contre_reaction=99e3, entree=RIN
    )
    assert nodale == pytest.approx(exacte, rel=1e-7)
    assert exacte == pytest.approx(RO * 100.0 / AV, rel=0.02)
    assert exacte == pytest.approx(1.0, abs=0.02)
    parallele = RIN * 1e3 / (RIN + 1e3)
    retour = parallele / (parallele + 99e3)
    imprimee = 1 / (1 / (99e3 + parallele) + 1 / RO - AV * retour / RO)
    assert imprimee < 0
    assert imprimee == pytest.approx(-exacte, rel=0.03)


def _inverseur(r_1: float, r_2: float, tension: float) -> tuple[Element, ...]:
    return (
        SourceDeTension(ENTREE, 0, sp.Float(tension), sp.Symbol("i_e")),
        *_amplificateur(0, MOINS),
        Admittance(ENTREE, MOINS, sp.Float(1 / r_1)),
        Admittance(MOINS, SORTIE, sp.Float(1 / r_2)),
    )


def test_le_gain_et_l_impedance_d_entree_de_l_inverseur() -> None:
    """Planches 25 a 27 : `V_out = -(R_2 / R_1) V_in`, `Z_in = R_1`."""
    assert gain_inverseur(1e3, 10e3) == pytest.approx(-10.0)
    solution = resoudre(_inverseur(1e3, 10e3, 0.1), 4)
    assert float(solution[SORTIE - 1]) == pytest.approx(-1.0, rel=2e-3)
    courant = float(solution[4])
    assert 0.1 / courant == pytest.approx(1e3, rel=2e-3)
    assert gain_inverseur(10e3, 1e3) == pytest.approx(-0.1)


def test_l_impedance_de_sortie_de_l_inverseur() -> None:
    """Planche 28 : « `R_o / (1 + A_v / A_cl)`, ou `A_cl ~ -R_2 / R_1` ».

    Prise au pied de la lettre, la formule est negative. La planche 60
    donne `R_o / (1 + beta A_v)`, avec `beta = R_1 / (R_1 + R_2)` : c'est
    ce que rend l'analyse nodale. `A_v / |A_cl|` en est proche quand le
    gain est grand, et differe de 10 % pour un gain de dix.
    """
    circuit = (
        Admittance(ENTREE, 0, sp.Float(1e9)),
        *_amplificateur(0, MOINS),
        Admittance(ENTREE, MOINS, sp.Float(1e-3)),
        Admittance(MOINS, SORTIE, sp.Float(1e-4)),
    )
    nodale = _impedance_de_sortie(circuit)
    retour = coefficient_de_retour(1e3, 10e3)
    assert nodale == pytest.approx(RO / (1 + retour * AV), rel=0.01)
    assert RO / (1 + AV / -10.0) < 0
    assert pytest.approx(nodale, rel=0.1) == RO / (1 + AV / 10.0)
    assert abs(RO / (1 + AV / 10.0) - nodale) > 0.05 * nodale


def test_le_suiveur_de_tension() -> None:
    """Planche 29 : gain unite, sortie a `R_o / (1 + A_v)`."""
    inutilise = Admittance(MOINS, 0, sp.Float(1.0))
    circuit = (
        SourceDeTension(ENTREE, 0, sp.Float(1.0), sp.Symbol("i_e")),
        *_amplificateur(ENTREE, SORTIE),
        inutilise,
    )
    assert float(resoudre(circuit, 4)[SORTIE - 1]) == pytest.approx(1.0, rel=2e-4)
    assert impedance_de_sortie_en_boucle_fermee(RO, AV, 1.0) == pytest.approx(
        RO / (1 + AV)
    )


R_IN, R_O, A_V, R_1, R_2 = sp.symbols("R_in R_o A_v R_1 R_2", positive=True)
H_AOP = parametres_h_amplificateur(R_IN, R_O, A_V)
H_RETOUR = parametres_h_reseau_resistif(R_1, R_2)
BETA = R_1 / (R_1 + R_2)


def test_les_matrices_h_de_l_amplificateur_et_du_reseau_de_retour() -> None:
    """Planche 57 : `H1 = [[R_in, 0], [-A_v R_in / R_o, 1 / R_o]]`, et `H2`."""
    attendue = sp.Matrix([[R_IN, 0], [-A_V * R_IN / R_O, 1 / R_O]])
    assert sp.simplify(H_AOP - attendue) == sp.zeros(2, 2)
    parallele = R_1 * R_2 / (R_1 + R_2)
    retour = sp.Matrix([[parallele, BETA], [-BETA, 1 / (R_1 + R_2)]])
    assert sp.simplify(H_RETOUR - retour) == sp.zeros(2, 2)
    assert sp.simplify(-(1 / R_2) / (1 / R_1 + 1 / R_2) + BETA) == 0


VALEURS = {R_IN: 1e6, R_O: 100.0, A_V: 1e4, R_1: 1e3, R_2: 99e3}


def test_la_retroaction_serie_parallele_additionne_les_matrices_h() -> None:
    """Planche 57 : seul `h_12 = beta` vient du reseau de retour, a 0,2 % pres."""
    globale = associer(Jeu.H, H_AOP, H_RETOUR).subs(VALEURS)
    seule = H_AOP.subs(VALEURS)
    assert float(globale[0, 0]) == pytest.approx(float(seule[0, 0]), rel=2e-3)
    assert float(globale[0, 1]) == pytest.approx(0.01)
    assert float(globale[1, 0]) == pytest.approx(float(seule[1, 0]), rel=1e-6)
    assert float(globale[1, 1]) == pytest.approx(float(seule[1, 1]), rel=2e-3)


def test_les_grandeurs_en_boucle_fermee_se_lisent_sur_la_matrice_g() -> None:
    """Planches 45 a 48 : `A_ol / (1 + beta A_ol)`, `Z_in (1 + beta A_ol)`, etc."""
    globale = associer(Jeu.H, H_AOP, H_RETOUR).subs(VALEURS)
    inverse = convertir(globale, depuis=Jeu.H, vers=Jeu.G)
    ouvert = float(-globale[1, 0] / (globale[0, 0] * globale[1, 1]))
    assert ouvert == pytest.approx(1e4, rel=3e-3)
    assert ouvert * 0.01 > 10
    assert not retroaction_efficace(ouvert, 0.01)
    assert retroaction_efficace(2 * ouvert, 0.01)
    assert float(inverse[1, 0]) == pytest.approx(gain_en_boucle_fermee(ouvert, 0.01))
    assert float(inverse[1, 0]) == pytest.approx(100.0, rel=0.011)
    entree = impedance_d_entree_en_boucle_fermee(float(globale[0, 0]), ouvert, 0.01)
    assert 1 / float(inverse[0, 0]) == pytest.approx(entree)
    sortie = impedance_de_sortie_en_boucle_fermee(
        1 / float(globale[1, 1]), ouvert, 0.01
    )
    assert float(inverse[1, 1]) == pytest.approx(sortie)
    assert sortie == pytest.approx(1.0, rel=0.02)
    assert entree == pytest.approx(1e6 * 1e4 / 100.0, rel=0.02)


def test_le_coefficient_de_retour_de_courant_est_negligeable() -> None:
    """Planche 48 : `g_12 ~ -(Z_out,ol / Z_in,ol) / A_ol`, soit `-1e-8`."""
    globale = associer(Jeu.H, H_AOP, H_RETOUR).subs(VALEURS)
    inverse = convertir(globale, depuis=Jeu.H, vers=Jeu.G)
    assert float(inverse[0, 1]) == pytest.approx(-1e-4 * 1e-4, rel=0.02)


def _entree_du_non_inverseur_charge(charge: float) -> float:
    """`V_in / I_in` du montage reel, une resistance `charge` a sa sortie."""
    circuit = (
        *_non_inverseur(1e3, 99e3, 1.0),
        Admittance(SORTIE, 0, sp.Float(1 / charge)),
    )
    return 1.0 / float(resoudre(circuit, 4)[4])


def test_la_charge_ramene_a_l_entree_ce_qu_elle_retire_a_la_boucle() -> None:
    """Planches 50 a 53 : `Z_in(R_L)`, par la matrice `G` et par l'analyse nodale.

    La planche 50 factorise `Z_in,cl^-1` en ecrivant le second terme
    `(Z_out,ol / R_L)(A_cl / A_ol)` : elle y prend `Z_in,cl / Z_in,ol`
    pour un, alors qu'il vaut `1 + beta A_ol`. Le terme exact est
    `(Z_out,ol / R_L) / (1 + Z_out,cl / R_L)`, sans le rapport des gains.

    La conclusion de la planche 52, « au plus 1 % », ne tient donc pas :
    1 kohm a la sortie retire deja 9 % a l'impedance d'entree, et un
    court-circuit, qui supprime la retroaction, la ramene a `Z_in,ol`.
    La formule imprimee elle-meme sort de sa borne : a 1 ohm elle rend
    `Z_in,cl / 1,5`, parce que la planche 51 majore par un le rapport
    `(Z_out,ol / R_L) / (1 + Z_out,cl / R_L)`, qui tend vers
    `Z_out,ol / Z_out,cl`.
    """
    globale = associer(Jeu.H, H_AOP, H_RETOUR).subs(VALEURS)
    inverse = convertir(globale, depuis=Jeu.H, vers=Jeu.G)
    a_vide, fermee = 1 / float(inverse[0, 0]), float(inverse[1, 1])
    ouverte = 1 / float(globale[1, 1])
    for charge in (1.0, 1e3, 1e6):
        chargee = float(impedance_d_entree_chargee(inverse, charge))
        assert chargee == pytest.approx(
            _entree_du_non_inverseur_charge(charge), rel=1e-6
        )
        exacte = a_vide / (1 + (ouverte / charge) / (1 + fermee / charge))
        assert chargee == pytest.approx(exacte, rel=0.02)

    def de_la_planche(charge: float) -> float:
        terme = (ouverte / charge) / (1 + fermee / charge) / 100.0
        return a_vide / (1 + terme)

    assert 1 - de_la_planche(1e3) / a_vide == pytest.approx(1e-3, rel=0.02)
    assert de_la_planche(1.0) == pytest.approx(a_vide / 1.5, rel=0.01)
    assert de_la_planche(1.0) < a_vide / 1.01
    sur_un_kilohm = float(impedance_d_entree_chargee(inverse, 1e3))
    assert 1 - sur_un_kilohm / a_vide == pytest.approx(0.09, abs=0.005)
    en_court_circuit = float(impedance_d_entree_chargee(inverse, 1e-6))
    assert en_court_circuit == pytest.approx(float(globale[0, 0]), rel=1e-3)
    assert en_court_circuit < a_vide / 90


def test_la_source_et_la_charge_forment_deux_diviseurs() -> None:
    """Planche 54 : `V_2 = A_cl R_L / (R_L + Z_out) Z_in / (Z_in + R_S) V_S`."""
    utile = gain_vu_de_la_charge(gain_vu_de_la_source(100.0, 50.0, 1e8), 1.0, 1e3)
    assert utile == pytest.approx(100.0 * (1e3 / 1001.0) * (1e8 / (1e8 + 50.0)))
    assert utile == pytest.approx(100.0, rel=2e-3)


def test_la_retroaction_rend_le_gain_insensible_a_l_amplificateur() -> None:
    """Planches 61 a 63 : doubler `A_v` deplace le gain ferme d'un demi pour cent."""
    simple, double = gain_en_boucle_fermee(1e4, 0.01), gain_en_boucle_fermee(2e4, 0.01)
    assert (double - simple) / simple == pytest.approx(0.005, rel=0.02)
    non_lineaire = [gain_en_boucle_fermee(1e4 + 10 * ecart, 0.01) for ecart in (0, 50)]
    assert non_lineaire[1] / non_lineaire[0] - 1 < 1e-3
