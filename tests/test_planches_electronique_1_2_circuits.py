"""Les planches d'electronique : des composants aux lois de Kirchhoff (1, II).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_1_part2.pdf`, 49
planches lues sur les pages rendues et sur leur couche de texte.

Les oracles : la ligne de 5 cm a 1 MHz et a 1 GHz, la resistance 0603
(`C ~ 9,4 fF`, contre 40 fF au catalogue), les 50 nH/m d'inductance
interne, l'exemple a 10 MHz du modele a constantes localisees, la
netliste de la planche 28 et les trois equations de l'exemple 2.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.resistance import resistance_uniforme
from polytech.electromagnetisme.domain.constantes import VITESSE_DE_LA_LUMIERE
from polytech.electromagnetisme.domain.jefimenko import (
    VITESSE_TYPIQUE_DANS_UN_CIRCUIT,
)
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electronique.domain.des_champs_aux_circuits import (
    capacite_parasite,
    frequence_limite,
    kirchhoff_des_courants,
    kirchhoff_des_tensions,
    les_constantes_localisees_valent,
    longueur_d_onde,
    longueur_electrique,
)
from polytech.electronique.domain.elements_localises import (
    INDUCTANCE_INTERNE_LINEIQUE,
    courant_de_la_resistance_reelle,
    courant_du_condensateur,
    sortie_de_la_ligne,
    tension_de_l_inductance,
    tension_de_la_resistance,
)
from polytech.electronique.domain.netliste import resoudre
from polytech.electronique.domain.netliste_textuelle import (
    Composant,
    elements_nodaux,
    graphe_du_circuit,
    lire_la_netliste,
)

OMEGA = sp.Symbol("omega", positive=True)
R, C, L = sp.symbols("R C L", positive=True)


def test_les_trois_composants_ideaux() -> None:
    """Planches 5 et 6 : `V = R I`, `I = C dV/dt`, `V = L dI/dt`."""
    sinus = sp.sin(OMEGA * TEMPS)
    assert tension_de_la_resistance(R, sinus) == R * sinus
    assert courant_du_condensateur(C, sinus) == C * OMEGA * sp.cos(OMEGA * TEMPS)
    assert tension_de_l_inductance(L, sinus) == L * OMEGA * sp.cos(OMEGA * TEMPS)
    assert courant_du_condensateur(C, sp.Integer(5)) == 0


LIGNE = 0.05


def test_la_ligne_retarde_le_signal() -> None:
    """Planche 9 : `V_out(t) = V_in(t - l / (c/3))`, soit un delai de `3 l / c`."""
    sortie = sortie_de_la_ligne(
        sp.sin(OMEGA * TEMPS), LIGNE, VITESSE_TYPIQUE_DANS_UN_CIRCUIT
    )
    delai = 3 * LIGNE / VITESSE_DE_LA_LUMIERE
    assert delai == pytest.approx(0.5e-9, rel=1e-3, abs=0)
    attendu = sp.sin(OMEGA * (TEMPS - delai))
    assert float((sortie - attendu).subs({OMEGA: 1e7, TEMPS: 3e-8})) == pytest.approx(
        0.0
    )
    with pytest.raises(ValueError, match="vitesse de propagation nulle ou negative"):
        sortie_de_la_ligne(sp.sin(TEMPS), LIGNE, 0.0)


def test_a_un_megahertz_la_ligne_est_transparente() -> None:
    """Planche 10 : 5 cm a 1 MHz, les deux courbes se confondent (0,18 degre)."""
    angle = longueur_electrique(LIGNE, VITESSE_TYPIQUE_DANS_UN_CIRCUIT, 1e6)
    assert math.degrees(angle) == pytest.approx(0.18, abs=0.005)
    assert les_constantes_localisees_valent(LIGNE, VITESSE_TYPIQUE_DANS_UN_CIRCUIT, 1e6)


def test_a_un_gigahertz_la_ligne_inverse_le_signal() -> None:
    """Planche 11 : 5 cm a 1 GHz valent une demi-longueur d'onde, soit 180 degres."""
    onde = longueur_d_onde(VITESSE_TYPIQUE_DANS_UN_CIRCUIT, 1e9)
    assert onde == pytest.approx(0.1, rel=1e-3)
    angle = longueur_electrique(LIGNE, VITESSE_TYPIQUE_DANS_UN_CIRCUIT, 1e9)
    assert angle == pytest.approx(math.pi, rel=1e-3)
    entree = sp.sin(2 * sp.pi * 1e9 * TEMPS)
    sortie = sortie_de_la_ligne(entree, LIGNE, VITESSE_TYPIQUE_DANS_UN_CIRCUIT)
    instant = {TEMPS: 0.25e-9}
    assert float(sortie.subs(instant)) == pytest.approx(
        -float(entree.subs(instant)), abs=2e-3
    )
    assert not les_constantes_localisees_valent(
        LIGNE, VITESSE_TYPIQUE_DANS_UN_CIRCUIT, 1e9
    )


def test_la_resistance_du_barreau() -> None:
    """Planches 15 a 17 : `E_R = sigma_s / eps`, `V = E_R l`, `R = l / (sigma_c A)`."""
    conductivite, longueur, aire = 20.0, 1.6e-3, 3.4e-7
    resistance = resistance_uniforme(1 / conductivite, longueur, aire)
    assert resistance == pytest.approx(longueur / (conductivite * aire))
    assert tension_de_la_resistance(resistance, 1e-3) == pytest.approx(
        resistance * 1e-3
    )


def test_la_capacite_cachee_d_une_resistance_0603() -> None:
    """Planche 23 : `A = 0,34 mm^2`, `l = 1,6 mm`, `eps_r = 5` : 9,4 fF.

    Le catalogue du fabricant annonce environ 40 fF : la planche le dit,
    l'estimation est « un peu basse » mais du bon ordre.
    """
    aire = 0.85e-3 * 0.4e-3
    assert aire == pytest.approx(3.4e-7)
    permittivite = 5 * 8.85e-12
    assert permittivite == pytest.approx(4.425e-11, rel=1e-9, abs=0)
    capacite = capacite_parasite(permittivite, aire, 1.6e-3)
    assert capacite == pytest.approx(9.4e-15, rel=5e-3, abs=0)
    assert 4 < 40e-15 / capacite < 5


def test_la_resistance_reelle_est_une_resistance_et_une_capacite() -> None:
    """Planches 21 et 22 : `I = V / R + C dV/dt`, de la continuite sur l'armature."""
    tension = sp.sin(OMEGA * TEMPS)
    courant = courant_de_la_resistance_reelle(R, C, tension)
    assert sp.simplify(courant - tension / R - C * sp.diff(tension, TEMPS)) == 0
    assert courant_de_la_resistance_reelle(R, C, sp.Integer(3)) == 3 / R
    rapport = sp.simplify((C * OMEGA) / (1 / R))
    assert rapport == OMEGA * R * C


def test_l_inductance_interne_d_un_fil() -> None:
    """Planche 24 : 50 nH/m pour un fil rond, a basse frequence (`mu_0 / 8 pi`)."""
    assert pytest.approx(50e-9, rel=1e-6) == INDUCTANCE_INTERNE_LINEIQUE


def test_le_modele_a_constantes_localisees() -> None:
    """Planche 26 : a 10 MHz, « wavelength is 30 meters », donc `l_c < 1,5 m`.

    Les 30 metres sont la longueur d'onde dans le vide. Avec la vitesse
    que le meme fascicule recommande, `c / 3`, elle tombe a 10 metres et
    la longueur admissible a 0,5 metre.
    """
    dans_le_vide = longueur_d_onde(VITESSE_DE_LA_LUMIERE, 10e6)
    assert dans_le_vide == pytest.approx(30.0, rel=1e-3)
    assert dans_le_vide / 20 == pytest.approx(1.5, rel=1e-3)
    assert frequence_limite(1.5, VITESSE_DE_LA_LUMIERE) == pytest.approx(10e6, rel=1e-3)
    en_ligne = longueur_d_onde(VITESSE_TYPIQUE_DANS_UN_CIRCUIT, 10e6)
    assert en_ligne / 20 == pytest.approx(0.5, rel=1e-3)
    assert not les_constantes_localisees_valent(
        1.0, VITESSE_TYPIQUE_DANS_UN_CIRCUIT, 10e6
    )


NETLISTE = """
v1 0 1 12
r1 1 2 1e3
r2 2 0 10e4
r3 2 0 10e4
c1 2 0 10e-6
"""


def test_la_netliste_de_la_planche_28() -> None:
    """Une lettre et un numero, le noeud de depart, celui d'arrivee, la valeur."""
    composants = lire_la_netliste(NETLISTE)
    assert composants[0] == Composant("v", 1, 0, 1, 12.0)
    assert composants[1] == Composant("r", 1, 1, 2, 1e3)
    assert [c.nature for c in composants] == ["v", "r", "r", "r", "c"]
    assert composants[4].valeur == pytest.approx(10e-6)
    with pytest.raises(ValueError, match="ligne de netliste illisible"):
        lire_la_netliste("r1 1 2")
    with pytest.raises(ValueError, match="composant inconnu"):
        lire_la_netliste("x1 1 2 5")
    with pytest.raises(ValueError, match="identifiant deja employe"):
        lire_la_netliste("r1 1 2 5\nr1 2 0 5")


def test_un_circuit_est_un_graphe_oriente() -> None:
    """Planche 27 : `G = (V, E)`, des couples ordonnes de noeuds distincts."""
    noeuds, aretes = graphe_du_circuit(lire_la_netliste(NETLISTE))
    assert noeuds == frozenset({0, 1, 2})
    assert aretes == ((0, 1), (1, 2), (2, 0), (2, 0), (2, 0))
    with pytest.raises(ValueError, match="relie un noeud a lui-meme"):
        graphe_du_circuit((Composant("r", 1, 2, 2, 5.0),))


def test_la_netliste_ecrit_cent_kiloohms_pour_dix() -> None:
    """Planche 28 : « R_2 = R_3 = 10 kOhm », mais la netliste porte `10e4`.

    `10e4` vaut cent mille. En continu, le noeud 2 est alors a 11,76 V ;
    avec les 10 kOhm de l'enonce, `10e3`, il est a 10 V.
    """
    composants = lire_la_netliste(NETLISTE)
    assert composants[2].valeur == pytest.approx(1e5)
    continu = resoudre(elements_nodaux(composants, laplace=0), 2)
    assert float(continu[0]) == pytest.approx(12.0)
    assert float(continu[1]) == pytest.approx(12.0 * 50 / 51)
    corrigee = lire_la_netliste(NETLISTE.replace("10e4", "10e3"))
    enonce = resoudre(elements_nodaux(corrigee, laplace=0), 2)
    assert float(enonce[1]) == pytest.approx(10.0)


def test_la_netliste_se_resout_aussi_en_laplace() -> None:
    """Le condensateur est l'admittance `s C` : un passe-bas du premier ordre."""
    s = sp.Symbol("s")
    tensions = resoudre(elements_nodaux(lire_la_netliste(NETLISTE), laplace=s), 2)
    attendue = 12 / (1 + 1e3 * (2 / 1e5 + 10e-6 * s))
    for valeur in (0, 50, 100j, 3 + 400j):
        obtenue = complex(tensions[1].subs(s, valeur))
        assert obtenue == pytest.approx(complex(attendue.subs(s, valeur)))
    coupure = 1.02 / (1e3 * 10e-6)
    assert abs(complex(tensions[1].subs(s, 1j * coupure))) == pytest.approx(
        12 / 1.02 / math.sqrt(2)
    )


def test_les_lois_de_kirchhoff() -> None:
    """Planches 38 a 41 : Faraday a flux constant, continuite sans accumulation."""
    assert kirchhoff_des_tensions((12.0, -7.5, -4.5)) == pytest.approx(0.0)
    assert kirchhoff_des_courants((2.0, -1.5, -0.5)) == pytest.approx(0.0)


B0, AIRE = sp.symbols("B_0 S", positive=True)


def test_un_flux_variable_devient_une_source_de_tension() -> None:
    """Planches 43 et 44 : `V_S = -B_0 omega sin(omega t) S`, et `V_S = V_1 + V_2`."""
    flux = B0 * sp.cos(OMEGA * TEMPS) * AIRE
    source = force_electromotrice(flux)
    derivee = sp.diff(flux, TEMPS)
    assert sp.simplify(derivee + B0 * OMEGA * sp.sin(OMEGA * TEMPS) * AIRE) == 0
    assert sp.simplify(source + derivee) == 0
    imprimee = -B0 * OMEGA * sp.sin(OMEGA * TEMPS) * AIRE
    assert sp.simplify(imprimee - derivee) == 0


def test_l_exemple_2_trois_equations() -> None:
    """Planches 45 a 47 : `-V_1 - V_2 + V_S = 0`, `V_2 = V_3`, `-I_1 + I_2 + I_3 = 0`.

    Resolu : `R_1` en serie avec `R_2` et `R_3` en parallele, sous `V_S`.
    """
    source, un, deux, trois = sp.symbols("V_S R_1 R_2 R_3", positive=True)
    netliste = (
        Composant("v", 1, 0, 1, source),
        Composant("r", 1, 1, 2, un),
        Composant("r", 2, 2, 0, deux),
        Composant("r", 3, 2, 0, trois),
    )
    tensions = resoudre(elements_nodaux(netliste, laplace=0), 2)
    v_2 = tensions[1]
    v_1 = tensions[0] - v_2
    assert sp.simplify(-v_1 - v_2 + source) == 0
    courants = (-v_1 / un, v_2 / deux, v_2 / trois)
    assert sp.simplify(sum(courants)) == 0
    parallele = deux * trois / (deux + trois)
    assert sp.simplify(v_2 - source * parallele / (un + parallele)) == 0


def test_une_inductance_en_netliste() -> None:
    """`V = L dI/dt` devient l'admittance `1 / (s L)` ; en continu, un fil."""
    s = sp.Symbol("s")
    composants = lire_la_netliste("v1 0 1 10\nr1 1 2 100\nl1 2 0 1e-3")
    tensions = resoudre(elements_nodaux(composants, laplace=s), 2)
    for valeur in (1e4, 2e5j):
        attendue = 10 * valeur * 1e-3 / (100 + valeur * 1e-3)
        assert complex(tensions[1].subs(s, valeur)) == pytest.approx(attendue)
    with pytest.raises(ValueError, match="court-circuit en continu"):
        elements_nodaux(composants, laplace=0)
