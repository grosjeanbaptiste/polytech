"""Recalcule les resultats numeriques des transcriptions manuscrites.

Une transcription est une *lecture* d'un manuscrit : le risque n'est pas
qu'elle plante, mais qu'un chiffre ait ete mal lu. Recalculer chaque
resultat depuis les donnees de l'enonce transcrit est la seule verification
possible -- si la lecture etait fausse, le calcul ne retomberait pas dessus.

Un ecart signale ici veut dire l'une de deux choses : la transcription est
fautive, ou le manuscrit lui-meme arrondit. Le cas connu est documente.
"""

import cmath
import math

import pytest

MU0 = 4 * math.pi * 1e-7


def test_force_sur_une_armature_de_condensateur() -> None:
    """Variable_Forces_6 : F = U^2 |dC| / (2 dx)."""
    assert pytest.approx(0.2) == 0.5 * (10e3) ** 2 * 4e-12 / 1e-3


def test_circuit_magnetique_inductance() -> None:
    """Stationnaire_CircuitsMagnetiques_6 : I puis L = N phi / I."""
    courant = 70 * (2 * math.pi * 16e-2) / 1000
    assert courant == pytest.approx(0.07037, rel=1e-3)
    flux = 1.6 * (2e-2 * 4e-2) + 0.5 * (1e-2 * 4e-2)
    assert 1000 * flux / courant == pytest.approx(21, rel=0.03)


def test_nappe_de_conducteurs_somme_geometrique() -> None:
    """Stationnaire_ChampMagnetique_8 : 10 paires en progression de raison 2."""
    champ = (6 / math.pi) * sum(1 / (2**i * 1e-2) for i in range(1, 11))
    assert champ == pytest.approx(190.8, rel=1e-3)


def test_arrachement_de_deux_demi_tores() -> None:
    """Variable_Forces_15 : I puis F a flux constant."""
    champ_entrefer = 1.6 / MU0
    courant = (70 * (2 * 50e-2) + 2 * champ_entrefer * 0.1e-3) / 500
    assert courant == pytest.approx(0.65, rel=0.03)
    assert champ_entrefer * 0.00064 == pytest.approx(815, rel=0.02)


def test_examen_mons_thevenin_et_force_de_levage() -> None:
    """Ex_cote_2024_2025_Mons : U = +5 V, I = 4,8 A, |F| > poids."""
    assert pytest.approx(4 / 3) == 2 * 4 / (2 + 4)
    courant_maille = (-10 - 4 * 2) / (2 + 4)
    tension_thevenin = -10 - 2 * courant_maille
    assert tension_thevenin == pytest.approx(-4.0)
    resistance = 4 / 3
    assert -(5 / 3) / (resistance + 5 / 3) * (tension_thevenin - 5) == pytest.approx(
        5.0
    )
    courant = (
        55 * (1 + 14 + 1) * 1e-2 + 4 * math.pi * 7e-2 + 2 * (0.3 / MU0) * 2e-3
    ) / 200
    assert courant == pytest.approx(4.8, rel=0.03)
    force = 0.25 * courant**2 * 200**2 * MU0 * 4e-4 / (2e-3) ** 2
    assert force == pytest.approx(29.0, rel=0.03)
    assert force > 0.5 * 9.81, "la plaque doit etre soulevee"


def test_examen_charleroi_thevenin_et_capteur() -> None:
    """Ex_cote_2024_2025_Charleroi : U = -6 V et S = 63,7 cm2."""
    assert pytest.approx(1.5) == (1 + 2) * 3 / (1 + 2 + 3)
    courant_maille = (-10 - 15 + 2 * 2) / (1 + 2 + 3)
    tension_thevenin = 3 * courant_maille
    assert tension_thevenin == pytest.approx(-10.5)
    courant = (tension_thevenin + 1.5 * 2) / (1.5 + 6)
    assert 6 * courant == pytest.approx(-6.0)
    assert (2e-2 / math.pi) * 1e4 == pytest.approx(63.7, rel=1e-3)


def test_examen_2016_energie_entrefer_et_inductance() -> None:
    """EG_Cote_2016 : 3,67e-7 J, l_e = 0,83 mm, L = 1,131 H."""
    assert pytest.approx(1 / 3) == 1 / (1 + 1 + 1)
    courant = 1 / (1 / 3 + 2)
    assert courant == pytest.approx(3 / 7)
    assert 0.5 * 1e-6 * (2 * courant) ** 2 == pytest.approx(3.67e-7, rel=1e-2)
    champ_entrefer = 1.51 / MU0
    spires = 0.9488 / (1.51 * math.pi * (1e-2) ** 2)
    assert spires == pytest.approx(2000, rel=1e-2)
    entrefer = (2000 - 50 * (2 * math.pi * 5e-2)) / (2 * champ_entrefer)
    assert entrefer * 1e3 == pytest.approx(0.8257, rel=0.02)
    assert 2000 / (2 * math.pi * 5e-2) == pytest.approx(6366.198, rel=1e-3)
    assert 2000 * 1.8 * math.pi * (1e-2) ** 2 == pytest.approx(1.131, rel=1e-2)


def test_couple_entre_deux_bobines() -> None:
    """Seance_Stationnaire_2.1 et Variable_Forces_Cours : -26 micro N.m."""
    couple = 1 * 5e-3 * (-90e-6) / (math.pi / 180)
    assert couple == pytest.approx(-2.6e-5, rel=0.02)


def test_deux_sources_en_parallele_puissance() -> None:
    """Seance_Variable_3.2 exercice 1 : I = 24,09 A et P = 5223 W."""
    courant = complex(23.71, -4.26)
    assert abs(courant) == pytest.approx(24.09, rel=1e-3)
    assert math.degrees(cmath.phase(courant)) == pytest.approx(-10.18, rel=1e-3)
    tension = 9 * abs(courant)
    assert tension == pytest.approx(216.81, rel=1e-3)
    assert tension * abs(courant) == pytest.approx(5223, rel=1e-3)


def test_source_de_courant_et_impedance_parallele() -> None:
    """Seance_Variable_3.2 exercice 5 : le manuscrit arrondit le module.

    Il ecrit P = 162,07 W, obtenu avec |I3| = 0,36 ; le module exact vaut
    0,3606, d'ou P = 162,32 W. Ecart de 0,15 %, imputable a la source.
    """
    pulsation = 2 * math.pi * 50
    courant = complex(0.2, -0.3)
    tension = 220 + 1000 * courant / (1 + 1j * pulsation * 1000 * 1e-6)
    assert abs(tension) == pytest.approx(457.25, rel=1e-3)
    assert abs(courant) == pytest.approx(0.3606, rel=1e-3)
    puissance = tension * courant.conjugate()
    assert puissance.real == pytest.approx(162.32, rel=1e-3)
    assert puissance.imag == pytest.approx(28.83, rel=1e-2)
    arrondi_du_manuscrit = abs(tension) * 0.36 * math.cos(math.radians(10.07))
    assert arrondi_du_manuscrit == pytest.approx(162.07, rel=1e-3)


def test_le_manuscrit_arrondit_le_temps_d_etablissement() -> None:
    """Seance_Variable_3.1 exercice 1 : le manuscrit ecrit 11 ms, non 11,8.

    Seul ecart trouve entre un resultat manuscrit et son recalcul. Le scan
    a ete relu a 260 dpi : il porte bien « t = 11 ms ». La transcription est
    donc fidele, et c'est la source qui arrondit.
    """
    amplitude = 12 / 53.1
    assert amplitude == pytest.approx(0.226, rel=1e-2)
    assert -2.5 * amplitude == pytest.approx(-0.565, rel=1e-2)
    pole = 53.1 / 2.5
    assert pole == pytest.approx(21.24, rel=1e-3)
    exact = -math.log(1 - 0.05 / amplitude) / pole
    assert exact * 1e3 == pytest.approx(11.77, rel=1e-2)
    assert round(exact * 1e3) != 11, "l'ecart au 11 ms manuscrit est reel"
