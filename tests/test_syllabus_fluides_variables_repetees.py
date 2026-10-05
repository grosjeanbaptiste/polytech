"""Le syllabus du theoreme de Vaschy-Buckingham : la methode, exercee.

Source : `MecaniqueFluides/SYLLABUS/Syllabus_Theoreme_Vaschy-Buckingham.pdf`
(Cengel, section 7-4, pages 281 a 287).

L'oracle est l'exemple du document, la bille qui tombe dans le vide :
`z = f(t, w_0, z_0, g)`, parametres repetes `w_0` et `z_0`. Ses exposants
y sont ecrits : `a_2 = 1`, `b_2 = -1`, `a_3 = -2`, `b_3 = 1`, et le
troisieme groupe est eleve a la puissance `-1/2` pour rendre le nombre de
Froude.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.fluides.domain.analyse_dimensionnelle import nombre_de_groupes
from polytech.fluides.domain.similitude import nombre_de_froude
from polytech.fluides.domain.variables_repetees import (
    dimensions_du_produit,
    elever,
    groupe_pi,
    groupes_pi,
    multiplier,
)

BILLE = {
    "z": {"L": 1},
    "t": {"t": 1},
    "w0": {"L": 1, "t": -1},
    "z0": {"L": 1},
    "g": {"L": 1, "t": -2},
}
"""Etape 2 du document : les dimensions primaires des cinq parametres."""

REPETES = ("w0", "z0")
DEMI = sp.Rational(1, 2)


def test_les_trois_groupes_de_la_bille_qui_tombe() -> None:
    """Etape 5 : `z / z_0`, `w_0 t / z_0` et `g z_0 / w_0^2`."""
    assert groupe_pi("z", REPETES, BILLE) == {"z": 1, "z0": -1}
    assert groupe_pi("t", REPETES, BILLE) == {"t": 1, "w0": 1, "z0": -1}
    assert groupe_pi("g", REPETES, BILLE) == {"g": 1, "w0": -2, "z0": 1}


def test_le_nombre_de_groupes_est_celui_du_theoreme() -> None:
    """Etape 3 : `k = n - j = 5 - 2 = 3`, la variable dependante en tete."""
    groupes = groupes_pi("z", REPETES, BILLE)
    assert len(groupes) == nombre_de_groupes(len(BILLE), len(REPETES))
    assert next(iter(groupes[0])) == "z"
    for groupe in groupes:
        assert dimensions_du_produit(groupe, BILLE) == {}


def test_le_troisieme_groupe_devient_le_nombre_de_froude() -> None:
    """« We imposed an exponent of -1/2 on Pi_3 » : `Fr = w_0 / racine(g z_0)`."""
    froude = elever(groupe_pi("g", REPETES, BILLE), -DEMI)
    assert froude == {"g": -DEMI, "w0": 1, "z0": -DEMI}
    assert dimensions_du_produit(froude, BILLE) == {}
    valeurs = {"g": 9.81, "w0": 3.0, "z0": 2.0}
    produit = 1.0
    for nom, exposant in froude.items():
        produit *= valeurs[nom] ** float(exposant)
    assert produit == pytest.approx(nombre_de_froude(3.0, 2.0))


def test_un_produit_de_groupes_reste_sans_dimension() -> None:
    """Table 7-4, regle 3 : remplacer `Pi_3` par `Pi_3 Pi_1`."""
    produit = multiplier(groupe_pi("g", REPETES, BILLE), groupe_pi("z", REPETES, BILLE))
    assert produit == {"g": 1, "w0": -2, "z": 1}
    assert dimensions_du_produit(produit, BILLE) == {}


def test_la_solution_exacte_ne_depend_que_des_groupes() -> None:
    """Etape 6 : `z* = f(t*, Fr)`. La chute libre donne `1 + t* - t*^2 / (2 Fr^2)`.

    Deux chutes differentes mais de memes `t*` et `Fr` ont le meme `z*`.
    """

    def reduits(z0: float, w0: float, g: float, t: float) -> tuple[float, ...]:
        z = z0 + w0 * t - g * t**2 / 2
        return (z / z0, w0 * t / z0, nombre_de_froude(w0, z0, g))

    terre = reduits(2.0, 3.0, 9.81, 0.4)
    ailleurs = reduits(8.0, 6.0, 9.81 * 4 * 6**2 / (3**2 * 8) / 2, 0.8)
    assert ailleurs[1:] == pytest.approx(terre[1:])
    assert ailleurs[0] == pytest.approx(terre[0])
    assert terre[0] == pytest.approx(1 + terre[1] - terre[1] ** 2 / (2 * terre[2] ** 2))


def test_la_trainee_rend_le_nombre_de_reynolds() -> None:
    """Longueur, vitesse et masse volumique repetees : `C_D = f(Re)`."""
    trainee = {
        "F": {"m": 1, "L": 1, "t": -2},
        "V": {"L": 1, "t": -1},
        "L": {"L": 1},
        "rho": {"m": 1, "L": -3},
        "mu": {"m": 1, "L": -1, "t": -1},
    }
    force, visqueux = groupes_pi("F", ("V", "L", "rho"), trainee)
    assert force == {"F": 1, "V": -2, "L": -2, "rho": -1}
    assert elever(visqueux, -1) == {"mu": -1, "V": 1, "L": 1, "rho": 1}


def test_deux_parametres_de_memes_dimensions_ne_se_repetent_pas() -> None:
    """Table 7-3, regle 5 : `z` et `z_0` sont deux longueurs."""
    with pytest.raises(ValueError, match="forment un groupe a eux seuls"):
        groupe_pi("g", ("z", "z0"), BILLE)


def test_des_parametres_repetes_qui_forment_un_groupe_sont_refuses() -> None:
    """Table 7-3, regle 2 : `t w_0 / z_0` est deja sans dimension."""
    with pytest.raises(ValueError, match="forment un groupe a eux seuls"):
        groupe_pi("g", ("t", "w0", "z0"), BILLE)


def test_les_parametres_repetes_doivent_porter_toutes_les_dimensions() -> None:
    """Table 7-3, regle 3 : une longueur seule ne compense pas un temps."""
    with pytest.raises(ValueError, match="ne portent pas toutes les dimensions"):
        groupe_pi("g", ("z0",), BILLE)


def test_la_variable_dependante_ne_se_repete_pas() -> None:
    """Table 7-3, regle 1."""
    with pytest.raises(ValueError, match="variable dependante"):
        groupes_pi("z", ("z", "w0"), BILLE)
    with pytest.raises(ValueError, match="parametre inconnu"):
        groupe_pi("masse", REPETES, BILLE)
