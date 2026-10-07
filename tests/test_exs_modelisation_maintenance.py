"""Optimisation : la modelisation, problemes VIIa et VIIb (la maintenance).

Source : `Optimisation/EXS/Exs_Modelisation.pdf`, planches 28 a 43. Cinq
ateliers a entretenir trois semaines chacun, dans une fenetre de dates.
Il n'y a que 720 plannings : ils sont tous passes en revue, ce qui donne
l'optimum et eprouve une a une les contraintes des planches 40 a 43.
"""

from __future__ import annotations

import pytest

from polytech.optimisation.application import modele_maintenance as ma
from polytech.optimisation.application import modele_maintenance_conditions as co

# --- Probleme VIIa (planches 28 a 37) ----------------------------------------


def test_viia_il_y_a_720_plannings() -> None:
    """Quatre debuts possibles pour A1, trois pour A2, quatre, cinq et trois."""
    assert len(ma.plannings()) == 4 * 3 * 4 * 5 * 3


def test_viia_les_besoins_d_un_planning() -> None:
    """Tout au plus tot : A1 et A2 en semaine 1, A3 et A4 en 2, A5 en 3."""
    assert ma.besoins((1, 1, 2, 2, 3)) == (5, 9, 7, 7, 2, 0, 0, 0)


def test_viia_la_ligne_de_la_semaine_2() -> None:
    """Planche 31 : `2 d11 + 2 d12 + 2 d21 + 3 d22 + 1 d42`. Il y manque `4 d32`."""
    imprimee = {(1, 1): 2, (1, 2): 2, (2, 1): 2, (2, 2): 3, (4, 2): 1}
    assert ma.ligne_imprimee(2) == imprimee
    assert ma.ligne_des_besoins(2) == imprimee | {(3, 2): 4}


def test_viia_les_planches_oublient_a3_quand_il_commence_en_semaine_2() -> None:
    """Sa fenetre est [2, 5] ; les planches 30 a 37 ne le montrent qu'a partir de 3."""
    assert ma.FENETRES[3] == (2, 5)
    oublies = [ma.ligne_des_besoins(s).get((3, 2), 0) for s in range(1, 9)]
    assert oublies == [0, 4, 2, 2, 0, 0, 0, 0]
    assert all((3, 2) not in ma.ligne_imprimee(s) for s in range(1, 9))


def test_viia_ce_que_le_modele_imprime_croit() -> None:
    """Meme planning, mais 26400 : il compte 2 ouvriers la ou il en faut 6."""
    cout, pointe, debuts = ma.classement(imprimes=True)[0]
    assert (cout, pointe, debuts) == (26400, 3, (3, 1, 2, 6, 5))
    assert ma.besoins(debuts, imprimes=True) == (3, 2, 3, 2, 3, 3, 3, 3)
    assert ma.besoins(debuts) == (3, 6, 5, 4, 3, 3, 3, 3)


def test_viia_la_planche_29_ecrit_3_devant_d12() -> None:
    """A1 demande 2 personnes sa premiere semaine : la planche 31 a raison."""
    assert ma.COEFFICIENT_IMPRIME_PLANCHE_29[(1, 2)] == 3
    assert ma.ligne_des_besoins(2)[(1, 2)] == ma.CHARGES[1][0] == 2


def test_viia_les_lignes_des_semaines_7_et_8() -> None:
    """Planches 36 et 37."""
    assert ma.ligne_des_besoins(7) == {(3, 5): 2, (4, 5): 3, (4, 6): 1, (5, 5): 2}
    assert ma.ligne_des_besoins(8) == {(4, 6): 3}


def test_viia_le_cout_du_personnel_pour_des_besoins_donnes() -> None:
    """Sept ouvriers au depart, neuf a la fin, payes la neuvieme semaine."""
    assert ma.cout_du_personnel((7,) * 8) == 600 * 7 * 8 + 600 * 9 + 300 * 2
    assert ma.cout_du_personnel((0,) * 8) == 1200 * 7 + 300 * 9 + 600 * 9


def test_viia_le_meilleur_planning() -> None:
    cout, debuts = ma.meilleur_planning()
    assert (cout, debuts) == (31800, (3, 1, 2, 6, 5))
    assert ma.besoins(debuts) == (3, 6, 5, 4, 3, 3, 3, 3)


def test_viia_le_meme_cout_par_le_simplexe() -> None:
    """Second chemin : le modele en `T`, `E`, `L` de la planche 29, plannings fixes."""
    for debuts in ((3, 1, 2, 6, 5), (1, 1, 4, 6, 5), (4, 3, 4, 2, 4)):
        besoins = ma.besoins(debuts)
        assert ma.cout_par_le_simplexe(besoins) == ma.cout_du_personnel(besoins)


# --- Probleme VIIb (planches 38 a 43) ----------------------------------------


def test_viib_la_plus_petite_pointe() -> None:
    """Un effectif constant de 5 suffit, pour six plannings."""
    pointe, plannings = ma.plus_petite_pointe()
    assert pointe == 5
    assert len(plannings) == 6
    assert (1, 1, 4, 6, 5) in plannings


@pytest.mark.parametrize("condition", sorted(co.CONDITIONS))
def test_viib_les_contraintes_des_planches_disent_ce_que_dit_l_enonce(
    condition: str,
) -> None:
    """Sur les 720 plannings, la condition et ses contraintes s'accordent."""
    for debuts in ma.plannings():
        assert co.respecte_la_planche(condition, debuts) == co.CONDITIONS[condition](
            debuts
        )


@pytest.mark.parametrize(
    ("condition", "permis", "cout", "pointe"),
    [
        ("A3 pas avant A1", 585, 32100, 5),
        ("A4 en meme temps que A3", 144, 35700, 7),
        ("A4 et A5 pas la meme semaine", 576, 31800, 5),
        ("A5 apres la fin de A1", 180, 32100, 5),
    ],
)
def test_viib_ce_que_chaque_condition_coute(
    condition: str, permis: int, cout: int, pointe: int
) -> None:
    assert co.bilan(condition) == (permis, cout, pointe)


def test_une_condition_inconnue_est_refusee() -> None:
    with pytest.raises(KeyError, match="condition inconnue : "):
        co.bilan("A2 avant A1")
