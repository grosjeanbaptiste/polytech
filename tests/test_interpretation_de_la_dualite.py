"""Chapitre IV.3 : ce que le dual dit, economiquement, et comment il s'ecrit.

Oracles (`Optimisation/SLIDES/Slides_04_Definitions_proprietes_et_`
`interpretation_de_la_dualite.pdf`) : page 102, le dual de la
planification de production du chapitre I, recopie ligne par ligne ;
page 103, le dual du probleme de transport, `v_j - u_i <= c_ij` ;
pages 100-102, le prix fictif (« shadow price ») et la fonction
lagrangienne dont l'optimum est l'equilibre des deux « concurrents ».
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.dual import dual_de
from polytech.optimisation.domain.ecarts_complementaires import duale_par_les_ecarts
from polytech.optimisation.domain.explications_de_la_dualite import (
    SOURCE,
    expliquer_le_dual,
)
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.interpretation_duale import (
    lagrangien,
    prix_fictifs,
    valeur_d_une_unite_de_plus,
)
from polytech.optimisation.domain.modeles import (
    planification_de_production,
    probleme_de_melange,
)
from polytech.optimisation.domain.resolution import resoudre
from polytech.optimisation.domain.transport import ProblemeDeTransport

R = sp.Rational
INF, SUP, EGA = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL
MAX, MIN = Optimisation.MAX, Optimisation.MIN
TEMPS = ((4, 0, 2, 0, 4), (2, 3, 0, 0, 3), (3, 0, 1, 4, 0), (0, 5, 4, 2, 0))


def _primal(capacites: tuple[int, int, int] = (8, 6, 2)) -> ProgrammeGeneral:
    """Page 99 : `max 6 x1 + 5 x2` ; les `d_i` sont les quantites disponibles."""
    lignes = ((1, 1), (-2, 3), (1, -1))
    contraintes = tuple(
        Contrainte(a, INF, d) for a, d in zip(lignes, capacites, strict=True)
    )
    return ProgrammeGeneral(MAX, (6, 5), contraintes)


def test_le_dual_de_la_planification_est_celui_de_la_page_102() -> None:
    primal = planification_de_production(
        gains=(100, 75, 50, 25, 50), capacites=(500, 800, 700, 900), temps=TEMPS
    )
    dual = dual_de(primal)
    assert dual.optimisation is MIN
    assert dual.objectif == (500, 800, 700, 900)
    assert dual.contraintes == (
        Contrainte((4, 2, 3, 0), SUP, 100),
        Contrainte((0, 3, 0, 5), SUP, 75),
        Contrainte((2, 0, 1, 4), SUP, 50),
        Contrainte((0, 0, 4, 2), SUP, 25),
        Contrainte((4, 3, 0, 0), SUP, 50),
    )


def test_le_dual_du_probleme_dietetique_impute_une_valeur_aux_nutriments() -> None:
    """Page 102 : `max sum d_i u_i`, `sum t_ij u_i <= c_j`, `u >= 0`."""
    primal = probleme_de_melange(prix=(3, 2), apports=((1, 2), (4, 1)), besoins=(8, 6))
    dual = dual_de(primal)
    assert dual.optimisation is MAX
    assert dual.objectif == (8, 6)
    assert dual.contraintes == (
        Contrainte((1, 4), INF, 3),
        Contrainte((2, 1), INF, 2),
    )


def test_le_dual_du_transport_compare_l_ecart_de_valeur_au_cout() -> None:
    """Page 103 : `max sum d_j v_j - sum q_i u_i`, `v_j - u_i <= c_ij`."""
    transport = ProblemeDeTransport(
        stocks=(7, 6), demandes=(5, 4), couts=((1, 2), (3, 4))
    )
    dual = dual_de(transport.programme())
    assert dual.optimisation is MAX
    assert dual.objectif == (-7, -6, 5, 4)
    assert dual.contraintes == (
        Contrainte((-1, 0, 1, 0), INF, 1),
        Contrainte((-1, 0, 0, 1), INF, 2),
        Contrainte((0, -1, 1, 0), INF, 3),
        Contrainte((0, -1, 0, 1), INF, 4),
    )


def test_a_l_optimum_l_excedent_de_valeur_egale_le_cout_de_transport() -> None:
    """Page 103 : « cet excedent est egal au cout minimum de transport ».

    A la main : le depot 1 coute 2 de moins par unite vers chaque centre ;
    il livre ses 7 unites, le depot 2 les 2 restantes : `5 + 2*2 + 2*4 = 17`.
    """
    transport = ProblemeDeTransport(
        stocks=(7, 6), demandes=(5, 4), couts=((1, 2), (3, 4))
    ).programme()
    assert resoudre(dual_de(transport)).valeur == resoudre(transport).valeur == 17


def test_les_prix_fictifs_sont_la_solution_optimale_du_dual() -> None:
    """Page 99 : `u = (11/2, 0, 1/2)`."""
    assert prix_fictifs(_primal()) == (R(11, 2), 0, R(1, 2))
    assert prix_fictifs(_primal()) == duale_par_les_ecarts(_primal(), x=(5, 3))


def test_un_bien_en_surplus_a_un_prix_fictif_nul() -> None:
    """Page 100 : `sum t_ij x_j < d_i` entraine `u_i = 0` ; ici la contrainte 2."""
    assert _primal().contraintes[1].ecart((5, 3)) > 0
    assert prix_fictifs(_primal())[1] == 0
    assert valeur_d_une_unite_de_plus(_primal(), contrainte=2) == 0


def test_le_prix_fictif_est_la_valeur_marginale_du_bien() -> None:
    """Page 100 : une unite de `d_i` en plus deplace `z` de `u_i`."""
    assert valeur_d_une_unite_de_plus(_primal(), contrainte=1) == R(11, 2)
    assert valeur_d_une_unite_de_plus(_primal(), contrainte=3) == R(1, 2)
    assert resoudre(_primal((9, 6, 2))).valeur == 45 + R(11, 2)


def test_le_lagrangien_s_ecrit_de_deux_manieres() -> None:
    """Pages 101-102 : `c x + u (d - T x) = u d + (c - u T) x`."""
    x, u = (1, 2), (3, 1, 2)
    de_l_entreprise = 6 * 1 + 5 * 2 + 3 * (8 - 3) + 1 * (6 - 4) + 2 * (2 + 1)
    du_marche = (3 * 8 + 1 * 6 + 2 * 2) + 1 * (6 - 3) + 2 * (5 - 4)
    assert lagrangien(_primal(), x, u) == de_l_entreprise == du_marche


def test_a_l_equilibre_le_lagrangien_vaut_les_deux_optimums() -> None:
    """Page 102 : `phi(x, u) = sum c_j x_j = sum u_i d_i` a l'optimum."""
    assert lagrangien(_primal(), (5, 3), (R(11, 2), 0, R(1, 2))) == 45


def test_l_equilibre_est_un_point_selle() -> None:
    """Page 101 : l'entreprise maximise en `x`, le marche minimise en `u`."""
    optimum_x, optimum_u = (5, 3), (R(11, 2), 0, R(1, 2))
    for x in ((0, 0), (2, 0), (R(18, 5), R(22, 5)), (0, 2)):
        assert lagrangien(_primal(), x, optimum_u) <= 45
    for u in ((6, 0, 0), (5, 0, 1), (0, 5, 16)):
        assert lagrangien(_primal(), optimum_x, u) >= 45


def test_l_explication_du_dual_dit_chaque_regle_de_la_page_91() -> None:
    primal = ProgrammeGeneral(
        optimisation=MAX,
        objectif=(1, -4),
        contraintes=(
            Contrainte((1, 0), INF, 7),
            Contrainte((3, -2), SUP, 4),
            Contrainte((1, 1), EGA, 5),
        ),
        libres=frozenset({1}),
    )
    raisonnement = expliquer_le_dual(primal)
    assert [e.affirmation for e in raisonnement.etapes] == [
        "la contrainte 2 est retournee : -3 x_1 + 2 x_2 <= -4",
        "le dual est une minimisation",
        "u_1 >= 0",
        "u_2 >= 0",
        "u_3 est de signe quelconque",
        "la contrainte duale 1 est une inegalite : u_1 - 3 u_2 + u_3 >= 1",
        "la contrainte duale 2 est une egalite : 2 u_2 + u_3 = -4",
        "la fonction economique du dual est 7 u_1 - 4 u_2 + 5 u_3",
    ]
    assert raisonnement.conclusion == (
        "min 7 u_1 - 4 u_2 + 5 u_3 sous 2 contraintes, 3 variables duales"
    )


def test_l_explication_justifie_chaque_regle_et_cite_la_page() -> None:
    primal = _primal()
    raisonnement = expliquer_le_dual(primal)
    assert raisonnement.sources == (f"{SOURCE}, IV.1, p. 91-92",)
    assert raisonnement.etapes[0].affirmation == "le dual est une minimisation"
    assert "maximisation" in raisonnement.etapes[0].justification
    assert "inegalite" in raisonnement.etapes[1].justification
    assert "non-negativite" in raisonnement.etapes[4].justification
    assert "seconds membres" in raisonnement.etapes[-1].justification


def test_l_explication_d_une_variable_libre_et_d_une_egalite_les_nomme() -> None:
    primal = ProgrammeGeneral(
        MIN, (1, 1), (Contrainte((1, 1), EGA, 2),), libres=frozenset({0})
    )
    etapes = expliquer_le_dual(primal).etapes
    assert "egalite" in etapes[1].justification
    assert "signe quelconque" in etapes[2].justification
    assert etapes[2].affirmation == "la contrainte duale 1 est une egalite : u_1 = 1"
    assert etapes[3].affirmation == "la contrainte duale 2 est une inegalite : u_1 <= 1"


def test_un_probleme_sans_optimum_fini_n_a_pas_de_prix_fictifs() -> None:
    """Le deuxieme couple de la page 95 : primal non borne, dual impossible."""
    contraintes = (Contrainte((-1, 1), INF, 1), Contrainte((1, -1), INF, 1))
    primal = ProgrammeGeneral(MAX, (2, 2), contraintes)
    with pytest.raises(ValueError, match="pas de solution optimale finie"):
        prix_fictifs(primal)


def test_le_lagrangien_a_l_origine_vaut_la_fonction_economique_du_dual() -> None:
    """`phi(0, u) = u d`, le primal etant d'abord ecrit sous la convention.

    Premier couple de la page 95 : `min 2 x1 + 2 x2` avec des `<=`, que la
    convention retourne ; son dual est `max -u1 - u2`.
    """
    contraintes = (Contrainte((-1, 1), INF, 1), Contrainte((1, -1), INF, 1))
    primal = ProgrammeGeneral(MIN, (2, 2), contraintes)
    assert lagrangien(primal, (0, 0), (1, 1)) == dual_de(primal).valeur((1, 1)) == -2


def test_l_explication_dit_dans_quel_sens_la_convention_ecrit_les_inegalites() -> None:
    maximisation = ProgrammeGeneral(MAX, (1,), (Contrainte((1,), SUP, 1),))
    minimisation = ProgrammeGeneral(MIN, (1,), (Contrainte((1,), INF, 1),))
    assert "s'ecrivent <=" in expliquer_le_dual(maximisation).etapes[0].justification
    assert "s'ecrivent >=" in expliquer_le_dual(minimisation).etapes[0].justification
