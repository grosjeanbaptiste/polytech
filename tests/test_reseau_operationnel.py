"""Un reseau en regime transitoire, resolu dans le domaine de Laplace.

Meme principe que le reseau stationnaire -- la loi des noeuds a chaque
noeud, une inconnue de courant par element qui impose une tension --
avec les deux elements qui ont de la memoire :

- une **bobine** impose `V = L (p I - i(0))`, plus un terme par couplage ;
- une **capacite** fait passer `C (p U - u(0))`.

Les conditions initiales entrent donc dans les equations elles-memes, et
non apres coup : c'est ce qui rend la methode de Laplace sure, et c'est
aussi la qu'un corrige manuscrit se trompe le plus volontiers.

Les oracles sont ici les reponses du premier ordre, que tout cours donne.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.elements_operationnels import (
    Bobine,
    Capacite,
    Couplage,
)
from polytech.electromagnetisme.domain.circuits.operationnel import (
    ReseauOperationnel,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Resistance,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE

TENSION = sp.Symbol("E", positive=True)
RESISTANCE = sp.Symbol("R", positive=True)
INDUCTANCE = sp.Symbol("L", positive=True)
CAPACITE = sp.Symbol("C", positive=True)
ECHELON = TENSION / VARIABLE
"""Une source continue enclenchee a l'instant zero."""


def test_l_etablissement_du_courant_dans_une_bobine() -> None:
    """`I(p) = E / (p (R + p L))` : le circuit `R`-`L` enclenche sur `E`."""
    bobine = Bobine("b", "masse", INDUCTANCE)
    reseau = ReseauOperationnel(
        elements=(
            SourceDeTension("s", "masse", ECHELON),
            Resistance("s", "b", RESISTANCE),
            bobine,
        ),
        reference="masse",
    )
    attendu = TENSION / (VARIABLE * (RESISTANCE + VARIABLE * INDUCTANCE))
    assert sp.simplify(reseau.courant(bobine) - attendu) == 0


def test_la_charge_d_un_condensateur() -> None:
    """`U(p) = E / (p (1 + p R C))` : le circuit `R`-`C` enclenche sur `E`."""
    reseau = ReseauOperationnel(
        elements=(
            SourceDeTension("s", "masse", ECHELON),
            Resistance("s", "c", RESISTANCE),
            Capacite("c", "masse", CAPACITE),
        ),
        reference="masse",
    )
    attendu = TENSION / (VARIABLE * (1 + VARIABLE * RESISTANCE * CAPACITE))
    assert sp.simplify(reseau.potentiels()["c"] - attendu) == 0


def test_le_courant_initial_d_une_bobine_s_eteint_dans_la_resistance() -> None:
    """Sans source : `I(p) = L i_0 / (R + p L)`, soit `i_0 exp(-R t / L)`."""
    initial = sp.Symbol("i_0", positive=True)
    bobine = Bobine("b", "masse", INDUCTANCE, courant_initial=initial)
    reseau = ReseauOperationnel(
        elements=(Resistance("masse", "b", RESISTANCE), bobine),
        reference="masse",
    )
    attendu = INDUCTANCE * initial / (RESISTANCE + VARIABLE * INDUCTANCE)
    assert sp.simplify(reseau.courant(bobine) - attendu) == 0


def test_la_tension_initiale_d_une_capacite_s_eteint_dans_la_resistance() -> None:
    """Sans source : `U(p) = u_0 / (p + 1/(R C))`, soit `u_0 exp(-t / (R C))`."""
    initiale = sp.Symbol("u_0", positive=True)
    reseau = ReseauOperationnel(
        elements=(
            Resistance("c", "masse", RESISTANCE),
            Capacite("c", "masse", CAPACITE, tension_initiale=initiale),
        ),
        reference="masse",
    )
    attendu = initiale / (VARIABLE + 1 / (RESISTANCE * CAPACITE))
    assert sp.simplify(reseau.potentiels()["c"] - attendu) == 0


def test_deux_bobines_couplees_obeissent_aux_equations_du_transformateur() -> None:
    """Primaire alimente, secondaire ferme sur une resistance.

    Oracle : les deux equations de maille ecrites a la main,
    `(R_1 + p L_1) I_1 + p M I_2 = E/p` et
    `p M I_1 + (R_2 + p L_2) I_2 = 0`, resolues par Cramer.
    """
    r1, r2, l1, l2, mutuelle = sp.symbols("R_1 R_2 L_1 L_2 M", positive=True)
    primaire = Bobine("a", "masse", l1)
    secondaire = Bobine("c", "masse", l2)
    reseau = ReseauOperationnel(
        elements=(
            SourceDeTension("s", "masse", ECHELON),
            Resistance("s", "a", r1),
            primaire,
            secondaire,
            Resistance("masse", "c", r2),
        ),
        couplages=(Couplage(primaire, secondaire, mutuelle),),
        reference="masse",
    )
    determinant = (r1 + VARIABLE * l1) * (r2 + VARIABLE * l2) - (
        VARIABLE * mutuelle
    ) ** 2
    attendu_1 = ECHELON * (r2 + VARIABLE * l2) / determinant
    attendu_2 = -ECHELON * VARIABLE * mutuelle / determinant
    assert sp.simplify(reseau.courant(primaire) - attendu_1) == 0
    assert sp.simplify(reseau.courant(secondaire) - attendu_2) == 0


def test_un_couplage_qui_nomme_une_bobine_absente_est_refuse() -> None:
    """Coupler une bobine qui n'est pas dans le reseau ne voudrait rien dire."""
    presente = Bobine("a", "masse", INDUCTANCE)
    absente = Bobine("x", "masse", INDUCTANCE)
    reseau = ReseauOperationnel(
        elements=(Resistance("a", "masse", RESISTANCE), presente),
        couplages=(Couplage(presente, absente, INDUCTANCE),),
        reference="masse",
    )
    with pytest.raises(ValueError, match="bobine absente"):
        reseau.potentiels()


def test_un_noeud_flottant_est_refuse() -> None:
    """Meme garde que le reseau stationnaire : pas de solution unique."""
    reseau = ReseauOperationnel(
        elements=(
            Resistance("a", "masse", RESISTANCE),
            Resistance("b", "c", RESISTANCE),
        ),
        reference="masse",
    )
    with pytest.raises(ValueError, match="pas de solution unique"):
        reseau.potentiels()


def test_la_source_debite_le_courant_qui_traverse_la_bobine() -> None:
    """Une seule maille : ce qui sort de la source est ce qui entre dans `L`.

    Les deux courants sont rendus dans des conventions differentes --
    sortant par la borne plus pour la source, de `de` vers `vers` pour la
    bobine -- et ici les deux coincident. Sans ce test, rien ne tenait
    le signe du courant d'une source.
    """
    source = SourceDeTension("s", "masse", ECHELON)
    bobine = Bobine("b", "masse", INDUCTANCE)
    reseau = ReseauOperationnel(
        elements=(source, Resistance("s", "b", RESISTANCE), bobine),
        reference="masse",
    )
    assert sp.simplify(reseau.courant(source) - reseau.courant(bobine)) == 0
