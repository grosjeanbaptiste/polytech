"""Le TP 2 d'electronique : impedances, diviseurs, superposition, phaseurs.

La bibliotheque combine les impedances par les formules du cours (serie,
parallele, diviseur). Ce fichier refait chaque resultat par un autre
chemin : la loi des noeuds ecrite a la main sur le circuit, resolue par
sympy. Les deux ne partagent aucune formule.

Une difference avec l'officiel est etablie : a l'exercice 5.3, la source
de courant tire son courant du noeud de sortie vers la masse (fleche vers
le bas) ; sa contribution est `-Z' I_s`, et non `+Z' I_s`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.application.circuits_tp_02 import (
    contributions_du_5_3,
    courants_du_3_1,
    impedances_du_2,
    noeuds_du_5_2,
    sortie_du_5_1,
    transferts_du_4_1,
    transferts_du_6_1,
)
from polytech.electronique.application.macros_tp_02 import paire_de_racines
from polytech.electronique.application.symboles_tp_02 import (
    CL,
    I1,
    I2,
    IS,
    R1,
    R2,
    R3,
    RL,
    RS,
    V1,
    V2,
    VIN,
    C,
    E,
    L,
    R,
    S,
    Z,
)
from polytech.electronique.domain.laplace import (
    impedances_en_parallele,
    impedances_en_serie,
)
from polytech.electronique.domain.netliste import (
    Admittance,
    SourceDeCourant,
    resoudre,
)


def _resoudre(
    equations: list[sp.Expr], inconnues: list[sp.Symbol]
) -> dict[sp.Symbol, sp.Expr]:
    (solution,) = sp.linsolve(equations, inconnues)
    return dict(zip(inconnues, solution, strict=True))


def _nul(expression: sp.Expr) -> bool:
    return bool(sp.simplify(expression) == 0)


def test_2_2_l_impedance_par_les_noeuds() -> None:
    """Une source de courant unite ; V au point d'entree, Vm entre L et R//C."""
    v, vm = sp.symbols("v vm")
    sol = _resoudre(
        [(v - vm) / (S * L) - 1, (vm - v) / (S * L) + vm / R + vm * S * C], [v, vm]
    )
    assert _nul(impedances_du_2()["2.2"] - sol[v])


def test_2_3_l_impedance_par_les_noeuds() -> None:
    """Courant unite injecte ; C a la masse, R puis L en serie a la masse."""
    v, vm = sp.symbols("v vm")
    sol = _resoudre(
        [v * S * C + (v - vm) / R - 1, (vm - v) / R + vm / (S * L)], [v, vm]
    )
    assert _nul(impedances_du_2()["2.3"] - sol[v])


def test_2_1_les_deux_impedances() -> None:
    z = impedances_du_2()
    assert _nul(z["2.1a"] - R / (1 + S * R * C))
    assert _nul(z["2.1b"] - (R + S * L))


def test_3_1_tout_le_courant_passe_par_le_court_circuit() -> None:
    i_r, i_sc = courants_du_3_1()
    assert (i_r, i_sc) == (0, sp.Symbol("I_tot"))


def test_4_1_le_premier_diviseur_donne_un_volt() -> None:
    assert transferts_du_4_1()["continu"] == 1


def test_4_1_le_troisieme_par_les_noeuds() -> None:
    v = sp.Symbol("v")
    sol = _resoudre([(v - VIN) / RS + v / Z + v / RL], [v])
    assert _nul(transferts_du_4_1()["charge"] - sol[v] / VIN)


def test_4_1_si_z_est_grande_elle_disparait() -> None:
    limite = sp.limit(transferts_du_4_1()["charge"], Z, sp.oo)
    assert _nul(limite - RL / (RS + RL))


def test_5_1_par_les_noeuds() -> None:
    v = sp.Symbol("v")
    sol = _resoudre([(v - V1) / R1 + (v - V2) / R1 + v / R2], [v])
    assert _nul(sortie_du_5_1() - sol[v])


def test_5_2_par_les_noeuds() -> None:
    a, b, c = sp.symbols("a b c")
    sol = _resoudre(
        [(a - b) / R1 - I1, (b - a) / R1 + (b - c) / R2 - I2, (c - b) / R2 + c * S * C],
        [a, b, c],
    )
    noeuds = noeuds_du_5_2()
    assert all(
        _nul(noeuds[k] - sol[x]) for k, x in zip((1, 2, 3), (a, b, c), strict=True)
    )


def test_5_3_la_source_de_courant_tire_vers_la_masse() -> None:
    """Fleche vers le bas : le courant quitte le noeud de sortie.

    Le noeud 1 est la sortie ; l'analyse nodale de la bibliotheque (netliste)
    recoit la source de `1` vers la masse (`0`) : elle tire `I_s` du noeud.
    """
    jw = sp.I * sp.Symbol("omega", positive=True)
    elements = [
        Admittance(1, 0, 1 / impedances_en_serie(R1, 1 / (jw * C))),
        Admittance(1, 0, 1 / R2),
        Admittance(1, 0, 1 / (R3 + jw * L)),
        SourceDeCourant(1, 0, IS),
    ]
    (sortie,) = resoudre(elements, 1)
    assert _nul(contributions_du_5_3()["courant"] - sortie)


def test_5_3_la_version_plus_z_prime_i_s_est_en_opposition_de_phase() -> None:
    """`V_out = -Z' I_s` ; la version `+Z' I_s` du corrige en est l'oppose.

    Le phaseur de la bibliotheque divise par celui de la version imprimee
    vaut `-1` : un demi-tour de phase dans la reponse temporelle.
    """
    jw = sp.I * sp.Symbol("omega", positive=True)
    branche = impedances_en_serie(R1, 1 / (jw * C))
    charge = impedances_en_parallele(R2, R3 + jw * L)
    imprime = IS * impedances_en_parallele(branche, charge)
    rapport = sp.simplify(contributions_du_5_3()["courant"] / imprime)
    assert rapport == -1
    assert sp.arg(rapport) == sp.pi
    assert sp.simplify(contributions_du_5_3()["courant"] + imprime) == 0


def test_5_3_la_source_alternative_par_les_noeuds() -> None:
    v = sp.Symbol("v")
    jw = sp.I * sp.Symbol("omega", positive=True)
    sol = _resoudre(
        [(v - E) * jw * C / (1 + jw * C * R1) + v / R2 + v / (R3 + jw * L)], [v]
    )
    assert _nul(contributions_du_5_3()["tension"] - sol[v])


def test_5_3_le_continu() -> None:
    v = sp.Symbol("v")
    sol = _resoudre([(v - 3) / R2 + v / R3], [v])
    assert _nul(contributions_du_5_3()["continu"] - sol[v])


def test_6_1_les_trois_transferts_par_les_noeuds() -> None:
    v = sp.Symbol("v")
    jw = sp.I * sp.Symbol("omega", positive=True)
    h = transferts_du_6_1()
    a = _resoudre([(v - VIN) * jw * C + v / R], [v])[v] / VIN
    b = _resoudre([(v - VIN) / R + v * jw * (C + CL) + v / RL], [v])[v] / VIN
    c = _resoudre([(v - VIN) / (R + RS) + v * jw * C + v / RL], [v])[v] / VIN
    assert _nul(h["a"] - a)
    assert _nul(h["b"] - b)
    assert _nul(h["c"] - c)


def test_les_paires_de_racines_annulent_leur_polynome() -> None:
    """La forme centre +- racine des macros est bien celle des racines."""
    for polynome in (C * L * R * S**2 + L * S + R, C * L * S**2 + C * R * S + 1):
        centre, carre = paire_de_racines(polynome)
        for signe in (1, -1):
            assert _nul(polynome.subs(S, centre + signe * sp.sqrt(carre)))
