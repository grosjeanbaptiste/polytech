"""Les processus aleatoires stationnaires : autocorrelation et densite spectrale.

Un processus stationnaire au sens large a une moyenne constante et une
autocorrelation `R(tau) = E[X(t) X(t + tau)]` qui ne depend que du
retard. Sa densite spectrale de puissance `S(f)` dit comment sa puissance
se repartit en frequence, et le theoreme de Wiener-Khintchine les relie :

    S(f) = int R(tau) e^(-j 2 pi f tau) dtau,
    R(tau) = int S(f) e^(+j 2 pi f tau) df.

D'ou `R(0) = int S df` : la variance d'un processus centre. La densite
est paire pour un processus reel, s'ajoute pour des processus
independants, et se multiplie par `|H(f)|^2` a la traversee d'un systeme
lineaire invariant.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.fourier.transformee import integrale_sur_la_droite


def densite_depuis_l_autocorrelation(
    autocorrelation: sp.Expr, retard: sp.Symbol, frequence: sp.Symbol
) -> sp.Expr:
    """`S(f) = int R(tau) e^(-j 2 pi f tau) dtau`."""
    noyau = sp.exp(-2 * sp.pi * sp.I * frequence * retard)
    integrale = integrale_sur_la_droite(autocorrelation * noyau, retard)
    return sp.simplify(sp.re(sp.simplify(integrale)))


def autocorrelation_depuis_la_densite(
    densite: sp.Expr, frequence: sp.Symbol, retard: sp.Symbol
) -> sp.Expr:
    """`R(tau) = int S(f) e^(+j 2 pi f tau) df`.

    `S` est paire : l'integrale se reduit au cosinus, et `R` est paire
    elle aussi. Elle se calcule donc pour un retard positif, puis se lit
    en `|tau|`.
    """
    positif = sp.Dummy("tau", positive=True)
    noyau = sp.cos(2 * sp.pi * frequence * positif)
    bornes = (frequence, -sp.oo, sp.oo)
    demi_droite = sp.integrate(densite * noyau, bornes, conds="none")
    return sp.simplify(demi_droite).subs(positif, sp.Abs(retard))


def densite_de_la_somme(*densites: sp.Expr) -> sp.Expr:
    """La densite d'une somme de processus **independants** : leur somme."""
    if not densites:
        msg = "aucune densite : la somme n'a pas d'objet"
        raise ValueError(msg)
    return sum(densites, sp.Integer(0))


def densite_en_sortie(reponse: sp.Expr, densite: sp.Expr) -> sp.Expr:
    """`S_YY(f) = |H(f)|^2 S_XX(f)`."""
    return sp.simplify(sp.Abs(reponse) ** 2 * densite)


def puissance_dans_la_bande(
    bilaterale: sp.Expr, frequence: sp.Symbol, bande: tuple[sp.Expr, sp.Expr]
) -> sp.Expr:
    """`2 int_(f_l)^(f_h) S(f) df` : l'integrale de la densite unilaterale."""
    basse, haute = bande
    if not sp.sympify(haute - basse).is_positive or sp.sympify(basse).is_negative:
        msg = f"bande vide ou inversee : ({basse}, {haute})"
        raise ValueError(msg)
    return 2 * sp.integrate(bilaterale, (frequence, basse, haute))


def part_de_puissance_repliee(
    bilaterale: sp.Expr, frequence: sp.Symbol, cadence: object
) -> sp.Expr:
    """`int_(f_s/2)^oo S df / int_0^(f_s/2) S df` : ce que l'echantillonnage replie."""
    moitie = sp.nsimplify(cadence, rational=True) / 2
    if not moitie.is_positive:
        msg = f"cadence d'echantillonnage nulle ou negative : {cadence} Hz"
        raise ValueError(msg)
    gardee = sp.integrate(bilaterale, (frequence, 0, moitie))
    repliee = sp.integrate(bilaterale, (frequence, moitie, sp.oo))
    return sp.simplify(repliee / gardee)
