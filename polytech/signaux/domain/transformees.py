"""Les transformees du cours, et les proprietes qui servent vraiment.

Le cours en retient trois : la linearite, le retard -- qui devient un
facteur `exp(-p tau)` -- et le theoreme de la derivation, qui fait entrer
la condition initiale dans l'algebre. Ce sont elles que les exercices
emploient, et elles seules sont exposees ici ; une table exhaustive serait
une table, pas un geste.

La transformee de Fourier n'est pas celle de Laplace avec `p = j omega` :
elle integre de moins l'infini a plus l'infini, Laplace de zero a plus
l'infini. Sur un signal causal les deux coincident, et c'est seulement
dans ce cas que le cours se permet le raccourci. La garde le rappelle.
"""

from __future__ import annotations

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE, PULSATION, TEMPS


def transformee_de_laplace(signal: sp.Expr) -> sp.Expr:
    """`L{x}(p)` ; leve si sympy ne conclut pas."""
    image = sp.laplace_transform(signal, TEMPS, IMAGE, noconds=True)
    if image.has(sp.LaplaceTransform):
        msg = f"transformee de Laplace non calculable : {signal}"
        raise ValueError(msg)
    return sp.simplify(image)


def retard(image: sp.Expr, duree: sp.Expr) -> sp.Expr:
    """Retarder de `tau` multiplie l'image par `exp(-p tau)`."""
    return sp.simplify(image * sp.exp(-IMAGE * duree))


def derivation(image: sp.Expr, valeur_initiale: sp.Expr) -> sp.Expr:
    """`L{x'} = p X(p) - x(0)` : la propriete qui rend la methode utile."""
    return sp.simplify(IMAGE * image - valeur_initiale)


def fourier_depuis_laplace(image: sp.Expr) -> sp.Expr:
    """`X(j omega)` pour un signal **causal**, par substitution `p = j omega`.

    Le raccourci n'est licite que sur un signal causal : Fourier integre
    sur toute la droite reelle, Laplace sur la demi-droite positive, et
    les deux ne coincident que si le signal est nul avant zero.
    """
    return sp.simplify(image.subs(IMAGE, sp.I * PULSATION))


def fourier_d_une_exponentielle_causale(taux: sp.Expr) -> sp.Expr:
    """`1 / (a + j omega)` : la transformee la plus utilisee du cours."""
    return 1 / (taux + sp.I * PULSATION)


def amplitude_et_phase(spectre: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Le module et l'argument du spectre, separes.

    Le fascicule les trace sur deux diagrammes parce que ce sont deux
    lectures differentes : l'une dit ce qui passe, l'autre dit quand.

    L'argument est calcule par `atan2` sur les parties reelle et
    imaginaire, et non par `arg` : sur `1/(1 + j omega)`, sympy laisse
    `arg(...)` non evalue et le corrige afficherait le symbole au lieu de
    la phase. `atan2` rend `-atan(omega)`, qui se compose.
    """
    reel, imaginaire = sp.re(spectre), sp.im(spectre)
    return sp.simplify(sp.Abs(spectre)), sp.simplify(sp.atan2(imaginaire, reel))


def dualite(signal: sp.Expr) -> sp.Expr:
    """Echange le temps et la pulsation : la propriete de dualite.

    Appliquee deux fois elle rend le signal de depart **retourne** --
    `x(-t)` et non `x(t)` -- et c'est la convention de signe qui l'impose.
    Le dire evite de croire a une involution qui n'en est pas une.
    """
    return signal.subs(TEMPS, PULSATION).subs(PULSATION, TEMPS)


def coefficient_de_serie_de_fourier(
    signal: sp.Expr,
    periode: sp.Expr,
    indice: int,
) -> sp.Expr:
    """`c_k = (1/T) int_0^T x(t) e^{-j k omega_0 t} dt`.

    C'est le pont que le fascicule etablit entre transformee et series :
    un signal periodique n'a pas un spectre continu mais une raie par
    harmonique, et ces raies **sont** ses coefficients de Fourier.
    """
    if sp.simplify(periode) == 0:
        msg = "periode nulle : un signal periodique en a une, par definition"
        raise ValueError(msg)
    fondamentale = 2 * sp.pi / periode
    noyau = sp.exp(-sp.I * indice * fondamentale * TEMPS)
    return sp.simplify(sp.integrate(signal * noyau, (TEMPS, 0, periode)) / periode)


def pulsation_de_l_estimation_numerique(
    indice: int,
    points: int,
    cadence: float,
) -> sp.Expr:
    """`omega_k = 2 pi k F_e / N` : la pulsation du k-ieme point d'une FFT.

    Le fascicule y insiste parce que c'est la source de toutes les
    erreurs de lecture d'un spectre calcule : la resolution n'est pas
    libre, elle vaut `F_e / N`. Doubler la duree de mesure est le seul
    moyen de la doubler.
    """
    if points <= 0:
        msg = f"nombre de points nul ou negatif : {points}"
        raise ValueError(msg)
    if cadence <= 0:
        msg = f"cadence d'echantillonnage nulle ou negative : {cadence} Hz"
        raise ValueError(msg)
    return 2 * sp.pi * indice * sp.Rational(cadence) / points


def puissance_moyenne_d_un_signal(signal: sp.Expr, periode: sp.Expr) -> sp.Expr:
    """`(1/T) int_0^T |x|^2 dt` : la puissance d'un signal periodique."""
    if sp.simplify(periode) == 0:
        msg = "periode nulle : la moyenne n'est pas definie"
        raise ValueError(msg)
    carre = sp.Abs(signal) ** 2
    return sp.simplify(sp.integrate(carre, (TEMPS, 0, periode)) / periode)
