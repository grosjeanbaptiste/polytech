"""Les proprietes de la transformee de Fourier, en notation `f`.

Syllabus de signaux et systemes, tableau 2.1. Chacune dit ce qu'une
operation sur le signal fait a ses exponentielles imaginaires :

* retarder ne change pas leur amplitude et ajoute une phase lineaire,
  `f(t - t_0) <-> F(f) exp(-j 2 pi f t_0)` ;
* reflechir un signal reel change le signe de leur phase,
  `f(-t) <-> F*(f)` ;
* dilater comprime, `f(a t) <-> F(f / a) / |a|` ;
* deriver multiplie par `j omega` : amplitude lineaire en frequence,
  phase de plus ou moins 90 degres ;
* la dualite echange les roles, `F(t) <-> f(-f)`, parce que la
  transformee et son inverse ne different que d'un signe.
"""

from __future__ import annotations

import sympy as sp

from polytech.signaux.domain.frequence import FREQUENCE, INSTANT


def du_decalage_temporel(image: sp.Expr, retard: sp.Expr) -> sp.Expr:
    """L'image de `f(t - t_0)`."""
    return image * sp.exp(-2 * sp.pi * sp.I * FREQUENCE * retard)


def de_la_reflexion(image: sp.Expr) -> sp.Expr:
    """L'image de `f(-t)`, pour un signal reel : la conjuguee."""
    return sp.conjugate(image)


def du_changement_d_echelle(image: sp.Expr, facteur: sp.Expr) -> sp.Expr:
    """L'image de `f(a t)`."""
    if facteur == 0:
        msg = "facteur d'echelle nul : le signal dilate n'est plus un signal"
        raise ValueError(msg)
    return image.subs(FREQUENCE, FREQUENCE / facteur) / sp.Abs(facteur)


def de_la_derivee(image: sp.Expr) -> sp.Expr:
    """L'image de `df/dt` : `j omega F(f)`, avec `omega = 2 pi f`."""
    return 2 * sp.pi * sp.I * FREQUENCE * image


def de_la_dualite(signal: sp.Expr) -> sp.Expr:
    """L'image de `F(t)`, connaissant `f` : c'est `f(-f)`."""
    return signal.subs(INSTANT, -FREQUENCE)
