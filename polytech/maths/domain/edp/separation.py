"""La methode de Fourier sur un domaine borne : separer, puis superposer.

On cherche `u(x, t) = a(x) b(t)`. L'equation donne `a'' / a = cte = lambda`,
et les bords `a(0) = a(l) = 0` n'admettent que `lambda = -(k pi / l)^2` :
`a(x) = sin(k pi x / l)`. L'equation etant lineaire et homogene, toute
combinaison de ces modes est solution ; les coefficients sont ceux de la
serie de Fourier en sinus des conditions initiales.

* **Diffusion** `u_t = D u_xx` : chaque mode s'eteint en
  `e^(-D (k pi / l)^2 t)`.
* **Corde vibrante** `u_xx = u_tt / c^2` : chaque mode oscille,
  `(C_k cos(k pi c t / l) + D_k sin(k pi c t / l)) sin(k pi x / l)`, avec
  `C_k` tire de la position initiale et `D_k` de la vitesse initiale.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import TEMPS, VARIABLE


def _nombre_d_onde(rang: int, longueur: object) -> sp.Expr:
    return rang * sp.pi / sp.sympify(longueur)


def constante_de_separation(rang: int, longueur: object) -> sp.Expr:
    """`lambda = -(k pi / l)^2`, la seule valeur que les bords tolerent."""
    return -(_nombre_d_onde(rang, longueur) ** 2)


def _coefficient_en_sinus(fonction: sp.Expr, rang: int, longueur: object) -> sp.Expr:
    """`(2/l) int_0^l f(x) sin(k pi x / l) dx`."""
    mode = sp.sin(_nombre_d_onde(rang, longueur) * VARIABLE)
    integrale = sp.integrate(fonction * mode, (VARIABLE, 0, longueur))
    return sp.simplify(2 * integrale / sp.sympify(longueur))


def mode_de_diffusion(rang: int, *, longueur: object, diffusivite: object) -> sp.Expr:
    """`sin(k pi x / l) e^(-D (k pi / l)^2 t)`."""
    onde = _nombre_d_onde(rang, longueur)
    extinction = sp.exp(-sp.sympify(diffusivite) * onde**2 * TEMPS)
    return sp.sin(onde * VARIABLE) * extinction


def diffusion_dans_une_barre(
    initiale: sp.Expr, *, longueur: object, diffusivite: object, ordre: int
) -> sp.Expr:
    """La solution de `u_t = D u_xx`, nulle aux deux bouts, tronquee a `ordre` modes."""
    modes = (
        _coefficient_en_sinus(initiale, k, longueur)
        * mode_de_diffusion(k, longueur=longueur, diffusivite=diffusivite)
        for k in range(1, ordre + 1)
    )
    return sum(modes, sp.Integer(0))


def corde_vibrante(
    position: sp.Expr,
    vitesse: sp.Expr,
    *,
    longueur: object,
    celerite: object,
    ordre: int,
) -> sp.Expr:
    """La corde fixee aux deux bouts, de position et de vitesse initiales donnees."""
    termes = []
    for k in range(1, ordre + 1):
        onde = _nombre_d_onde(k, longueur)
        pulsation = onde * sp.sympify(celerite)
        en_cosinus = _coefficient_en_sinus(position, k, longueur)
        en_sinus = _coefficient_en_sinus(vitesse, k, longueur) / pulsation
        temporel = en_cosinus * sp.cos(pulsation * TEMPS) + en_sinus * sp.sin(
            pulsation * TEMPS
        )
        termes.append(temporel * sp.sin(onde * VARIABLE))
    return sum(termes, sp.Integer(0))
