"""Les criteres de convergence d'une serie que l'annexe A enonce.

Mathematiques appliquees, annexe A.2 ; les notes de 2011 les donnent avec
leurs exemples (pages 121 et 122). `numeriques.py` porte la convergence
absolue, le quotient, la racine et les series alternees ; voici le reste :

1. **Cauchy** -- la serie converge si et seulement si la tranche
   `|u_(n+1) + ... + u_(n+p)|` devient petite quel que soit `p` ;
2. si la serie converge, son terme general tend vers 0 ;
4. a termes positifs, elle converge si et seulement si ses sommes
   partielles sont bornees ;
5. **comparaison**, a termes positifs : `0 <= u_n <= v_n` ;
6. **equivalence**, a termes positifs, et sa remarque sur les limites 0 et
   l'infini.

Une suite reelle est de Cauchy si et seulement si elle converge (A.1).
"""

from __future__ import annotations

from enum import Enum

import sympy as sp


class Comparaison(Enum):
    """Ce que dit la limite de `u_n / v_n`."""

    EQUIVALENTES = "meme nature : elles convergent ou divergent ensemble"
    NEGLIGEABLE = "limite nulle : si somme v_n converge, somme u_n converge"
    PREPONDERANTE = "limite infinie : si somme u_n converge, somme v_n converge"


def tranche_de_cauchy(terme: sp.Expr, indice: sp.Symbol, rang: int, p: int) -> sp.Expr:
    """`|u_(n+1) + ... + u_(n+p)|`, pour `n = rang`."""
    if p < 1:
        msg = f"une tranche compte au moins un terme : p = {p}"
        raise ValueError(msg)
    termes = (terme.subs(indice, rang + k) for k in range(1, p + 1))
    return sp.Abs(sum(termes, sp.Integer(0)))


def le_terme_general_tend_vers_zero(terme: sp.Expr, indice: sp.Symbol) -> bool:
    """Critere 2 : condition necessaire, non suffisante, de convergence."""
    return bool(sp.limit(terme, indice, sp.oo) == 0)


def est_de_cauchy(terme: sp.Expr, indice: sp.Symbol) -> bool:
    """Vrai si la suite a une limite finie : dans R, c'est la meme chose."""
    limite = sp.limit_seq(terme, indice)
    if limite is None or isinstance(limite, sp.AccumBounds):
        return False
    return bool(limite.is_finite)


def serie_de_riemann_converge(alpha: object) -> bool:
    """`somme 1 / n^alpha` converge si et seulement si `alpha > 1`."""
    ecart = sp.sympify(alpha) - 1
    if ecart.is_positive is None:
        msg = f"l'exposant {alpha} ne se compare pas a 1"
        raise ValueError(msg)
    return bool(ecart.is_positive)


def _exiger_positif(terme: sp.Expr, role: str) -> None:
    if not sp.simplify(terme).is_nonnegative:
        msg = f"critere reserve aux series a termes positifs : {role} = {terme}"
        raise ValueError(msg)


def borne_des_sommes_partielles(terme: sp.Expr, indice: sp.Symbol) -> sp.Expr:
    """Critere 4 : la borne superieure des `S_n`, infinie si la serie diverge."""
    _exiger_positif(terme, "u_n")
    return sp.summation(terme, (indice, 1, sp.oo))


def _exiger_l_encadrement(petit: sp.Expr, grand: sp.Expr) -> None:
    _exiger_positif(petit, "le plus petit terme")
    if not sp.simplify(sp.together(grand - petit)).is_nonnegative:
        msg = f"l'encadrement 0 <= {petit} <= {grand} n'est pas etabli pour tout n"
        raise ValueError(msg)


def converge_par_majoration(
    terme: sp.Expr, majorante: sp.Expr, indice: sp.Symbol
) -> bool:
    """Critere 5 : `0 <= u_n <= v_n` et `somme v_n` converge."""
    _exiger_l_encadrement(terme, majorante)
    if not sp.Sum(majorante, (indice, 1, sp.oo)).is_convergent():
        msg = f"la majorante {majorante} diverge : aucune conclusion"
        raise ValueError(msg)
    return True


def diverge_par_minoration(
    terme: sp.Expr, minorante: sp.Expr, indice: sp.Symbol
) -> bool:
    """Critere 5 : `0 <= v_n <= u_n` et `somme v_n` diverge."""
    _exiger_l_encadrement(minorante, terme)
    if sp.Sum(minorante, (indice, 1, sp.oo)).is_convergent():
        msg = f"la minorante {minorante} converge : aucune conclusion"
        raise ValueError(msg)
    return True


def comparer_par_le_quotient(
    premiere: sp.Expr, seconde: sp.Expr, indice: sp.Symbol
) -> Comparaison:
    """Critere 6 et sa remarque : la limite de `u_n / v_n`, a termes positifs."""
    _exiger_positif(premiere, "u_n")
    _exiger_positif(seconde, "v_n")
    limite = sp.limit(premiere / seconde, indice, sp.oo)
    if limite == 0:
        return Comparaison.NEGLIGEABLE
    return Comparaison.PREPONDERANTE if limite == sp.oo else Comparaison.EQUIVALENTES
