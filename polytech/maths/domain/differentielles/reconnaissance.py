"""Les trois derniers cas du cours, quand `type()` n'a rien reconnu.

Le syllabus de mathematiques appliquees les traite apres la lineaire et
l'homogene (1.3.5 a 1.3.8) :
- **exacte** : `y' = -P/Q` avec `dP/dy = dQ/dx` ;
- **facteur integrant** : pas exacte, mais `h(x)` ou `h(y)` la rend telle
  (exemple 1.4) ;
- **Bernoulli** : `y' = -P(x) y + Q(x) y^a`, `a` ni 0 ni 1 (exemple 1.6).

`P` et `Q` se lisent sur la fraction ecrite, comme dans l'exemple 1.4 : la
decomposition `f = -P/Q` n'est pas unique, et c'est pourquoi `type()` ne
rend jamais `EXACTE` -- une autre ecriture de la meme equation pourrait
changer le verdict.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.differentielles.bernoulli import (
    DEJA_LINEAIRES,
    EquationDeBernoulli,
)
from polytech.maths.domain.differentielles.exactes import FormeDifferentielle
from polytech.maths.domain.differentielles.facteur_integrant import facteur_integrant
from polytech.maths.domain.differentielles.premier_ordre import (
    INCONNUE,
    EquationDuPremierOrdre,
    Type,
)


def en_bernoulli(second_membre: sp.Expr) -> EquationDeBernoulli:
    """`y' + P y = Q y^a`, lu sur `y' = f(x, y)` ; leve si `f` n'a pas cette forme."""
    puissances: dict[sp.Expr, sp.Expr] = {}
    for terme in sp.Add.make_args(sp.expand(second_membre)):
        # un terme ou y n'est pas une puissance pure (sin y, e^y) revient
        # avec l'exposant 0, et la garde ci-dessous le refuse
        coefficient, exposant = terme.as_coeff_exponent(INCONNUE)
        puissances[exposant] = puissances.get(exposant, 0) + coefficient
    autres = [e for e in puissances if e != 1]
    if 1 in puissances and len(autres) == 1 and autres[0] not in DEJA_LINEAIRES:
        return EquationDeBernoulli(
            sp.simplify(-puissances[1]), puissances[autres[0]], autres[0]
        )
    msg = f"pas de la forme de Bernoulli -P(x) y + Q(x) y^a : {second_membre}"
    raise ValueError(msg)


def type_du_cours(equation: EquationDuPremierOrdre) -> Type:
    """Le type du cours, les six cas compris ; leve si aucun ne convient."""
    try:
        return equation.type()
    except ValueError:
        pass  # ni separee, ni homogene, ni lineaire : les trois cas suivants
    p, q = sp.fraction(sp.together(-equation.second_membre))
    forme = FormeDifferentielle(sp.expand(p), sp.expand(q))
    if forme.est_exacte():
        return Type.EXACTE
    try:
        facteur_integrant(forme)
    except ValueError:
        pass  # pas de facteur integrant simple : reste Bernoulli
    else:
        return Type.FACTEUR_INTEGRANT
    try:
        en_bernoulli(equation.second_membre)
    except ValueError as erreur:
        msg = f"type non reconnu parmi ceux du cours : {equation.second_membre}"
        raise ValueError(msg) from erreur
    return Type.BERNOULLI
