"""Les deux theoremes de valeur de la transformee de Laplace.

Ils donnent les limites temporelles d'un signal **sans le ramener dans
le domaine du temps**, ce qui est leur interet : inverser une transformee
rationnelle a cinq poles demanderait une decomposition en fractions
simples, alors qu'une limite en `p` se calcule en une ligne.

`valeur_initiale` : `f(0^+) = lim_{p -> oo} p F(p)`.
`valeur_finale`   : `f(oo)  = lim_{p -> 0}  p F(p)`.

**La condition que les enonces passent sous silence.** Le theoreme de la
valeur finale n'est licite que si la limite temporelle existe, c'est-a-dire
si tous les poles de `F` sont a partie reelle strictement negative (plus
l'eventuel pole simple en zero d'un regime etabli non nul). Sinon il rend
un nombre tout de meme, et ce nombre est faux : pour `sin(omega t)` il
rend zero, alors que le sinus n'a pas de limite.

Dans l'exercice du regime transitoire la condition est remplie, et il
faut dire pourquoi : chaque maille porte une resistance, donc chaque mode
est amorti. Ce n'est pas une formalite -- un circuit `L-C` pur la
violerait.

**Pourquoi les comparer revele un transitoire.** Les deux limites ne
diffèrent que si le signal evolue. Leur ecart **est** le transitoire, et
leur egalite dit qu'il n'y en a pas.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS

VARIABLE: Final = sp.Symbol("p", positive=True)
"""La variable de Laplace. Prise positive : les limites sont reelles."""


def valeur_initiale(transformee: sp.Expr) -> sp.Expr:
    """`f(0^+) = lim_{p -> oo} p F(p)`, l'instant juste apres la commutation.

    Zero-**plus** et non zero : c'est la valeur apres le saut eventuel,
    pas avant. Un courant peut sauter quand sa branche ne porte pas
    d'inductance, et c'est precisement ce que ce theoreme revele.
    """
    return sp.limit(VARIABLE * transformee, VARIABLE, sp.oo)


def valeur_finale(transformee: sp.Expr) -> sp.Expr:
    """`f(oo) = lim_{p -> 0} p F(p)`, le regime etabli.

    A n'employer qu'apres avoir verifie que les poles sont amortis : voir
    la mise en garde du module. Ici chaque maille porte une resistance.
    """
    return sp.limit(VARIABLE * transformee, VARIABLE, 0)


def ecart_des_degres(transformee: sp.Expr) -> int:
    """`deg(denominateur) - deg(numerateur)`, et ce qu'il annonce.

    Critere **structurel** du comportement a l'instant zero-plus : il se
    lit sur les degres, sans calculer aucune limite. Puisque `p F(p)` se
    comporte au loin comme `p^{1 - ecart}` :

    * ecart de **un** : une constante, donc une valeur initiale finie et
      non nulle -- le signal **saute** ;
    * ecart de **deux ou plus** : tend vers zero, le signal part de zero
      continument ;
    * ecart **nul** : diverge, ce qui signale une impulsion -- un regime
      qu'un circuit a sources bornees ne produit pas.

    Dans l'exercice du regime transitoire, l'ecart vaut un pour `i_1` et
    deux pour `i_2`. C'est l'explication structurelle de ce que les
    inductances font : la branche de `i_2` en porte une, qui interdit le
    saut ; celle de `i_1` est purement `R-C`, et le saut est licite.
    """
    numerateur, denominateur = sp.fraction(sp.cancel(sp.together(transformee)))
    degre = sp.degree(sp.Poly(denominateur, VARIABLE)) - sp.degree(
        sp.Poly(numerateur, VARIABLE),
    )
    return int(degre)


def original(transformee: sp.Expr) -> sp.Expr:
    """La fonction du temps dont `transformee` est l'image, pour `t > 0`.

    C'est la derniere etape d'une resolution par Laplace. Un corrige
    manuscrit la fait par decomposition en elements simples ; ici c'est
    l'inversion symbolique, qui ne partage rien avec cette decomposition.

    Le resultat vaut pour les instants positifs : l'echelon unite que
    l'inversion ajoute est remplace par un, ce qu'il vaut alors.
    """
    temps = sp.Symbol("t", positive=True)
    inverse = sp.inverse_laplace_transform(transformee, VARIABLE, temps)
    return _exposants_reduits(sp.simplify(inverse.subs(temps, TEMPS)))


def _exposants_reduits(expression: sp.Expr) -> sp.Expr:
    """Reduit chaque exposant d'exponentielle a sa fraction la plus simple.

    L'inversion rend volontiers `exp(-t (a + b)^2 / (c (a + b)))`, que la
    simplification generale ne reconnait pas pour `exp(-t (a + b) / c)`.
    Deux ecritures du meme amortissement ne se compareraient alors pas.
    """
    return expression.replace(
        lambda terme: isinstance(terme, sp.exp),
        lambda terme: sp.exp(sp.factor(sp.cancel(terme.args[0]))),
    )
