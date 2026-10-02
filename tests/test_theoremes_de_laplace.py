"""Les deux theoremes de valeur, et la condition que le cours ne dit pas.

Ils servent l'exercice 5 du regime transitoire, ou ils levent une
ambiguite de notation du corrige manuscrit : celui-ci note `i_2(0) = 0`
la valeur initiale, puis `i_2(0) = E_2 / (R_2 + R_3)` la valeur de
regime que la troisieme question de l'enonce demande. Les deux sont
justes, mais ne parlent pas du meme instant. Les theoremes de valeur
disent lequel est lequel, en calculant les deux limites a partir de la
**meme** transformee.

Et ils ne sont pas gratuits : le theoreme de la valeur finale demande que
la limite temporelle existe. Pour un oscillateur non amorti elle n'existe
pas, et le theoreme rend un nombre tout de meme -- un nombre faux. Le
dernier test de ce fichier le montre, parce qu'une garde qu'on applique
sans connaitre sa condition est un piege.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    ecart_des_degres,
    original,
    valeur_finale,
    valeur_initiale,
)

CONSTANTE = sp.Symbol("A", positive=True)
AMORTISSEMENT = sp.Symbol("alpha", positive=True)


def test_la_valeur_finale_d_un_echelon_est_sa_hauteur() -> None:
    """`A/p` est la transformee de `A . Heaviside(t)`, qui tend vers `A`.

    Le cas le plus simple, et celui qui fixe la convention : le facteur
    `p` du theoreme annule exactement le `1/p` de l'echelon.
    """
    assert sp.simplify(valeur_finale(CONSTANTE / VARIABLE) - CONSTANTE) == 0


def test_la_valeur_initiale_d_un_echelon_est_aussi_sa_hauteur() -> None:
    """Un echelon vaut `A` des l'instant zero-plus, pas seulement au loin.

    C'est ce qui distingue les deux theoremes : ils ne diffèrent que
    lorsque le signal **evolue**, et les comparer revele donc un
    transitoire.
    """
    assert sp.simplify(valeur_initiale(CONSTANTE / VARIABLE) - CONSTANTE) == 0


def test_les_deux_valeurs_d_une_exponentielle_decroissante_differrent() -> None:
    """`A/(p+alpha)` part de `A` et tend vers zero : le cas d'ecole.

    C'est le transitoire type, et c'est la forme que prendra chaque
    courant de l'exercice. La difference entre les deux limites **est**
    le transitoire -- s'il n'y en avait pas, les deux coincideraient.
    """
    transformee = CONSTANTE / (VARIABLE + AMORTISSEMENT)
    assert sp.simplify(valeur_initiale(transformee) - CONSTANTE) == 0
    assert sp.simplify(valeur_finale(transformee)) == 0


def test_la_valeur_initiale_voit_le_saut_qu_un_condensateur_permet() -> None:
    """Une transformee en `1/(p + alpha)` **saute** a l'instant zero-plus.

    Et c'est physique : un courant peut etre discontinu quand sa branche
    ne porte pas d'inductance. C'est ce qui arrive au courant `i_1` de
    l'exercice, dont la branche est purement `R-C` -- et que le manuscrit
    annonce nul alors qu'il saute.
    """
    transformee = CONSTANTE / (VARIABLE + AMORTISSEMENT)
    assert valeur_initiale(transformee) != 0


def test_la_valeur_finale_d_un_oscillateur_non_amorti_est_un_faux_resultat() -> None:
    """La condition que le theoreme cache, et qui le rend dangereux.

    `omega / (p^2 + omega^2)` est la transformee de `sin(omega t)`, qui
    **n'a pas** de limite : elle oscille indefiniment. Le theoreme rend
    pourtant zero, sans rien signaler.

    Ce test ne verifie donc pas une propriete souhaitable, il **fixe le
    piege** : la valeur finale n'est licite que si tous les poles sont a
    partie reelle strictement negative. Dans l'exercice ils le sont,
    parce que chaque maille porte une resistance -- et c'est ce qui
    autorise l'emploi du theoreme.
    """
    pulsation = sp.Symbol("omega", positive=True)
    oscillateur = pulsation / (VARIABLE**2 + pulsation**2)
    assert sp.simplify(valeur_finale(oscillateur)) == 0
    assert sp.limit(sp.sin(pulsation * sp.Symbol("t", positive=True)), "t", sp.oo) != 0


def test_l_ecart_des_degres_d_un_echelon_vaut_un() -> None:
    """`A/p` : numerateur de degre zero, denominateur de degre un.

    Et sa valeur initiale est non nulle. C'est le cas de reference du
    critere que les deux tests suivants etablissent.
    """
    assert ecart_des_degres(CONSTANTE / VARIABLE) == 1


def test_un_ecart_de_degres_egal_a_un_annonce_une_valeur_initiale_non_nulle() -> None:
    """Le critere, et il est structurel : il se lit sans calculer la limite.

    `p F(p)` se comporte au loin comme `p^{1 - ecart}`. Avec un ecart de
    un, c'est une constante -- donc une valeur initiale finie et non
    nulle, c'est-a-dire un **saut**. C'est ce qui arrive au courant `i_1`
    de l'exercice du regime transitoire.
    """
    transformee = CONSTANTE / (VARIABLE + AMORTISSEMENT)
    assert ecart_des_degres(transformee) == 1
    assert valeur_initiale(transformee) != 0


def test_un_ecart_de_degres_superieur_a_un_annonce_un_depart_de_zero() -> None:
    """L'autre moitie du critere, sans laquelle il ne trancherait rien.

    Avec un ecart de deux, `p F(p)` se comporte comme `1/p` et tend vers
    zero : le signal part de zero, continument. C'est le cas du courant
    `i_2`, dont la branche porte une inductance qui interdit le saut.
    """
    transformee = CONSTANTE / (VARIABLE + AMORTISSEMENT) ** 2
    assert ecart_des_degres(transformee) == 2
    assert sp.simplify(valeur_initiale(transformee)) == 0


def test_l_original_d_un_etablissement_de_courant_est_une_exponentielle() -> None:
    """`E / (p (R + p L))` revient a `(E/R) (1 - exp(-R t / L))`.

    Le retour dans le temps est la derniere etape de toute resolution par
    Laplace, et celle ou les corriges manuscrits decomposent en elements
    simples a la main. L'oracle est la reponse du premier ordre.
    """
    tension, resistance, inductance = sp.symbols("E R L", positive=True)
    transformee = tension / (VARIABLE * (resistance + VARIABLE * inductance))
    attendu = tension / resistance * (1 - sp.exp(-resistance * TEMPS / inductance))
    assert sp.simplify(original(transformee) - attendu) == 0


def test_l_original_d_une_constante_sur_p_est_cette_constante() -> None:
    """`a / p` revient a `a` : un echelon, pour `t > 0`."""
    hauteur = sp.Symbol("a", positive=True)
    assert sp.simplify(original(hauteur / VARIABLE) - hauteur) == 0


def test_l_original_respecte_les_deux_theoremes_de_valeur() -> None:
    """Les limites de l'original sont celles que les theoremes annoncent.

    Trois calculs distincts -- deux limites en `p`, une inversion -- et
    ils doivent s'accorder aux deux bouts.
    """
    tension, resistance, capacite = sp.symbols("E R C", positive=True)
    transformee = tension / (VARIABLE * (1 + VARIABLE * resistance * capacite))
    temporel = original(transformee)
    assert (
        sp.simplify(sp.limit(temporel, TEMPS, 0, "+") - valeur_initiale(transformee))
        == 0
    )
    assert (
        sp.simplify(sp.limit(temporel, TEMPS, sp.oo) - valeur_finale(transformee)) == 0
    )
