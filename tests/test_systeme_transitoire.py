"""Le systeme 5x5 de l'exercice 5, resolu -- ce que le manuscrit ne fait pas.

Le corrige manuscrit pose les cinq equations, constate « cinq equations,
cinq inconnues : on peut donc calculer `I_1(p)` et `I_2(p)` », et
s'arrete la. Ce fichier les calcule, et verifie le resultat par quatre
limites dont **chacune se derive a la main** des conditions initiales que
le manuscrit lui-meme ecrit -- sans jamais servir de son propre oracle.

Les quatre derivations independantes :

* `i_2(oo) = E_2 / (R_2 + R_3)` -- c'est le resultat encadre du
  manuscrit, obtenu par son propre raisonnement de regime etabli ;
* `i_2(0^+) = 0` -- l'inductance `L_2` interdit un saut de courant, et le
  manuscrit pose bien `i_2(0) = 0` ;
* `i_1(oo) = 0` -- le condensateur `C_1` bloque le continu, ce que le
  manuscrit dit aussi ;
* `i_1(0^+) = E_2 / R_1` -- par les equations (I) et (III) du manuscrit a
  l'instant zero-plus : la (III) donne `R_3 i_3 + L_3 di_3/dt = u_{C_4}`,
  nul a cet instant, donc la (I) se reduit a
  `E_1 + (E_2 - E_1) = R_1 i_1`.

C'est cette derniere qui montre ce que la notation du manuscrit cache :
il annonce `i_1(0) = 0`, et le courant **saute** a `E_2 / R_1`. Les deux
sont vrais, l'un a zero-moins et l'autre a zero-plus.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE as VARIABLE_DE_LAPLACE,
)
from polytech.electromagnetisme.domain.transitoire.laplace import (
    ecart_des_degres,
    valeur_finale,
    valeur_initiale,
)
from polytech.electromagnetisme.domain.transitoire.systeme import (
    CAPACITES,
    COURANT_1,
    COURANT_2,
    ETAT_INITIAL_C1,
    FORCES,
    INCONNUES,
    INDUCTANCES,
    RESISTANCES,
    TENSION_C1,
    degre_caracteristique,
    equations,
    solution,
    terme_source_de_la_premiere_maille,
)


def test_le_systeme_compte_cinq_equations_pour_cinq_inconnues() -> None:
    """Le constat du manuscrit, verifie plutot que recopie."""
    assert len(equations()) == len(INCONNUES)


def test_le_systeme_admet_une_solution_et_une_seule() -> None:
    """Cinq equations pour cinq inconnues ne garantit pas l'unicite.

    Il faut encore que le determinant du systeme ne soit pas nul, et
    aucun comptage ne le dit. Le verifier est donc necessaire -- c'est la
    difference entre « on peut calculer » et « on a calcule ».
    """
    resolutions = sp.solve(equations(), list(INCONNUES), dict=True)
    assert len(resolutions) == 1


def test_la_solution_satisfait_les_cinq_equations() -> None:
    """Le controle le plus direct : on resubstitue et chaque reste est nul.

    Independant de la methode de resolution : si sympy s'etait trompe,
    une des cinq differences ne s'annulerait pas.
    """
    valeurs = solution()
    restes = [
        sp.simplify((equation.lhs - equation.rhs).subs(valeurs))
        for equation in equations()
    ]
    assert restes == [0, 0, 0, 0, 0]


def test_le_courant_de_la_maille_de_droite_tend_vers_le_resultat_encadre() -> None:
    """`i_2(oo) = E_2 / (R_2 + R_3)` : le resultat du manuscrit, retrouve.

    Oracle independant : le manuscrit l'obtient par un raisonnement de
    regime etabli -- `C_1` bloque, `C_4` est sature -- sans resoudre le
    systeme. Nous l'obtenons en resolvant, puis par le theoreme de la
    valeur finale. Deux chemins, un resultat.
    """
    resistance_2, resistance_3 = RESISTANCES[1], RESISTANCES[2]
    attendu = FORCES[1] / (resistance_2 + resistance_3)
    assert sp.simplify(valeur_finale(solution()[COURANT_2]) - attendu) == 0


def test_le_courant_de_la_maille_de_droite_part_de_zero() -> None:
    """`i_2(0^+) = 0`, parce que `L_2` interdit un saut de courant.

    Et c'est ce qui etablit que le `i_2(0)` encadre par le manuscrit ne
    peut pas designer l'instant zero : a cet instant le courant vaut
    zero, pas `E_2 / (R_2 + R_3)`.
    """
    assert sp.simplify(valeur_initiale(solution()[COURANT_2])) == 0


def test_le_courant_du_condensateur_s_annule_en_regime_etabli() -> None:
    """`i_1(oo) = 0` : un condensateur ne laisse pas passer le continu."""
    assert sp.simplify(valeur_finale(solution()[COURANT_1])) == 0


def test_le_courant_du_condensateur_saute_a_l_instant_zero_plus() -> None:
    """`i_1(0^+) = E_2 / R_1`, et le manuscrit l'annonce nul.

    La derivation a la main, a partir des equations du manuscrit a
    l'instant zero-plus : son equation (III) donne
    `R_3 i_3 + L_3 di_3/dt = u_{C_4}`, que le manuscrit pose nul a cet
    instant ; son equation (I) se reduit donc a
    `E_1 + u_{C_1}(0) = R_1 i_1`, avec `u_{C_1}(0) = E_2 - E_1`. Reste
    `E_2 = R_1 i_1(0^+)`.

    Le saut est licite : la branche de `i_1` est purement `R-C`, et seule
    une inductance interdit a un courant d'etre discontinu.
    """
    attendu = FORCES[1] / RESISTANCES[0]
    assert sp.simplify(valeur_initiale(solution()[COURANT_1]) - attendu) == 0


def test_l_ordre_du_circuit_est_le_nombre_de_ses_elements_reactifs() -> None:
    """Quatre : `C_1`, `C_4`, `L_2`, `L_3`. Et c'est ce qui tient `L_3`.

    L'ordre d'un circuit -- le degre de son polynome caracteristique --
    egale le nombre de ses elements emmagasinant de l'energie, car chacun
    apporte une variable d'etat. Le verifier n'est donc pas une
    curiosite : c'est le seul controle qui atteste que les **quatre**
    sont bien dans les equations.

    Il a ete ecrit apres qu'une mutation l'ait prouve necessaire :
    retirer `L_3` de l'impedance de la branche centrale ne faisait
    echouer aucun des sept tests precedents. Les quatre limites de valeur
    y sont insensibles -- `i_3` ne saute pas de toute facon, puisque
    `u_{C_4}(0) = 0` -- et seule l'ordre du systeme le voit : il tombe de
    quatre a trois.
    """
    assert degre_caracteristique() == len(INDUCTANCES) + len(CAPACITES)


def test_le_pole_de_la_source_ne_compte_pas_dans_l_ordre() -> None:
    """Le denominateur porte un `p` de plus : celui de l'echelon.

    Une source constante appliquee a `t = 0` a pour transformee `E/p`, et
    ce pole en zero se retrouve au denominateur de chaque courant. Il
    n'appartient pas au circuit, et le confondre avec un mode propre
    donnerait un ordre cinq pour quatre elements reactifs.
    """
    transformee = solution()[COURANT_2]
    _, denominateur = sp.fraction(sp.cancel(sp.together(transformee)))
    entier = sp.Poly(denominateur, VARIABLE_DE_LAPLACE)
    assert sp.degree(entier) == degre_caracteristique() + 1


def test_l_ecart_des_degres_distingue_le_courant_qui_saute() -> None:
    """Un pour `i_1`, deux pour `i_2` : le critere lu sur l'exercice meme.

    C'est l'explication **structurelle** de ce que les inductances font,
    et elle se lit sans calculer aucune limite : la branche de `i_2`
    porte `L_2`, qui interdit le saut ; celle de `i_1` est purement
    `R-C`, et le saut y est licite.

    Les deux ecarts sont donc redondants avec les deux limites deja
    testees -- et cette redondance est le but : deux lectures
    independantes du meme fait se confirment l'une l'autre, alors
    qu'une seule ne se confirmerait pas.
    """
    assert ecart_des_degres(solution()[COURANT_1]) == 1
    assert ecart_des_degres(solution()[COURANT_2]) == 2


def test_le_numerateur_du_courant_de_droite_est_proportionnel_a_sa_source() -> None:
    """`I_2(p)` porte `E_2` en facteur, et pas `E_1`.

    Fait que le manuscrit ne pouvait pas voir, faute d'avoir resolu : le
    courant de la maille de droite ne depend **pas** de la source de
    gauche. L'explication est que `C_1` bloque tout transfert continu
    entre les deux mailles, et que le couplage ne passe que par `C_4` --
    dont la charge initiale est nulle.

    C'est le genre de resultat qui justifie de resoudre : il ne se
    devine pas sur les equations.
    """
    numerateur, _ = sp.fraction(sp.cancel(sp.together(solution()[COURANT_2])))
    force_1, force_2 = FORCES
    assert force_1 not in numerateur.free_symbols
    assert sp.simplify(sp.expand(numerateur) / force_2).free_symbols.isdisjoint(FORCES)


def test_aucun_courant_ne_depend_de_la_source_de_gauche() -> None:
    """`E_1` disparait des trois courants, et c'est le resultat de l'exercice.

    Derivation independante, a la main et sur la seule maille (I) : elle
    porte `E_1 + u_{C_1}(t)`, et l'etat initial du condensateur est
    `u_{C_1}(0) = E_2 - E_1`. A l'instant de la fermeture, le terme
    source de cette maille vaut donc

        `E_1 + (E_2 - E_1) = E_2`,

    ou `E_1` s'est **exactement** compense. La charge accumulee par
    `C_1` pendant que l'interrupteur etait ouvert annule la source qui
    l'a chargee -- et le circuit se comporte comme si seule `E_2`
    existait.

    C'est le fait le plus frappant de l'exercice, et le manuscrit ne
    pouvait pas le voir : il faut avoir resolu pour le constater. Il ne
    se devine pas sur le systeme, ou `E_1` figure deux fois.

    `U_{C_1}` est la seule inconnue qui garde `E_1`, et c'est coherent :
    c'est la tension de ce condensateur, dont l'etat initial est
    precisement `E_2 - E_1`.
    """
    force_1 = FORCES[0]
    valeurs = solution()
    porteuses = sorted(
        str(inconnue)
        for inconnue in INCONNUES
        if force_1 in sp.simplify(valeurs[inconnue]).free_symbols
    )
    assert porteuses == [str(TENSION_C1)]


def test_le_terme_source_de_la_premiere_maille_se_reduit_a_la_seconde_force() -> None:
    """`E_1 + u_{C_1}(0) = E_2` : la compensation, calculee et non affirmee.

    Le document composera cette reduction plutot que de l'ecrire a la
    main, ce qui serait reaffirmer ce qu'on veut montrer. Le calcul
    part de l'etat initial que le manuscrit donne,
    `u_{C_1}(0) = E_2 - E_1`, et rend `E_2`.
    """
    assert sp.simplify(terme_source_de_la_premiere_maille() - FORCES[1]) == 0


def test_le_terme_source_reduit_ne_porte_plus_la_premiere_force() -> None:
    """Et `E_1` n'y figure plus : c'est la disparition elle-meme.

    Verifier la valeur ne suffit pas -- `E_2` pourrait s'ecrire sous une
    forme ou `E_1` s'annule sans etre simplifie, et le document
    composerait alors une expression ou `E_1` reste visible.
    """
    assert FORCES[0] not in terme_source_de_la_premiere_maille().free_symbols


def test_l_etat_initial_du_condensateur_de_gauche_est_la_difference_des_forces() -> (
    None
):
    """`u_{C_1}(0) = E_2 - E_1`, comme le manuscrit l'etablit.

    Pendant que l'interrupteur etait ouvert, le circuit se reduisait aux
    deux sources et aux deux resistances en serie, sans courant : la
    tension du condensateur s'etait donc etablie a la difference des
    deux forces electromotrices.
    """
    force_1, force_2 = FORCES
    assert sp.simplify(ETAT_INITIAL_C1 - (force_2 - force_1)) == 0
