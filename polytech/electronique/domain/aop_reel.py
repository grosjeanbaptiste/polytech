"""L'amplificateur operationnel reel : ce que la masse virtuelle ne dit pas.

Planches du chapitre 2, partie I. Un amplificateur reel a une impedance
d'entree finie, `R_in`, une impedance de sortie non nulle, `R_o`, et un
gain `A_v` grand mais fini. Tant qu'il ne sature pas, sa tension
differentielle reste sous `V_pos / A_v` -- quelques dizaines de
microvolts --, et c'est ce qui autorise l'approximation `V+ ~ V-`.

Cette approximation donne le gain d'un montage. Elle ne donne pas ses
impedances, qui dependent de `A_v` : il faut alors l'analyse complete.
Pour le non-inverseur, une source d'essai a la sortie, l'entree a la
masse :

    I_out / V_out = 1 / (R_2 + R_in // R_1) + 1 / R_o
                    + (A_v / R_o) (R_in // R_1) / (R_in // R_1 + R_2),

d'ou `Z_out ~ R_o (1 + R_2 / R_1) / A_v` : la retroaction divise
l'impedance de sortie par le rapport du gain ouvert au gain ferme.
"""

from __future__ import annotations

import sympy as sp


def ecart_differentiel_maximal(alimentation: float, gain: float) -> float:
    """`|V+ - V-| <= V_pos / A_v`, hors saturation, en volts."""
    if gain <= 0:
        msg = f"gain nul ou negatif : {gain}"
        raise ValueError(msg)
    return alimentation / gain


def impedance_de_sortie_du_non_inverseur(
    sortie: float,
    gain: float,
    *,
    vers_la_masse: float,
    contre_reaction: float,
    entree: float,
) -> float:
    """`Z_out` du non-inverseur, sans masse virtuelle, en ohms.

    Un `gain` negatif reproduit le signe de la planche 20
    (`I_OA = (V_out - A_v V_-) / R_o`), qui rend une impedance negative.
    """
    parallele = entree * vers_la_masse / (entree + vers_la_masse)
    retour = parallele / (parallele + contre_reaction)
    par_le_pont = 1 / (contre_reaction + parallele)
    return 1 / (par_le_pont + 1 / sortie + gain * retour / sortie)


def impedance_d_entree_chargee(inverse: sp.Matrix, charge: object) -> sp.Expr:
    """`V_1 / I_1` d'un quadripole de matrice `G`, ferme sur `charge`.

    `I_1 = g_11 V_1 + g_12 I_2` et `V_2 = g_21 V_1 + g_22 I_2 = -R_L I_2` :
    le courant de retour `g_12` ramene la charge a l'entree.
    """
    g_11, g_12, g_21, g_22 = inverse
    ramene = g_12 * g_21 / (g_22 + sp.sympify(charge))
    return 1 / (g_11 - ramene)


def impedance_d_entree_chargee_selon_la_planche_50(
    a_vide: float, ouverte: float, fermee: float, charge: float, gains: float
) -> float:
    """La formule imprimee : le terme exact divise par `A_ol / A_cl`.

    La planche prend `Z_in,cl / Z_in,ol` pour un ; le courrier a
    l'enseignant releve que ce rapport vaut `1 + beta A_ol`.
    """
    terme = (ouverte / charge) / (1 + fermee / charge) / gains
    return a_vide / (1 + terme)
