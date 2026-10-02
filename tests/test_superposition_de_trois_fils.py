"""L'exercice 11 : force par unite de longueur sur un conducteur.

Entierement symbolique -- l'enonce ne donne aucun nombre -- et c'est ce
qui en fait le plus severe des exercices du chapitre : il n'y a pas
d'application numerique pour rattraper une erreur de signe.

Trois choses s'y verifient, et sympy les verifie par son propre chemin.

**La superposition.** Les trois fils contribuent chacun `I/(2 pi d)`, et
leurs **directions** diffferent : deux portent `u_y` en sens opposes, le
troisieme `-u_x`. Additionner les modules serait l'erreur, et le resultat
serait plus grand -- une force qui ne s'annule jamais.

**Le produit vectoriel.** `u_z ^ u_x = u_y` et `u_z ^ u_y = -u_x` : la
force est donc **tournee** de quatre-vingt-dix degres par rapport au
champ, et ses composantes s'echangent en changeant un signe. C'est la ou
une faute passe le plus facilement.

**La forme developpee du module.** Le manuscrit developpe le carre de la
difference, et le terme croise `-2 I_1 I_3/(d_1 d_3)` porte une
information physique : les fils 1 et 3 **s'opposent**, et leurs
contributions peuvent s'annuler. Le test verifie que la forme developpee
et la forme compacte sont la meme expression.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.superposition import (
    Fil,
    champ_de_trois_fils,
    force_par_unite_de_longueur,
    module_de_la_force,
)

MU0, COURANT = sp.symbols("mu_0 I", positive=True)
I1, I2, I3 = sp.symbols("I_1 I_2 I_3", positive=True)
D1, D2, D3 = sp.symbols("d_1 d_2 d_3", positive=True)


def test_le_champ_resultant_a_la_forme_de_la_transcription() -> None:
    """Deux composantes, et la seconde est une **difference**.

    Les fils 1 et 3 portent tous deux `u_y`, mais en sens opposes : leurs
    contributions se retranchent. Un corrige qui les additionnerait
    obtiendrait un champ qui ne s'annule jamais.
    """
    champ_x, champ_y = champ_de_trois_fils(Fil(I1, D1), Fil(I2, D2), Fil(I3, D3))
    assert sp.simplify(champ_x - (-I2 / (2 * sp.pi * D2))) == 0
    assert sp.simplify(champ_y - (I1 / D1 - I3 / D3) / (2 * sp.pi)) == 0


def test_le_champ_s_annule_quand_les_deux_fils_opposes_se_compensent() -> None:
    """`I_1/d_1 = I_3/d_3` annule la composante `u_y`.

    C'est le cas particulier que la forme en difference rend visible, et
    qu'une somme de modules aurait cache. Il n'est pas d'ecole : c'est
    ainsi qu'on neutralise un champ parasite dans un cablage.
    """
    _, champ_y = champ_de_trois_fils(Fil(I1, D1), Fil(I1, D2), Fil(I1, D1))
    assert sp.simplify(champ_y) == 0


def test_la_force_est_tournee_de_quatre_vingt_dix_degres() -> None:
    """`u_z ^ u_x = u_y` et `u_z ^ u_y = -u_x` : les composantes s'echangent.

    La force est orthogonale au champ, comme toute force de Laplace sur
    un conducteur selon `u_z`. Le test le verifie par le produit scalaire,
    qui doit etre nul -- controle independant des expressions elles-memes.
    """
    champ = champ_de_trois_fils(Fil(I1, D1), Fil(I2, D2), Fil(I3, D3))
    force = force_par_unite_de_longueur(COURANT, champ, MU0)
    produit_scalaire = force[0] * champ[0] + force[1] * champ[1]
    assert sp.simplify(produit_scalaire) == 0


def test_la_force_concorde_avec_la_forme_du_manuscrit() -> None:
    """Les deux composantes, telles que la transcription les ecrit."""
    champ = champ_de_trois_fils(Fil(I1, D1), Fil(I2, D2), Fil(I3, D3))
    force_x, force_y = force_par_unite_de_longueur(COURANT, champ, MU0)
    facteur = MU0 * COURANT / (2 * sp.pi)
    assert sp.simplify(force_x - facteur * (I3 / D3 - I1 / D1)) == 0
    assert sp.simplify(force_y - facteur * (-I2 / D2)) == 0


def test_le_module_compact_et_le_module_developpe_sont_la_meme_expression() -> None:
    """Le manuscrit donne les deux, et leur egalite n'est pas evidente a l'oeil.

    Le developpement fait apparaitre un terme croise
    `-2 I_1 I_3/(d_1 d_3)` qui porte toute l'information physique : les
    fils 1 et 3 s'opposent. Un corrige qui developperait en mettant `+`
    obtiendrait une expression toujours superieure, donc jamais nulle.
    """
    champ = champ_de_trois_fils(Fil(I1, D1), Fil(I2, D2), Fil(I3, D3))
    module = module_de_la_force(COURANT, champ, MU0)
    facteur = MU0 * COURANT / (2 * sp.pi)
    compact = facteur * sp.sqrt(I2**2 / D2**2 + (I3 / D3 - I1 / D1) ** 2)
    developpe = facteur * sp.sqrt(
        I2**2 / D2**2 + I3**2 / D3**2 + I1**2 / D1**2 - 2 * I1 * I3 / (D1 * D3),
    )
    assert sp.simplify(module - compact) == 0
    assert sp.simplify(module - developpe) == 0


def test_le_terme_croise_est_bien_negatif() -> None:
    """La garde qui attrape l'erreur de signe du developpement.

    Avec un `+` a la place du `-`, l'expression resterait toujours
    superieure au module de la seule contribution de `I_2`, donc ne
    pourrait jamais descendre en dessous -- alors que le vrai module
    l'atteint exactement quand `I_1/d_1 = I_3/d_3`.
    """
    champ = champ_de_trois_fils(Fil(I1, D1), Fil(I2, D2), Fil(I1, D1))
    module = module_de_la_force(COURANT, champ, MU0)
    seul_deux = MU0 * COURANT * I2 / (2 * sp.pi * D2)
    assert sp.simplify(module - seul_deux) == 0


def test_un_courant_nul_dans_le_conducteur_annule_la_force() -> None:
    """`dF/dz = mu_0 I u_z ^ H` : sans courant, pas de force.

    Evident, et c'est pourquoi il faut le verifier : une expression qui
    aurait perdu son facteur `I` passerait tous les autres tests, qui
    portent sur la direction et sur les rapports.
    """
    champ = champ_de_trois_fils(Fil(I1, D1), Fil(I2, D2), Fil(I3, D3))
    force = force_par_unite_de_longueur(sp.Integer(0), champ, MU0)
    assert sp.simplify(force[0]) == 0
    assert sp.simplify(force[1]) == 0
