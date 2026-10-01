"""Le theoreme de Vaschy-Buckingham et la methode des variables repetees.

Le depot porte un syllabus entier sur ce theoreme -- l'extrait du
chapitre 7 de Cengel -- et rien ne le portait ici, parce que la mesure de
couverture ne regardait que les planches.

Le theoreme tient en une soustraction : un probleme a `n` parametres dont
les dimensions primaires se reduisent a `j` porte `k = n - j` groupes sans
dimension. C'est l'equation 7-14 du document.

Ce qui en fait une **methode** plutot qu'une formule est que `j` se
devine : « guess the reduction j. As a first guess, set j equal to the
number of primary dimensions represented in the problem ». Le devin peut
se tromper, et le document donne la regle de repli en clair -- verifier
d'abord qu'on a liste assez de parametres a l'etape 1, sinon « go back and
reduce j by one and try again ».

La portee du theoreme se mesure a ce qu'il **predit**, et la bibliotheque
de ce depot en donne deux verifications :

* la trainee a cinq parametres et trois dimensions primaires, donc deux
  groupes -- d'ou `C_D = f(Re)`, la forme que `aerodynamique` emploie ;
* la convection forcee en a sept et quatre, la temperature s'ajoutant aux
  trois de la mecanique, donc trois groupes -- d'ou `Nu = f(Re, Pr)`, le
  nombre exact que les correlations de thermique emploient.

Les deux concordent, et c'est la verification : un theoreme mal transcrit
annoncerait un groupe de trop ou de moins, donc une correlation
introuvable ou une de trop.
"""

from __future__ import annotations

from typing import Final

ETAPES_DE_LA_METHODE: Final = (
    "lister les parametres du probleme et compter leur nombre total n",
    "lister les dimensions primaires de chacun des n parametres",
    "poser la reduction j et calculer k = n - j, le nombre attendu de groupes",
    "choisir j parametres repetees, qui paraitront dans chaque groupe",
    "construire les k groupes, et les manipuler autant qu'il faut",
    "ecrire la relation fonctionnelle finale et verifier l'algebre",
)
"""Les six etapes, dans l'ordre du document.

L'ordre porte le sens : on ne choisit les parametres repetees qu'en
quatrieme position, parce que leur nombre **est** `j`, et que `j` ne se
connait qu'apres avoir recense les dimensions primaires.
"""


def nombre_de_groupes(parametres: int, reduction: int) -> int:
    """`k = n - j` : le theoreme de Vaschy-Buckingham, equation 7-14.

    A reduction nulle il rend `n` : un probleme dont tous les parametres
    sont deja sans dimension n'a rien a adimensionner.
    """
    if reduction < 0:
        msg = f"reduction negative : {reduction}"
        raise ValueError(msg)
    if parametres < reduction:
        msg = (
            f"reduction superieure au nombre de parametres : {reduction} "
            f"pour {parametres} -- le nombre de groupes serait negatif"
        )
        raise ValueError(msg)
    return parametres - reduction


def reduction_apres_echec(reduction: int) -> int:
    """`j - 1` : la regle de repli que le document donne en clair.

    « Go back and reduce j by one and try again ». Elle est ce qui rend
    la methode utilisable malgre un `j` devine : un echec d'algebre
    relance l'analyse au lieu de la bloquer.

    A zero elle refuse, parce que le document nomme l'autre cause avant
    celle-ci : « verify that you have included enough parameters in step
    1 ». L'echec vient alors de la liste, non de la reduction.
    """
    if reduction <= 0:
        msg = (
            f"reduction deja nulle : {reduction} -- l'echec vient de "
            f"l'etape 1, ou la liste des parametres est incomplete"
        )
        raise ValueError(msg)
    return reduction - 1
