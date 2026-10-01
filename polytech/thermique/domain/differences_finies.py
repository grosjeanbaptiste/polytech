"""Les differences finies appliquees a la conductibilite thermique.

Chapitre 5 du tome 1 du syllabus, absent de la bibliotheque jusqu'ici
parce que la mesure de couverture ne regardait que les planches.

Le syllabus pose la methode en clair : on decoupe le domaine en elements
parallelepipediques dont les sommets sont les noeuds, le champ continu
devient un jeu de temperatures nodales, et les derivees partielles sont
approchees par differences. Il retient les **differences centrees**, dont
l'erreur de troncature est en `Delta x^2`, parce qu'elles sont « les plus
utilisees » -- reduire l'erreur en prenant plus de voisins elargirait la
bande de la matrice, et le gain serait « partiellement compense ».

Le partage stationnaire / transitoire est celui du syllabus :

* en stationnaire, sur un reseau quadrille et sans source, la temperature
  d'un noeud est la **moyenne de ses quatre voisins** -- ce module ne le
  pose pas, il le deduit du laplacien a cinq points ;
* en transitoire, l'explicite d'Euler « pose de gros problemes de
  stabilite, celle-ci n'etant assuree que pour des pas de temps tres
  petits », l'implicite est « inconditionnellement stable », et
  Crank-Nicholson est entre les deux.

La **borne** de stabilite, que le syllabus annonce par « on peut montrer
qu'il faut » suivi d'une formule en image que la conversion n'a pas
rendue, n'est pas affirmee sur cette foi : `tests/test_stabilite_explicite`
la verifie en faisant diverger le schema au-dela et converger en dessous,
et `stabilite_numerique` porte les deux bornes -- l'asymptotique du cours
et l'exacte du maillage fini.
"""

from __future__ import annotations

from polytech.thermique.domain.stabilite_numerique import fourier_critique


def _exiger_le_pas(pas: float) -> None:
    """Le pas d'espace apparait au carre : nul, il divise par zero."""
    if pas <= 0:
        msg = f"pas d'espace nul ou negatif : {pas} m"
        raise ValueError(msg)


def derivee_seconde_centree(
    gauche: float,
    centre: float,
    droite: float,
    pas: float,
) -> float:
    """`(t_- - 2 t_0 + t_+) / Delta x^2`, d'erreur en `Delta x^2`.

    Exacte sur une parabole -- l'erreur porte sur le terme d'ordre trois
    du developpement de Taylor, que la parabole n'a pas.
    """
    _exiger_le_pas(pas)
    return (gauche - 2.0 * centre + droite) / pas**2


def laplacien_a_cinq_points(
    centre: float,
    voisins: tuple[float, float, float, float],
    pas: float,
) -> float:
    """Le laplacien sur un reseau quadrille : `(somme des voisins - 4 t_0)/h^2`.

    Les quatre voisins sont pris dans l'ordre gauche, droite, bas, haut ;
    la somme ne distingue pas les directions, le reseau etant quadrille
    (`Delta x = Delta y`), ce qui est l'hypothese du syllabus.
    """
    _exiger_le_pas(pas)
    return (sum(voisins) - 4.0 * centre) / pas**2


def temperature_nodale_stationnaire(
    voisins: tuple[float, float, float, float],
) -> float:
    """La moyenne des quatre voisins : la solution de `laplacien = 0`.

    Sans source interne et sur un reseau quadrille, annuler le laplacien
    a cinq points donne `4 t_0 = somme des voisins`. Le resultat du cours
    se deduit donc, il ne se postule pas -- et un test verifie que la
    valeur rendue annule bien le laplacien.
    """
    return sum(voisins) / 4.0


def nombre_de_fourier_de_maille(
    diffusivite: float,
    pas_de_temps: float,
    pas_d_espace: float,
) -> float:
    """`Fo = a Delta tau / Delta x^2`, sans dimension.

    C'est lui, et non le pas de temps seul, qui porte la stabilite : a
    `Fo` constant, doubler le maillage autorise a quadrupler le pas de
    temps.
    """
    _exiger_le_pas(pas_d_espace)
    if diffusivite <= 0:
        msg = f"diffusivite nulle ou negative : {diffusivite} m^2/s"
        raise ValueError(msg)
    if pas_de_temps <= 0:
        msg = f"pas de temps nul ou negatif : {pas_de_temps} s"
        raise ValueError(msg)
    return diffusivite * pas_de_temps / pas_d_espace**2


def pas_de_temps_critique(
    diffusivite: float,
    pas_d_espace: float,
    dimensions: int,
) -> float:
    """Le plus grand pas de temps qu'Euler explicite supporte, en secondes.

    Il decroit comme le **carre** du maillage : c'est le cout que le
    syllabus reproche a l'explicite, et la raison de preferer
    l'implicite des que le maillage se raffine.
    """
    _exiger_le_pas(pas_d_espace)
    if diffusivite <= 0:
        msg = f"diffusivite nulle ou negative : {diffusivite} m^2/s"
        raise ValueError(msg)
    return fourier_critique(dimensions) * pas_d_espace**2 / diffusivite
