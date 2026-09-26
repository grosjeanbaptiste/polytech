"""L'exposant du volume doit etre integre, pas retourne en dur.

Toute la reponse sur le barrage repose sur la chaine V -> H -> F. Le
deuxieme maillon etait derive (integrale de la pression), le premier etait
un `return 3` justifie par un commentaire. Ces tests exigent qu'il vienne
lui aussi d'une integrale, a partir de la seule section horizontale du
bassin.
"""

import sympy as sp

from polytech.fluides.domain.grandeurs import MasseVolumique
from polytech.fluides.domain.statique.parements import (
    ParementIncline,
    ParementTriangulaire,
    ParementVertical,
)
from polytech.fluides.domain.statique.symboles import AIRE, ALLONGEMENT, ALPHA, H

EAU = MasseVolumique(kg_par_m3=1_000.0)
MUR = ParementVertical(fluide=EAU, largeur=1.0)
VALLEE = ParementTriangulaire(fluide=EAU, pente=0.5)


def _egal(obtenu: sp.Expr, attendu: sp.Expr) -> bool:
    return bool(sp.simplify(obtenu - attendu) == 0)


def test_le_bassin_prismatique_a_une_section_constante() -> None:
    """Berges verticales : l'aire de la surface libre ne depend pas de H."""
    assert not MUR.section_horizontale().has(H)


def test_son_volume_s_integre_en_a_fois_h() -> None:
    assert _egal(MUR.expression_volume(), AIRE * H)


def test_la_vallee_auto_similaire_a_une_section_en_h_carre() -> None:
    """Largeur 2 alpha H et longueur lambda H : l'aire varie comme H^2."""
    assert _egal(VALLEE.section_horizontale(), 2 * ALPHA * ALLONGEMENT * H**2)


def test_son_volume_s_integre_en_h_cube() -> None:
    """C'est la justification de « V proportionnel a H^3 », enfin calculee."""
    attendu = 2 * ALPHA * ALLONGEMENT * H**3 / 3
    assert _egal(VALLEE.expression_volume(), attendu)


def test_les_exposants_se_lisent_sur_les_integrales() -> None:
    assert MUR.exposant_volume() == 1
    assert VALLEE.exposant_volume() == 3


def test_le_parement_incline_retient_un_bassin_prismatique() -> None:
    """L'inclinaison change la paroi, pas la facon dont le bassin se remplit."""
    incline = ParementIncline(fluide=EAU, largeur=1.0, angle=sp.pi / 3)
    assert incline.exposant_volume() == 1


def test_le_rapport_des_poussees_ne_depend_pas_de_la_hauteur_de_reference() -> None:
    """Le corrige evalue les rapports sur H = 10 m : un choix sans effet.

    F'/F ne depend que de H'/H : le verifier evite qu'une valeur arbitraire
    de la couche application ne se glisse dans un resultat.
    """
    moitie = sp.Rational(1, 2)
    assert MUR.rapport_poussee(moitie) == sp.Rational(1, 4)
    assert VALLEE.rapport_poussee(moitie) == moitie
