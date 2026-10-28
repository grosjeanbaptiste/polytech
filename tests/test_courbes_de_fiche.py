"""Les courbes de fiche generees par la bibliotheque valent les formules d'avant.

Les anciennes figures tapaient leur formule en pgfplots ; chaque courbe est
maintenant calculee par `polytech/`, et recoupee ici par la formule d'origine.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

from polytech.maths.domain.comparaison import proches, proches_relativement
from polytech.thermique.application import courbes_de_fiche as registre
from polytech.thermique.application import courbes_fiche_conduction as cond
from polytech.thermique.application import courbes_fiche_convection as conv

if TYPE_CHECKING:
    from collections.abc import Callable


def _tous(nom: str, formule: Callable[[float], float], tolerance: float = 1e-9) -> bool:
    return all(
        proches_relativement(y, formule(x), tolerance)
        for x, y in registre.COURBES[nom]()
    )


def test_chaque_courbe_est_finie_et_a_plusieurs_points() -> None:
    """Aucune courbe vide ni non finie."""
    for nom, courbe in registre.COURBES.items():
        points = courbe()
        assert len(points) >= 2, nom
        assert all(math.isfinite(x) and math.isfinite(y) for x, y in points), nom


def test_les_courbes_a_formule_ferme_valent_l_ancienne_formule() -> None:
    """tanh(x)/x, exp(-x), 1-exp(-x), x/(1+x), (1-exp(-2x))/2, 1/(1/0.9+1/x-1)."""
    assert _tous("ailette_efficacite", lambda x: math.tanh(x) / x)
    assert _tous("solide_decroissance", lambda x: math.exp(-x))
    assert _tous("rc_charge", lambda x: 1 - math.exp(-x))
    assert _tous("rc_decharge", lambda x: math.exp(-x))
    assert _tous("efficacite_isotherme", lambda x: 1 - math.exp(-x))
    assert _tous("efficacite_contre_courant", lambda x: x / (1 + x))
    assert _tous("efficacite_courants_paralleles", lambda x: (1 - math.exp(-2 * x)) / 2)
    assert _tous("facteur_deux_plans", lambda x: 1 / (1 / 0.9 + 1 / x - 1))
    assert _tous("coefficient_rayonnement", lambda x: 4 * 5.67e-8 * x**3, 1e-3)


def test_les_courbes_de_conduction_valent_leur_formule() -> None:
    """`1/(ln(x)/2 + 1/x)`, `1 - x^2`, tangente `1 - x`, plaque `x^(-1/2)`."""
    assert _tous("tube_flux", lambda x: 1 / (math.log(x) / 2 + 1 / x))
    assert _tous("barre_refroidie", lambda x: 1 - x * x)
    assert _tous("solide_tangente", lambda x: 1 - x)
    reduit = registre.COURBES["plaque_locale"]()
    depart = reduit[0][0] ** -0.5
    assert all(proches_relativement(y, x**-0.5 / depart, 1e-9) for x, y in reduit)


def test_les_valeurs_nommees_sont_celles_de_la_fiche() -> None:
    """`r_c/r_i = 2`, 63 %, 2400, decollement, le 2 de `h_moyen`."""
    v = registre.VALEURS
    assert proches(v["tube_rayon_critique"](), 2.0, 1e-12)
    assert round(100 * v["rc_charge_a_tau"]()) == 63
    assert v["reynolds_critique_du_tube"]() == 2400
    assert 80 <= v["decollement"]() <= 110
    assert proches(conv.rapport_moyen_sur_local(), 2.0, 1e-12)


def test_la_corde_de_la_fausse_position_tombe_pres_de_la_racine() -> None:
    """La figure dit : la corde coupe l'axe pres de la racine du bilan."""
    assert proches(cond.racine_de_la_corde(), cond.racine_du_bilan(), 3.0)


def test_la_resistance_de_la_lame_monte_puis_s_aplatit() -> None:
    """La resistance de la lame monte puis s'aplatit : la convection s'installe."""
    points = conv.resistance_de_la_lame()
    pentes = [
        (y2 - y1) / (x2 - x1) for (x1, y1), (x2, y2) in itertools.pairwise(points)
    ]
    assert pentes[0] > 5 * pentes[-1] > 0
