"""Le registre des courbes et des valeurs que les figures de fiche demandent.

Une figure ecrit `@courbe:nom@` (des points `(x,y)`) ou `@valeur:nom@` (un
nombre) : le gabarit les remplace par ce que la bibliotheque calcule.
"""

from __future__ import annotations

from collections.abc import Callable

from polytech.electromagnetisme.application import courbes_fiche as em
from polytech.thermique.application import (
    courbes_fiche_conduction as cond,
)
from polytech.thermique.application import (
    courbes_fiche_convection as conv,
)
from polytech.thermique.application import (
    courbes_fiche_rayonnement as ray,
)

type Points = list[tuple[float, float]]

COURBES: dict[str, Callable[[], Points]] = {
    "continuite_bilan": cond.bilan_du_toit,
    "continuite_cordes": cond.cordes,
    "tube_flux": cond.flux_du_tube_isole,
    "barre_refroidie": cond.barre_refroidie,
    "solide_decroissance": cond.decroissance_du_solide,
    "solide_tangente": cond.tangente_du_solide,
    "planck_soleil": ray.planck_du_soleil,
    "planck_corps": ray.planck_du_corps,
    "verre_transmission": ray.transmission_du_verre,
    "facteur_deux_plans": ray.facteur_de_deux_plans,
    "coefficient_rayonnement": ray.coefficient_de_rayonnement,
    "plaque_locale": conv.coefficient_local_de_la_plaque,
    "lame_resistance": conv.resistance_de_la_lame,
    "echangeur_chaud": conv.profil_chaud,
    "echangeur_froid": conv.profil_froid,
    "efficacite_isotherme": conv.efficacite_a_paroi_isotherme,
    "efficacite_contre_courant": conv.efficacite_a_contre_courant,
    "efficacite_courants_paralleles": conv.efficacite_a_courants_paralleles,
    "rc_charge": em.charge,
    "rc_decharge": em.decharge,
    "rc_tangente": em.tangente_a_l_origine,
}

VALEURS: dict[str, Callable[[], float]] = {
    "continuite_racine_de_la_corde": cond.racine_de_la_corde,
    "tube_rayon_critique": cond.rayon_critique_reduit,
    "tube_flux_critique": cond.flux_au_rayon_critique,
    "temperature_du_soleil": ray.temperature_du_soleil,
    "temperature_du_corps": ray.temperature_du_corps,
    "decollement": conv.angle_de_separation,
    "rc_charge_a_tau": em.charge_a_tau,
    "echangeur_dt1_froid": lambda: conv.ecart_en(0.02)[0],
    "echangeur_dt1_chaud": lambda: conv.ecart_en(0.02)[1],
    "echangeur_dt2_froid": lambda: conv.ecart_en(0.98)[0],
    "echangeur_dt2_chaud": lambda: conv.ecart_en(0.98)[1],
}
