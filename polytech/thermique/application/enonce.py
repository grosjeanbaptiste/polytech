"""L'enonce du TP 1 de conductibilite, transcrit une seule fois.

Chaque donnee est ici et nulle part ailleurs : le document la cite par une
macro. Corriger un chiffre a cet endroit corrige donc tout le corrige, et
aucune valeur ne peut deriver entre deux pages.

Le facteur d'absorption et le flux solaire sont gardes **separes**, alors
que seul leur produit entre dans les equations. L'enonce les donne
separes, et les multiplier ici ferait disparaitre une donnee que le
lecteur doit pouvoir retrouver.
"""

from __future__ import annotations

from typing import Final

from polytech.thermique.domain.conduction.plan import CoucheAvecGeneration
from polytech.thermique.domain.conduction.spherique import CoqueSpherique
from polytech.thermique.domain.echanges import CoefficientDEchange, DensiteDeFlux
from polytech.thermique.domain.geometrie import Rayon
from polytech.thermique.domain.grandeurs import Temperature
from polytech.thermique.domain.materiaux import (
    ChaleurLatente,
    Conductivite,
    Epaisseur,
    GenerationVolumique,
)
from polytech.thermique.domain.reservoir import ReservoirCryogenique

# --- Exercice 1 : oxydation dans un stockage de charbon ---------------------
EPAISSEUR_CHARBON: Final = Epaisseur(metres=2.0)
GENERATION_CHARBON: Final = GenerationVolumique(watts_par_m3=40.0)
CONDUCTIVITE_CHARBON: Final = Conductivite(watts_par_m_k=0.2)
FLUX_SOLAIRE: Final = DensiteDeFlux(watts_par_m2=150.0)
ABSORPTION_CHARBON: Final = 0.95
TEMPERATURE_AIR: Final = Temperature.en_celsius(20.0)
ECHANGE_EXTERIEUR: Final = CoefficientDEchange(watts_par_m2_k=25.0)
TEMPERATURE_SOL: Final = Temperature.en_celsius(10.0)
ECHANGE_SOL: Final = CoefficientDEchange(watts_par_m2_k=1.0)

FLUX_ABSORBE: Final = DensiteDeFlux(
    watts_par_m2=ABSORPTION_CHARBON * FLUX_SOLAIRE.watts_par_m2,
)


def tas_de_charbon(echange_bas: CoefficientDEchange) -> CoucheAvecGeneration:
    """Le tas de l'enonce, le seul parametre de ses deux cas etant K_bas.

    Les champs sont ecrits en clair, et non deplies d'un dictionnaire
    commun : mypy ne verifie pas un `**dict`, et c'est precisement ici
    qu'une donnee rangee au mauvais champ passerait inapercue.
    """
    return CoucheAvecGeneration(
        conductivite=CONDUCTIVITE_CHARBON,
        generation=GENERATION_CHARBON,
        epaisseur=EPAISSEUR_CHARBON,
        echange_bas=echange_bas,
        temperature_bas=TEMPERATURE_SOL,
        echange_haut=ECHANGE_EXTERIEUR,
        temperature_haut=TEMPERATURE_AIR,
        flux_absorbe=FLUX_ABSORBE,
    )


SOL_ADIABATIQUE: Final = tas_de_charbon(CoefficientDEchange(watts_par_m2_k=0.0))
"""Cas (a) de l'enonce : la face basse ne laisse rien passer."""

SOL_PERMEABLE: Final = tas_de_charbon(ECHANGE_SOL)
"""Cas (b) : la meme couche, avec perte vers le sol."""

# --- Exercice 2 : stockage d'hydrogene liquide ------------------------------
DIAMETRE_INTERIEUR: Final = 0.6
EPAISSEUR_SILICE: Final = Epaisseur(metres=0.025)
CONDUCTIVITE_SILICE: Final = Conductivite(watts_par_m_k=0.0017)
TEMPERATURE_HYDROGENE: Final = Temperature.en_kelvins(20.0)
TEMPERATURE_AMBIANTE: Final = Temperature.en_kelvins(300.0)
ECHANGE_RESERVOIR: Final = CoefficientDEchange(watts_par_m2_k=8.0)
CHALEUR_LATENTE: Final = ChaleurLatente(joules_par_kg=4.5e5)

RAYON_INTERIEUR: Final = Rayon.depuis_le_diametre(DIAMETRE_INTERIEUR)
COQUE: Final = CoqueSpherique(
    conductivite=CONDUCTIVITE_SILICE,
    interieur=RAYON_INTERIEUR,
    exterieur=RAYON_INTERIEUR.augmente_de(EPAISSEUR_SILICE),
)
RESERVOIR: Final = ReservoirCryogenique(
    coque=COQUE,
    echange_exterieur=ECHANGE_RESERVOIR,
    temperature_interieure=TEMPERATURE_HYDROGENE,
    temperature_exterieure=TEMPERATURE_AMBIANTE,
    chaleur_latente=CHALEUR_LATENTE,
)
