"""Laboratoire 3 d'electronique : les donnees des enonces.

Fichier de donnees, exempt de la limite de taille : il ne porte que les
tableaux de composants des exercices, tels que l'enonce les imprime, une
classe par exercice pour que deux « AMP1 » ne se confondent pas.
"""

from __future__ import annotations

from typing import Final


class UnUn:
    """Exercice 1.1 : la chaine type, cinq blocs imposes."""

    PLEINE_ECHELLE: Final = 5.0
    SIGNAL: Final = 1e-3
    DECALAGE: Final = 1.0
    RESOLUTION: Final = 2e-6
    BANDE_DU_SIGNAL: Final = 10e3
    BANDE_DU_FILTRE: Final = 20e3
    """La coupure du passe-bas impose : c'est elle qui borne le bruit et Nyquist."""
    PRODUITS_GAIN_BANDE: Final = {"AMP1": 0.5e6, "AMP2": 5e6, "AMP3": 50e6}
    CONVERTISSEURS: Final = {
        "CAN1": (10e3, 8),
        "CAN2": (10e3, 10),
        "CAN3": (10e3, 12),
        "CAN4": (100e3, 8),
        "CAN5": (100e3, 10),
        "CAN6": (100e3, 12),
    }
    """Cadence maximale, en hertz, et nombre de bits."""


class UnDeux:
    """Exercice 1.2 : huit microphones (examen 2022, question 3)."""

    VOIES: Final = 8
    BANDE: Final = 20e3
    SIGNAL: Final = 1e-3
    PLEINE_ECHELLE: Final = 5.0
    SOURCE: Final = 32.0
    BUDGET: Final = (2e-6) ** 2
    """Le bruit permis a l'entree, en volts carres."""
    PAS: Final = 1e-6
    AMPLIFICATEURS: Final = {
        "AMP1": (2e-6, 50e-15, 1e6, 1.0),
        "AMP2": (5e-6, 100e-15, 2e6, 1.0),
        "AMP3": (0.5e-6, 10e-15, 25e6, 4.0),
    }
    """Bruit efficace sur la bande, densite de courant, produit gain-bande, prix."""
    FILTRE: Final = 10e-6
    MULTIPLEXEURS: Final = {"M1": (10e-6, 2.0), "M2": (20e-6, 0.5)}
    """Densite de bruit, en volts par racine de hertz, et prix."""
    CONVERTISSEURS: Final = {
        "ADC1": (10, 1e6, 4),
        "ADC2": (12, 100e3, 2),
        "ADC3": (12, 400e3, 1),
    }
    """Nombre de bits, cadence maximale et stock."""


class UnTrois:
    """Exercice 1.3 : quarante-cinq signaux (examen 2022, question 1)."""

    VOIES: Final = 45
    BANDE: Final = 500.0
    SIGNAL: Final = 5e-3
    DECALAGE: Final = 250e-3
    PLEINE_ECHELLE: Final = 5.0
    SOURCE: Final = 1e3
    BUDGET: Final = (5e-6) ** 2
    PAS: Final = 2e-6
    AMPLIFICATEURS: Final = {
        "AMP1": (375e-9, 5e-15, (3.3, 15.0), 0.5e6),
        "AMP2": (12.5e-6, 2e-12, (3.3, 5.5), 50e6),
        "AMP3": (200e-9, 10e-15, (2.5, 3.7), 5e6),
        "AMP4": (375e-9, 10e-15, (3.3, 6.6), 10e6),
    }
    """Bruit efficace, densite de courant, plage d'alimentation, produit gain-bande."""
    PRIORITE: Final = ("AMP1", "AMP3", "AMP2", "AMP4")
    PASSE_BAS: Final = 10e-6
    PASSE_HAUT: Final = 1e-6
    MULTIPLEXEUR: Final = 10e-6
    CANAUX: Final = (16, 32)
    CONVERTISSEURS: Final = {
        "ADC1": (10, 100e3),
        "ADC2": (12, 60e3),
        "ADC3": (14, 50e3),
    }
    """Nombre de bits et cadence maximale."""


class UnQuatre:
    """Exercice 1.4 : deux cents capteurs de force sous un tapis."""

    CAPTEURS: Final = 200
    BANDE: Final = 100.0
    SIGNAL: Final = 10e-3
    PLEINE_ECHELLE: Final = 5.0
    SOURCE: Final = 10e3
    CAPTEUR: Final = 100e-9
    BUDGET: Final = 60e-6
    """Le bruit permis a l'entree, en volts efficaces."""
    PAS: Final = 0.2e-3
    AMPLIFICATEURS: Final = {
        "AMP1": (10e-9, 300e-12, 50e6),
        "AMP2": (5e-9, 3e-9, 1e6),
        "AMP3": (5e-9, 300e-9, 200e6),
    }
    """Densite de tension, densite de courant, produit gain-bande."""
    FILTRE: Final = (2e-6, 200e-9)
    CONVERTISSEUR: Final = 20e-6
    CONVERTISSEURS: Final = {
        "CAN1": (10e3, 8),
        "CAN2": (10e3, 10),
        "CAN3": (10e3, 12),
        "CAN4": (30e3, 8),
        "CAN5": (30e3, 10),
        "CAN6": (30e3, 12),
    }
    MULTIPLEXEUR: Final = 10e-6
    MULTIPLEXEURS: Final = {"MUX1": 32, "MUX2": 64, "MUX3": 128}
