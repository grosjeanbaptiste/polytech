"""Ce qu'un pole fait au signal : la forme de sa composante.

Syllabus de signaux et systemes, section 2.2.5 et figure 2.25. Mis a part
les impulsions qu'apporte un numerateur trop grand, les poles decident
seuls du type des composantes d'un signal. Chacun donne une
exponentielle imaginaire amortie, de pulsation egale a sa partie
imaginaire et d'amortissement egal a sa partie reelle :

* pole en zero, un terme constant ;
* poles imaginaires purs, une sinusoide non amortie ;
* poles complexes conjugues, une sinusoide amortie -- ou amplifiee, a
  droite de l'axe imaginaire ;
* pole reel, une exponentielle ;
* pole double, la meme forme multipliee par `t`.

Les zeros ne changent que les residus : l'amplitude et la phase.
"""

from __future__ import annotations

TOLERANCE = 1e-12
"""En dessous, une partie reelle ou imaginaire est tenue pour nulle."""


def _forme(pole: complex) -> str:
    oscille = abs(pole.imag) > TOLERANCE
    if abs(pole.real) <= TOLERANCE:
        return "sinusoide non amortie" if oscille else "terme constant"
    if oscille:
        return "sinusoide amortie" if pole.real < 0 else "sinusoide amplifiee"
    sens = "decroissante" if pole.real < 0 else "croissante"
    return f"exponentielle {sens}"


def nature_de_la_composante(pole: complex, *, double: bool = False) -> str:
    """La forme de la composante qu'apporte ce pole."""
    forme = _forme(complex(pole))
    if not double:
        return forme
    accord = "multiplie" if forme == "terme constant" else "multipliee"
    return f"{forme}, {accord} par t"
