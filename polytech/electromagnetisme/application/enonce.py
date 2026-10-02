"""Les donnees de l'exercice 1, et elles ne viennent que de la transcription.

Un seul endroit les porte, pour qu'une donnee mal lue se corrige une fois
et se propage partout -- texte, calculs et recoupements.

La transcription est celle du scan manuscrit, sous
`transcriptions/Electromagnetisme/EXS/Exs_03_Stationnaire_Thevenin_Norton_01.md`,
et elle a ete relue contre le scan. Ces quatre nombres sont tout l'enonce.
"""

from __future__ import annotations

from typing import Final

TENSION_DE_LA_SOURCE: Final = 110.0
"""La source de tension, en volts."""

RESISTANCE_EN_SERIE: Final = 1.0
"""La resistance en serie avec elle, en ohms."""

COURANT_DE_LA_SOURCE: Final = 10.0
"""La source de courant en parallele, en amperes."""

CHARGE_DE_LA_QUESTION_D: Final = 5.0
"""La charge resistive de la question (d), en ohms."""

DECIMALES_AFFICHEES: Final = 2
"""L'enonce ne donne que des entiers : deux decimales suffisent au rendu."""
