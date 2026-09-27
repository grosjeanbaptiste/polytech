"""Comparer une valeur calculee a une valeur de reference.

Le corrige recoupe sa pression calculee avec celle que le cours publie.
Dire « l'ecart vaut 0,27 % » n'a de sens qu'une fois nomme le denominateur :
la difference est rapportee a la **reference**, c'est-a-dire a la valeur
publiee, et non a celle qu'on vient de calculer. C'est la convention usuelle
d'un recoupement -- on mesure de combien on s'ecarte de ce qui fait foi --
et elle est fixee ici plutot que laissee implicite dans une mise en forme.
"""

from __future__ import annotations


def ecart_relatif(mesure: float, reference: float) -> float:
    """Rend l'ecart de ``mesure`` a ``reference``, rapporte a la reference.

    Le resultat est une fraction, sans signe : seul compte de combien les
    deux valeurs different, pas laquelle depasse l'autre.
    """
    if reference == 0:
        msg = "reference nulle : l'ecart relatif n'est pas defini"
        raise ValueError(msg)
    return abs(mesure - reference) / abs(reference)
