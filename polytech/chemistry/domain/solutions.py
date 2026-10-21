"""La composition d'une solution : molarite et dilution.

Source : syllabus « Introduction aux sciences de l'ingenieur » 2025-2026,
paragraphes K.3.1 et K.3.2, relation (K.2), pages 97 et 98.
"""

from __future__ import annotations


def molarite(quantite: float, volume: float) -> float:
    """C = n / V, en mol/l pour n en moles et V en litres ; leve pour un volume nul."""
    if volume <= 0:
        msg = f"une solution a un volume positif, pas {volume}"
        raise ValueError(msg)
    return quantite / volume


def volume_de_solution_mere(c_mere: float, c_diluee: float, v_dilue: float) -> float:
    """(K.2) C_dil V_dil = C_conc V_conc, resolue pour V_conc.

    Les moles de solute sont les memes avant et apres : la dilution ajoute du
    solvant. Leve si la solution visee est plus concentree que la mere.
    """
    if c_diluee > c_mere:
        msg = f"une dilution ne concentre pas : {c_diluee} > {c_mere} mol/l"
        raise ValueError(msg)
    return c_diluee * v_dilue / c_mere
