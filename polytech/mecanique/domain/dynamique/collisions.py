"""La collision elastique a une dimension.

Source : syllabus « Notions fondamentales » de « Physique generale »,
paragraphe 6.6.1, equations (6.48) et (6.49), pages 74 et 75.
"""

from __future__ import annotations


def collision_elastique(
    m1: float, v1: float, m2: float, v2: float
) -> tuple[float, float]:
    """Les vitesses apres un choc frontal elastique ; leve pour une masse nulle."""
    if m1 <= 0 or m2 <= 0:
        msg = f"une collision demande deux masses positives, pas {m1} et {m2}"
        raise ValueError(msg)
    somme = m1 + m2
    v1f = (m1 - m2) / somme * v1 + 2 * m2 / somme * v2
    v2f = 2 * m1 / somme * v1 + (m2 - m1) / somme * v2
    return v1f, v2f
