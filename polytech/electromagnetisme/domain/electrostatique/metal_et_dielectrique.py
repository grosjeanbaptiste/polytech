"""Un dielectrique parfait sous un metal charge : l'exemple de synthese.

Planches du cours d'electronique (chapitre 1, partie I, planches 30 a
36). Un demi-espace de metal, de conductivite infinie, porte sur sa face
inferieure une charge de surface `sigma_s` ; sous lui, un mince vide,
puis un demi-espace de dielectrique parfait, de permittivite `eps_1`.

Dans le metal, le champ est nul. Le theoreme de Gauss sur un cylindre a
cheval sur la face chargee donne le champ dans le vide, `E_0 = sigma_s /
eps_0` ; sa version macroscopique, sur un cylindre a cheval sur
l'interface, ou aucune charge **libre** ne se trouve, donne `eps_1 E_1 =
eps_0 E_0`. Le dielectrique abaisse donc le champ, et il le fait par une
charge **liee** qui s'accumule a sa surface -- celle que la loi de Gauss
microscopique, qui compte toutes les charges, fait apparaitre.
"""

from __future__ import annotations

import sympy as sp


def champ_dans_le_vide(charge_surfacique: object, permittivite: object) -> sp.Expr:
    """`E_0 = sigma_s / eps_0`, entre le metal et le dielectrique."""
    return sp.sympify(charge_surfacique) / sp.sympify(permittivite)


def champ_dans_le_dielectrique(
    charge_surfacique: object, permittivite: object
) -> sp.Expr:
    """`E_1 = sigma_s / eps_1` : `D` se conserve, faute de charge libre."""
    return sp.sympify(charge_surfacique) / sp.sympify(permittivite)


def charge_liee_a_l_interface(
    charge_surfacique: object, vide: object, dielectrique: object
) -> sp.Expr:
    """`sigma_b = eps_0 (E_1 - E_0) = -(eps_1 - eps_0) sigma_s / eps_1`."""
    dans_le_vide = champ_dans_le_vide(charge_surfacique, vide)
    dans_la_matiere = champ_dans_le_dielectrique(charge_surfacique, dielectrique)
    return sp.simplify(sp.sympify(vide) * (dans_la_matiere - dans_le_vide))
