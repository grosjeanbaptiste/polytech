"""Les pieges de la fiche d'electromagnetisme, chiffres par les corriges de `polytech/`.

Chaque piege que la fiche annonce est recalcule par l'application du
corrige qu'elle cite, pas redit a la main.
"""

from __future__ import annotations

from polytech.electromagnetisme.application import (
    cote_2017_puissance,
    cote_2019_fils,
    cote_2019_noyau,
    cote_2022_coquille,
    cote_2026_couplage,
    cote_2026_thevenin,
)
from polytech.maths.domain.comparaison import (
    dephasage_en_degres,
    egales,
    eloignees,
    proches,
    proches_relativement,
)


def test_oublier_la_bobine_change_e_t_et_r_t() -> None:
    """Thevenin 2025-2026 : sans la bobine, `E_T` et `R_T` changent tous deux."""
    avec = cote_2026_thevenin.thevenin()
    sans = cote_2026_thevenin.thevenin(avec_la_bobine=False)
    assert not egales(avec[0], sans[0])
    assert not egales(avec[1], sans[1])


def test_dix_ampere_par_metre_deplacent_le_nombre_de_spires_de_plusieurs() -> None:
    """Noyau 2018-2019 : environ 4535 spires ; `+-10 A/m` en deplace plusieurs."""
    base = cote_2019_noyau.spires()
    assert proches(base, 4535.0, 1.0)
    haut = cote_2019_noyau.spires(champ=cote_2019_noyau.CHAMP_LU_DANS_LE_FER + 10)
    assert eloignees(haut, base, 3.0)


def test_arrondir_la_hauteur_deplace_le_champ_bien_plus_que_la_hauteur() -> None:
    """Images 2018-2019 : `h_2` arrondi a 4,5 ; le champ bouge dix fois plus."""
    h2 = cote_2019_fils.hauteur()
    exact = cote_2019_fils.champ_en_p(h2)[1]
    arrondi = cote_2019_fils.champ_en_p(4.5)[1]
    sur_h = abs(4.5 - h2) / h2
    sur_champ = abs(arrondi - exact) / exact
    assert sur_champ > 5 * sur_h
    assert proches(sur_champ, 0.003, 0.001)  # 0,3 % : et non plusieurs pour cent


def test_d_est_continu_a_la_surface_sans_charge_et_e_saute_avec_epsilon() -> None:
    """Coquille 2021-2022 : `D` continu a `R_2` ; `E` saute de `eps/eps_0`."""
    _, dedans, dehors = cote_2022_coquille.champs()
    r2, eps, eps0 = (
        cote_2022_coquille.R2,
        cote_2022_coquille.EPS,
        cote_2022_coquille.EPS0,
    )
    r = cote_2022_coquille.RAYON
    assert egales((eps * dedans).subs(r, r2), (eps0 * dehors).subs(r, r2))
    assert egales(dedans.subs(r, r2) / dehors.subs(r, r2), eps0 / eps)


def test_des_points_inverses_font_passer_l_avance_en_retard_d_un_quart() -> None:
    """Couplage 2025-2026 : `I_2/I_1 = +-j w L_12 / R_2` selon les points."""
    droit = cote_2026_couplage.rapport_a_la_resonance()
    inverse = cote_2026_couplage.rapport_a_la_resonance(
        signe=-cote_2026_couplage.signe()
    )
    unite = dict.fromkeys(droit.free_symbols, 1)
    assert proches(dephasage_en_degres(complex(droit.subs(unite)), 1.0), 90.0, 1e-9)
    assert proches(dephasage_en_degres(complex(inverse.subs(unite)), 1.0), -90.0, 1e-9)


def test_une_puissance_active_negative_dit_que_la_branche_fournit() -> None:
    """Puissance 2016-2017 : `P < 0` en convention recepteur ; le bilan la retrouve."""
    puissance = cote_2017_puissance.puissance_de_la_branche()
    bilan = cote_2017_puissance.bilan_par_element()
    assert puissance.real < 0
    assert proches_relativement(
        bilan.source_active + bilan.resistance, puissance.real, 1e-9
    )
