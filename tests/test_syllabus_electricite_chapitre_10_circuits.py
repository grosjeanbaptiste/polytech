"""Syllabus d'electricite generale, chapitre 10 : reluctances et bobines sur noyau.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 10-19 a 10-29, lues sur les pages rendues.

Le chapitre ne chiffre rien. Les oracles sont des recoupements : un tube
conique dont la reluctance s'integre a la main, la reluctance uniforme
(10.11.12) que la bibliotheque portait deja, et le facteur de couplage
calcule par ses deux ecritures, (10.15.5) sur les inductances et (10.15.6)
sur les reluctances.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.magnetostatique.bobines_sur_noyau import (
    BobinesSurNoyau,
    facteur_de_dispersion,
)
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    flux_par_hopkinson,
    reluctance,
)
from polytech.electromagnetisme.domain.magnetostatique.couplage import (
    CouplageMagnetique,
    inductance_propre_d_un_solenoide,
)
from polytech.electromagnetisme.domain.magnetostatique.reluctances import (
    flux_utile_et_de_dispersion,
    force_magnetomotrice,
    permeance,
    reluctance_d_un_tube,
    reluctance_en_parallele,
    reluctance_en_serie,
)

S = sp.Symbol("s", nonnegative=True)


def test_la_reluctance_d_un_tube_uniforme_est_l_sur_mu_s() -> None:
    """(10.11.10) se reduit a (10.11.12) quand la section ne varie pas."""
    permeabilite = 1000 * PERMEABILITE_DU_VIDE
    tube = reluctance_d_un_tube(sp.Float(4e-4), S, 0.3, permeabilite)
    assert float(tube) == pytest.approx(reluctance(0.3, 4e-4, 1000))


def test_la_reluctance_d_un_tube_conique_est_l_sur_mu_pi_r1_r2() -> None:
    """Figure 10.11.1 : un tube qui s'evase. `integrale de ds / (mu pi r(s)^2)`.

    Avec `r(s) = r_1 + (r_2 - r_1) s / l`, la primitive de `1 / r^2` donne
    `l / (r_1 r_2)` : la moyenne geometrique des deux sections.
    """
    mu, longueur, r1, r2 = sp.symbols("mu l r_1 r_2", positive=True)
    rayon = r1 + (r2 - r1) * S / longueur
    tube = reluctance_d_un_tube(sp.pi * rayon**2, S, longueur, mu)
    assert sp.simplify(tube - longueur / (mu * sp.pi * r1 * r2)) == 0


def test_une_permeabilite_nulle_n_a_pas_de_reluctance() -> None:
    with pytest.raises(ValueError, match="permeabilite nulle"):
        reluctance_d_un_tube(sp.Float(1e-4), S, 0.1, 0)


def test_les_reluctances_en_serie_s_ajoutent() -> None:
    """(10.12.1) : un meme flux les traverse."""
    assert reluctance_en_serie(2e5, 3e5, 5e5) == pytest.approx(1e6)


def test_les_reluctances_en_parallele_ajoutent_leurs_inverses() -> None:
    """(10.12.2) : une meme difference de potentiel magnetique."""
    assert reluctance_en_parallele(2e5, 3e5, 6e5) == pytest.approx(1e5)


def test_une_association_vide_ou_une_reluctance_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="aucune reluctance"):
        reluctance_en_serie()
    with pytest.raises(ValueError, match="aucune reluctance"):
        reluctance_en_parallele()
    with pytest.raises(ValueError, match="reluctance nulle ou negative"):
        reluctance_en_parallele(2e5, 0.0)
    with pytest.raises(ValueError, match="reluctance nulle ou negative"):
        reluctance_en_serie(2e5, -1.0)


def test_la_permeance_est_l_inverse_de_la_reluctance() -> None:
    """Page 10-20 : « son inverse se note Lambda »."""
    assert permeance(4e5) == pytest.approx(2.5e-6)
    assert permeance(reluctance_en_parallele(2e5, 3e5)) == pytest.approx(
        permeance(2e5) + permeance(3e5)
    )


def test_la_force_magnetomotrice_est_la_somme_algebrique_des_amperes_tours() -> None:
    """(10.11.15), puis les deux cas de la figure 10.11.3."""
    assert force_magnetomotrice([(200, 1.5)]) == pytest.approx(300.0)
    # (10.11.17) : la seconde bobine s'oppose a la premiere.
    assert force_magnetomotrice([(200, 1.5), (50, -2.0)]) == pytest.approx(200.0)
    assert flux_par_hopkinson(force_magnetomotrice([(200, 1.5)]), 4e5) == (
        pytest.approx(7.5e-4)
    )


def test_le_flux_de_dispersion_est_petit_devant_le_flux_utile() -> None:
    """(10.14.1) `N I = R_u Phi_u = R_d Phi_d`, avec `R_d` bien plus grande."""
    utile, dispersion = flux_utile_et_de_dispersion(300.0, 4e5, 2e7)
    assert utile == pytest.approx(7.5e-4)
    assert dispersion == pytest.approx(1.5e-5)
    assert utile + dispersion == pytest.approx(
        flux_par_hopkinson(300.0, reluctance_en_parallele(4e5, 2e7))
    )


BOBINES = BobinesSurNoyau(spires=(400, 100), commune=5e5, dispersions=(2e7, 5e6))
"""`R_12` du tube commun, puis `R'_1` et `R'_2` des tubes de dispersion."""


def test_la_reluctance_vue_d_une_bobine_est_un_parallele() -> None:
    """Page 10-27 : `R_1` resulte de `R_12` et `R'_1` en parallele, donc `<= R_12`."""
    assert BOBINES.reluctance_vue(0) == pytest.approx(reluctance_en_parallele(5e5, 2e7))
    assert BOBINES.reluctance_vue(0) <= BOBINES.commune


def test_les_inductances_sont_des_carres_de_spires_sur_des_reluctances() -> None:
    """(10.15.2), (10.15.3) et (10.15.4)."""
    assert BOBINES.propre(0) == pytest.approx(400**2 / BOBINES.reluctance_vue(0))
    assert BOBINES.propre(1) == pytest.approx(100**2 / BOBINES.reluctance_vue(1))
    assert BOBINES.mutuelle() == pytest.approx(400 * 100 / 5e5)


def test_les_deux_ecritures_du_facteur_de_couplage_s_accordent() -> None:
    """(10.15.6) sur les reluctances, (10.15.5) sur les inductances."""
    par_les_inductances = CouplageMagnetique(
        BOBINES.propre(0), BOBINES.propre(1), BOBINES.mutuelle()
    ).coefficient()
    assert BOBINES.facteur_de_couplage() == pytest.approx(par_les_inductances)
    assert 0 <= BOBINES.facteur_de_couplage() <= 1


def test_le_facteur_de_dispersion_est_un_moins_k_carre() -> None:
    """(10.15.8) : nul si le couplage est parfait, un sans flux mutuel."""
    assert facteur_de_dispersion(1.0) == 0
    assert facteur_de_dispersion(0.0) == 1
    assert facteur_de_dispersion(0.6) == pytest.approx(0.64)


def test_un_facteur_de_couplage_hors_de_zero_un_est_refuse() -> None:
    with pytest.raises(ValueError, match="facteur de couplage hors de"):
        facteur_de_dispersion(1.2)


def test_l_inductance_de_dispersion_est_celle_du_tube_qui_evite_l_autre() -> None:
    """(10.15.11) `L_1 = (N_1 / N_2) L_12 + l_1`, et `l_1 = N_1 Phi'_1 / I_1`.

    Le second chemin ne passe pas par les inductances : le flux de
    dispersion d'un courant `I_1` vaut `N_1 I_1 / R'_1`.
    """
    assert BOBINES.de_dispersion(0) == pytest.approx(400**2 / 2e7)
    assert BOBINES.de_dispersion(1) == pytest.approx(100**2 / 5e6)


def test_des_bobines_parfaitement_couplees_ont_le_rapport_de_leurs_spires() -> None:
    """(10.15.9) et (10.15.10), sans dispersion : `R'` infinie."""
    parfaites = BobinesSurNoyau(
        spires=(400, 100), commune=5e5, dispersions=(math.inf, math.inf)
    )
    assert parfaites.facteur_de_couplage() == pytest.approx(1.0)
    assert parfaites.propre(0) / parfaites.mutuelle() == pytest.approx(4.0)
    assert parfaites.propre(0) / parfaites.propre(1) == pytest.approx(16.0)
    assert parfaites.de_dispersion(0) == pytest.approx(0.0)


def test_la_mutuelle_est_bornee_par_les_deux_moyennes() -> None:
    """(10.15.13) la moyenne geometrique, (10.15.14) l'arithmetique."""
    propre_1, propre_2, mutuelle = (
        BOBINES.propre(0),
        BOBINES.propre(1),
        BOBINES.mutuelle(),
    )
    assert mutuelle**2 <= propre_1 * propre_2
    assert mutuelle <= (propre_1 + propre_2) / 2


def test_des_spires_ou_des_reluctances_non_positives_sont_refusees() -> None:
    with pytest.raises(ValueError, match="nombre de spires nul ou negatif"):
        BobinesSurNoyau(spires=(0, 100), commune=5e5, dispersions=(2e7, 5e6))
    with pytest.raises(ValueError, match="reluctance nulle ou negative"):
        BobinesSurNoyau(spires=(400, 100), commune=0.0, dispersions=(2e7, 5e6))
    with pytest.raises(ValueError, match="reluctance nulle ou negative"):
        BobinesSurNoyau(spires=(400, 100), commune=5e5, dispersions=(2e7, -1.0))


def test_l_inductance_d_un_tore_mince_est_n2_mu_s_sur_l() -> None:
    """(10.15.12), que la bibliotheque ecrivait deja pour un solenoide."""
    seule = BobinesSurNoyau(
        spires=(300, 300),
        commune=reluctance(0.4, 2e-4, 800),
        dispersions=(math.inf, math.inf),
    )
    assert seule.propre(0) == pytest.approx(
        inductance_propre_d_un_solenoide(300, 2e-4, 0.4, 800)
    )


def test_un_tube_de_dispersion_sans_reluctance_est_refuse() -> None:
    with pytest.raises(ValueError, match="reluctance nulle ou negative"):
        flux_utile_et_de_dispersion(300.0, 4e5, 0.0)


def test_un_facteur_de_couplage_negatif_est_refuse() -> None:
    with pytest.raises(ValueError, match="facteur de couplage hors de"):
        facteur_de_dispersion(-0.5)
