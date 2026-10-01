"""Trois fascicules d'electromagnetisme stationnaire, repris en profondeur.

**III.1** : la force de Lorentz, l'element de courant et la force qu'il
subit, l'intensite comme flux de la densite, l'equation de continuite, et
les conditions de continuite de l'induction et du champ a une interface --
que le fascicule met en regard de celles de l'electrostatique, parce que
les composantes conservees ne sont **pas** les memes.

**III.4** : l'inductance propre, le flux a travers un circuit, et surtout
l'association d'inductances couplees -- dont le fascicule dit que « le
signe des inductances est lie au sens positif des flux » et qu'elle
« depend du sens du flux ». C'est le piege du chapitre.

**III.5** : le comportement de la matiere, les trois magnetismes --
diamagnetisme, paramagnetisme, ferromagnetisme --, la permeabilite qui
passe par un maximum, et la dissipation par hysteresis.

L'ancrage le plus net est celui des continuites, parce qu'il est
contre-intuitif : a une interface, c'est la composante **normale** de
l'induction qui se conserve et la composante **tangentielle** du champ --
l'inverse exact de l'electrostatique, ou la normale du deplacement et la
tangentielle du champ se conservent.
"""

from __future__ import annotations

import math

import pytest

from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    Magnetisme,
    classe_de_magnetisme,
    pertes_par_hysteresis,
)
from polytech.electromagnetisme.domain.magnetostatique.couplage import (
    CouplageMagnetique,
    inductance_propre_d_un_solenoide,
    signe_du_couplage,
)
from polytech.electromagnetisme.domain.magnetostatique.lois import (
    continuite_a_une_interface,
    force_de_lorentz,
    force_sur_un_element_de_courant,
    intensite_depuis_la_densite,
)


def test_la_force_de_lorentz_est_orthogonale_a_la_vitesse() -> None:
    """`q v x B` : elle ne travaille jamais, donc ne change pas l'energie.

    C'est la propriete qui fait qu'un champ magnetique courbe une
    trajectoire sans l'accelerer -- et le produit scalaire avec la vitesse
    doit valoir exactement zero.
    """
    force = force_de_lorentz(
        charge=1.6e-19,
        vitesse=(1e6, 0.0, 0.0),
        induction=(0.0, 0.0, 1.0),
    )
    assert sum(f * v for f, v in zip(force, (1e6, 0.0, 0.0), strict=True)) == (
        pytest.approx(0.0)
    )


def test_la_force_de_lorentz_est_nulle_quand_la_vitesse_suit_le_champ() -> None:
    """Un porteur qui longe les lignes de champ ne subit rien."""
    force = force_de_lorentz(
        charge=1.6e-19,
        vitesse=(0.0, 0.0, 1e6),
        induction=(0.0, 0.0, 1.0),
    )
    assert all(composante == pytest.approx(0.0) for composante in force)


def test_la_force_sur_un_element_de_courant_vaut_i_dl_croix_b() -> None:
    """`I dl x B`, en newtons : la forme integrable de la force de Lorentz."""
    force = force_sur_un_element_de_courant(
        courant=10.0,
        element=(0.1, 0.0, 0.0),
        induction=(0.0, 0.5, 0.0),
    )
    assert force[2] == pytest.approx(10.0 * 0.1 * 0.5)


def test_l_intensite_est_le_flux_de_la_densite_de_courant() -> None:
    """`I = J S` pour une densite uniforme et normale a la section."""
    assert intensite_depuis_la_densite(1e6, 1e-6) == pytest.approx(1.0)


def test_une_section_nulle_ne_porte_aucun_courant() -> None:
    with pytest.raises(ValueError, match="section nulle"):
        intensite_depuis_la_densite(1e6, 0.0)


def test_les_continuites_magnetiques_sont_l_inverse_des_electrostatiques() -> None:
    """C'est le point contre-intuitif que le fascicule met en regard.

    En magnetostatique c'est la composante **normale** de `B` qui se
    conserve et la **tangentielle** de `H`. En electrostatique c'est la
    normale de `D` et la tangentielle de `E`. Confondre les deux fait
    traverser une interface de travers.
    """
    conservees = continuite_a_une_interface()
    assert conservees == ("composante normale de B", "composante tangentielle de H")


def test_l_inductance_d_un_solenoide_croit_comme_le_carre_des_spires() -> None:
    """`mu N^2 S / l` : doubler les spires quadruple l'inductance.

    C'est ce qui rend une bobine si sensible a son bobinage, et c'est le
    `N^2` que l'on oublie.
    """
    simple = inductance_propre_d_un_solenoide(100, 1e-4, 0.1)
    double = inductance_propre_d_un_solenoide(200, 1e-4, 0.1)
    assert double == pytest.approx(4 * simple)


def test_un_solenoide_sans_spire_est_refuse() -> None:
    with pytest.raises(ValueError, match="nombre de spires"):
        inductance_propre_d_un_solenoide(0, 1e-4, 0.1)


def test_le_signe_du_couplage_depend_du_sens_des_flux() -> None:
    """« Le signe des inductances est lie au sens positif des flux ».

    C'est le piege du chapitre : la meme paire de bobines donne deux
    inductances equivalentes selon la facon dont on les branche, et le
    signe n'est pas une convention libre.
    """
    assert signe_du_couplage(flux_de_meme_sens=True) == 1
    assert signe_du_couplage(flux_de_meme_sens=False) == -1


def test_l_association_en_serie_suit_le_signe_du_couplage() -> None:
    """Recoupement avec le module existant : les deux doivent concorder."""
    paire = CouplageMagnetique(premiere=1e-3, seconde=4e-3, mutuelle=1e-3)
    aidant = (
        paire.premiere
        + paire.seconde
        + signe_du_couplage(
            flux_de_meme_sens=True,
        )
        * 2
        * paire.mutuelle
    )
    assert paire.en_serie(aidant=True) == pytest.approx(aidant)


@pytest.mark.parametrize(
    ("permeabilite_relative", "attendu"),
    [
        (0.9999, Magnetisme.DIAMAGNETIQUE),
        (1.0001, Magnetisme.PARAMAGNETIQUE),
        (5000.0, Magnetisme.FERROMAGNETIQUE),
    ],
)
def test_les_trois_magnetismes_du_fascicule(
    permeabilite_relative: float,
    attendu: Magnetisme,
) -> None:
    """Le fascicule les classe par la permeabilite relative, et les seuils sont dits.

    Sous un : diamagnetique, le materiau repousse le champ. Juste au-dessus
    de un : paramagnetique. Tres au-dessus : ferromagnetique, et la
    permeabilite n'est plus une constante mais passe par un maximum.
    """
    assert classe_de_magnetisme(permeabilite_relative) == attendu


def test_une_permeabilite_relative_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="permeabilite relative"):
        classe_de_magnetisme(-1.0)


def test_les_pertes_par_hysteresis_croissent_avec_la_frequence() -> None:
    """`f x aire du cycle x volume` : proportionnelles a la frequence.

    C'est pourquoi un transformateur chauffe plus a 400 Hz qu'a 50 Hz pour
    la meme induction, et pourquoi on empile des toles minces de materiau
    a cycle etroit.
    """
    basse = pertes_par_hysteresis(aire_du_cycle=200.0, frequence=50.0, volume=1e-3)
    haute = pertes_par_hysteresis(aire_du_cycle=200.0, frequence=400.0, volume=1e-3)
    assert haute == pytest.approx(8 * basse)
    assert basse == pytest.approx(200.0 * 50.0 * 1e-3)


def test_un_cycle_d_aire_nulle_ne_dissipe_rien() -> None:
    """Un materiau sans hysteresis : la limite idealisee du chapitre."""
    assert pertes_par_hysteresis(0.0, 50.0, 1e-3) == pytest.approx(0.0)


def test_une_frequence_negative_est_refusee_dans_l_hysteresis() -> None:
    with pytest.raises(ValueError, match="frequence negative"):
        pertes_par_hysteresis(200.0, -50.0, 1e-3)


def test_le_couplage_existant_n_est_pas_altere() -> None:
    """Non-regression sur le module deja en place."""
    paire = CouplageMagnetique(premiere=1e-3, seconde=4e-3, mutuelle=2e-3)
    assert paire.coefficient() == pytest.approx(1.0)
    assert paire.est_parfait()


def test_la_force_de_lorentz_change_de_signe_avec_la_charge() -> None:
    """Un electron et un proton tournent en sens inverses dans le meme champ."""
    positive = force_de_lorentz(1.6e-19, (1e6, 0.0, 0.0), (0.0, 0.0, 1.0))
    negative = force_de_lorentz(-1.6e-19, (1e6, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert all(p == pytest.approx(-n) for p, n in zip(positive, negative, strict=True))


def test_le_module_de_la_force_de_lorentz_vaut_q_v_b_sinus() -> None:
    """Recoupement par le module, independant du calcul composante par composante."""
    force = force_de_lorentz(1.6e-19, (1e6, 0.0, 0.0), (0.0, 1.0, 0.0))
    module = math.sqrt(sum(composante**2 for composante in force))
    assert module == pytest.approx(1.6e-19 * 1e6 * 1.0)
