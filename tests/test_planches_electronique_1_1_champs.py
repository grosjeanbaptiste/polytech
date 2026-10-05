"""Les planches d'electronique : des equations de Maxwell aux potentiels (1, I).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_1_part1.pdf`, 51
planches lues sur les pages rendues et sur leur couche de texte.

Le fascicule reprend l'electromagnetisme « en accelere ». Ses notions
vivent donc dans `polytech.electromagnetisme`, ou le syllabus
d'electricite generale les a deja portees ; ce fichier eprouve ce que
les planches ecrivent elles-memes -- les deux jeux d'equations de
Maxwell, la conductivite, l'exemple du dielectrique sous un metal
charge, la continuite, les jauges, les equations de Jefimenko.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.resistivite import (
    resistivite_de_matthiessen,
)
from polytech.electromagnetisme.domain.conduction import champ_d_ohm
from polytech.electromagnetisme.domain.constantes import (
    CHARGE_ELEMENTAIRE,
    PERMEABILITE_DU_VIDE,
    PERMITTIVITE_DU_VIDE,
    VITESSE_DE_LA_LUMIERE,
)
from polytech.electromagnetisme.domain.electrostatique.metal_et_dielectrique import (
    champ_dans_le_dielectrique,
    champ_dans_le_vide,
    charge_liee_a_l_interface,
)
from polytech.electromagnetisme.domain.jefimenko import (
    VITESSE_TYPIQUE_DANS_UN_CIRCUIT,
    champ_electrique_elementaire,
    champ_magnetique_elementaire,
    poids_du_terme_retarde,
    temps_retarde,
)
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    Magnetisme,
    classe_de_magnetisme,
)
from polytech.electromagnetisme.domain.magnetostatique.milieux_aimantes import (
    induction_du_milieu,
)
from polytech.electromagnetisme.domain.maxwell import (
    conserve_la_charge,
    equation_de_continuite,
    residu_de_gauss_electrique,
    residu_de_gauss_magnetique,
    residus_d_ampere_maxwell,
    residus_de_faraday,
)
from polytech.electromagnetisme.domain.potentiels_variables import (
    champ_des_potentiels,
    changer_de_jauge,
    potentiel_retarde,
    residu_de_la_jauge_de_lorenz,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES, TEMPS
from polytech.electronique.domain.des_champs_aux_circuits import (
    MASSE_DE_L_ELECTRON,
    conductivite_de_drude,
    longueur_d_onde,
)

X, Y, Z = CARTESIENNES
C = sp.Symbol("c", positive=True)
OMEGA, NOMBRE = sp.symbols("omega k", positive=True)
EPS, MU = sp.symbols("epsilon mu", positive=True)


def test_les_constantes_du_vide() -> None:
    """Planche 5 : `eps_0 ~ 8,854e-12 F/m`, `mu_0 ~ 4 pi 1e-7 H/m`."""
    assert pytest.approx(8.854e-12, rel=1e-4) == PERMITTIVITE_DU_VIDE
    assert pytest.approx(4 * math.pi * 1e-7, rel=1e-9) == PERMEABILITE_DU_VIDE
    celerite = 1 / math.sqrt(PERMITTIVITE_DU_VIDE * PERMEABILITE_DU_VIDE)
    assert celerite == pytest.approx(VITESSE_DE_LA_LUMIERE, rel=1e-9)


def test_une_onde_plane_satisfait_les_quatre_equations() -> None:
    """Planches 5 et 22 : Gauss, Gauss magnetique, Faraday, Ampere, dans un milieu.

    `E = cos(omega t - k z) u_x` et `H = (k / (mu omega)) cos(...) u_y`,
    avec `k^2 = eps mu omega^2`, sans charge ni courant libres.
    """
    phase = OMEGA * TEMPS - NOMBRE * Z
    electrique = (sp.cos(phase), sp.Integer(0), sp.Integer(0))
    magnetisant = (sp.Integer(0), NOMBRE / (MU * OMEGA) * sp.cos(phase), sp.Integer(0))
    deplacement = tuple(EPS * composante for composante in electrique)
    induction = tuple(MU * composante for composante in magnetisant)
    nul = (sp.Integer(0),) * 3
    assert residu_de_gauss_electrique(deplacement, sp.Integer(0)) == 0
    assert residu_de_gauss_magnetique(induction) == 0
    assert residus_de_faraday(electrique, induction) == nul
    ampere = residus_d_ampere_maxwell(magnetisant, nul, deplacement)
    dispersion = {NOMBRE: OMEGA * sp.sqrt(EPS * MU)}
    assert tuple(sp.simplify(r.subs(dispersion)) for r in ampere) == nul
    assert sp.simplify(ampere[0]) != 0


def test_la_force_de_lorentz() -> None:
    """Planche 8 : `F = q (E + v x B)` ; le terme magnetique ne travaille pas."""
    charge, champ = -CHARGE_ELEMENTAIRE, sp.Matrix([0, 0, 100.0])
    vitesse, induction = sp.Matrix([1e5, 0, 0]), sp.Matrix([0, 0.2, 0])
    force = charge * (champ + vitesse.cross(induction))
    magnetique = charge * vitesse.cross(induction)
    assert magnetique.dot(vitesse) == 0
    assert float(force[2]) == pytest.approx(charge * (100.0 + 1e5 * 0.2))


def test_le_champ_magnetisant_et_les_trois_magnetismes() -> None:
    """Planches 15 a 19 : `H := B / mu_0 - M` ; cuivre, aluminium, fer, metglas."""
    champ, aimantation = 50.0, 2e5
    induction = induction_du_milieu(champ, aimantation)
    assert induction / PERMEABILITE_DU_VIDE - aimantation == pytest.approx(champ)
    assert classe_de_magnetisme(0.999994) is Magnetisme.DIAMAGNETIQUE
    assert classe_de_magnetisme(1.000022) is Magnetisme.PARAMAGNETIQUE
    assert classe_de_magnetisme(5000.0) is Magnetisme.FERROMAGNETIQUE
    assert classe_de_magnetisme(1e6) is Magnetisme.FERROMAGNETIQUE


def test_la_conductivite_et_le_modele_de_drude() -> None:
    """Planches 25 et 26 : `J = sigma E`, `sigma = n q_e^2 tau / m_e`."""
    densite, vol_libre = 8.5e28, 2.5e-14
    sigma = conductivite_de_drude(densite, vol_libre)
    attendue = densite * CHARGE_ELEMENTAIRE**2 * vol_libre / MASSE_DE_L_ELECTRON
    assert sigma == pytest.approx(attendue)
    courant = (sigma * 0.1, 0.0, 0.0)
    assert champ_d_ohm(1 / sigma, courant)[0] == pytest.approx(0.1)


def test_la_regle_de_matthiessen() -> None:
    """Planche 28 : `rho = rho_i + rho_ph`, seule la seconde croit avec `T`."""
    froid = resistivite_de_matthiessen(impuretes=2e-10, phonons=1.5e-8)
    chaud = resistivite_de_matthiessen(impuretes=2e-10, phonons=2.1e-8)
    assert froid == pytest.approx(1.52e-8)
    assert chaud - froid == pytest.approx(2.1e-8 - 1.5e-8)
    assert resistivite_de_matthiessen(impuretes=2e-10, phonons=0.0) == 2e-10
    with pytest.raises(ValueError, match="resistivite negative"):
        resistivite_de_matthiessen(impuretes=-1e-10, phonons=1e-8)


SIGMA, EPS_0, EPS_1 = sp.symbols("sigma_s epsilon_0 epsilon_1", positive=True)


def test_l_exemple_du_dielectrique_sous_un_metal_charge() -> None:
    """Planches 33 et 34 : `E_0 = sigma_s / eps_0`, `E_1 = (eps_0 / eps_1) E_0`."""
    vide = champ_dans_le_vide(SIGMA, EPS_0)
    dielectrique = champ_dans_le_dielectrique(SIGMA, EPS_1)
    assert vide == SIGMA / EPS_0
    assert sp.simplify(dielectrique - EPS_0 / EPS_1 * vide) == 0
    assert sp.simplify(EPS_1 * dielectrique - EPS_0 * vide) == 0
    assert float(dielectrique.subs({SIGMA: 1e-6, EPS_1: 4 * 8.854e-12})) < float(
        vide.subs({SIGMA: 1e-6, EPS_0: 8.854e-12})
    )


def test_la_charge_liee_a_l_interface() -> None:
    """Planche 36 : `sigma_b = -(eps_1 - eps_0) sigma_s / eps_1`.

    La planche ecrit entre les deux « sigma_b = (eps_1 - eps_0)(eps_0 /
    eps_1) E_0 » : il y manque le signe moins que porte sa propre ligne
    de Gauss, `-E_0 + E_1 = ((eps_0 - eps_1) / eps_1) E_0 = sigma_b / eps_0`.
    Le resultat final, lui, est juste.
    """
    liee = charge_liee_a_l_interface(SIGMA, EPS_0, EPS_1)
    assert sp.simplify(liee + (EPS_1 - EPS_0) / EPS_1 * SIGMA) == 0
    vide = champ_dans_le_vide(SIGMA, EPS_0)
    dielectrique = champ_dans_le_dielectrique(SIGMA, EPS_1)
    assert sp.simplify(EPS_0 * (dielectrique - vide) - liee) == 0
    imprimee = (EPS_1 - EPS_0) * EPS_0 / EPS_1 * vide
    assert sp.simplify(imprimee + liee) == 0
    polarisation = (EPS_1 - EPS_0) * dielectrique
    assert sp.simplify(liee + polarisation) == 0
    assert liee.subs({EPS_1: EPS_0}) == 0


def test_la_continuite_decoule_de_maxwell() -> None:
    """Planches 38 et 39 : `div J = -d rho / dt`, de `div rot B = 0` et de Gauss."""
    charge = sp.exp(-TEMPS) * sp.cos(X)
    courant = (sp.exp(-TEMPS) * sp.sin(X), sp.Integer(0), sp.Integer(0))
    assert equation_de_continuite(courant, charge) == 0
    assert conserve_la_charge(courant, charge)
    assert not conserve_la_charge(courant, 2 * charge)


PSI = sp.Function("psi")(X, Y, Z, TEMPS)
SCALAIRE = sp.Function("phi")(X, Y, Z, TEMPS)
VECTEUR = tuple(sp.Function(nom)(X, Y, Z, TEMPS) for nom in ("A_x", "A_y", "A_z"))


def test_la_liberte_de_jauge() -> None:
    """Planches 41 a 44 : `(phi - d psi/dt, A + grad psi)` rend les memes champs."""
    autre_scalaire, autre_vecteur = changer_de_jauge(SCALAIRE, VECTEUR, PSI)
    avant = champ_des_potentiels(SCALAIRE, VECTEUR)
    apres = champ_des_potentiels(autre_scalaire, autre_vecteur)
    assert all(sp.simplify(a - b) == 0 for a, b in zip(avant, apres, strict=True))
    rotation = sp.Matrix(
        [
            sp.diff(autre_vecteur[2] - VECTEUR[2], Y)
            - sp.diff(autre_vecteur[1] - VECTEUR[1], Z),
            sp.diff(autre_vecteur[0] - VECTEUR[0], Z)
            - sp.diff(autre_vecteur[2] - VECTEUR[2], X),
            sp.diff(autre_vecteur[1] - VECTEUR[1], X)
            - sp.diff(autre_vecteur[0] - VECTEUR[0], Y),
        ]
    )
    assert sp.simplify(rotation) == sp.zeros(3, 1)


def test_la_jauge_de_lorenz_et_le_potentiel_retarde() -> None:
    """Planches 45 et 46 : `div A + (1 / c^2) d phi/dt = 0`, et le temps retarde."""
    distance = sp.sqrt(X**2 + Y**2 + Z**2)
    charge = sp.Function("q")(TEMPS)
    potentiel = potentiel_retarde(charge, distance, C)
    assert potentiel.has(charge.subs(TEMPS, temps_retarde(TEMPS, distance, C)))
    onde = sp.diff(potentiel, TEMPS, 2) / C**2 - sum(
        sp.diff(potentiel, axe, 2) for axe in CARTESIENNES
    )
    assert sp.simplify(onde.doit()) == 0
    plan = sp.cos(OMEGA * (TEMPS - Z / C))
    assert residu_de_la_jauge_de_lorenz(plan, (0, 0, plan / C), C) == 0
    assert residu_de_la_jauge_de_lorenz(plan, (0, 0, 2 * plan / C), C) != 0
    with pytest.raises(ValueError, match="celerite nulle ou negative"):
        temps_retarde(1.0, 3.0, 0.0)


ECART = (sp.Integer(3), sp.Integer(0), sp.Integer(4))


def test_jefimenko_rend_coulomb_en_electrostatique() -> None:
    """Planches 47 et 48 : sans derivees ni retard, `E = rho R / (4 pi eps_0 R^3)`."""
    champ = champ_electrique_elementaire(
        ECART, sp.Integer(2), (sp.Integer(0),) * 3, celerite=C
    )
    coulomb = 1 / (4 * math.pi * PERMITTIVITE_DU_VIDE)
    attendu = [coulomb * 2 * float(composante) / 125 for composante in ECART]
    assert [float(composante) for composante in champ] == pytest.approx(attendu)


def test_jefimenko_porte_deux_termes_de_plus_que_coulomb() -> None:
    """Planche 47 : `d rho/dt` en `1 / (R^2 c)`, `-dJ/dt` en `1 / (R c^2)`, retardes."""
    charge = sp.sin(OMEGA * TEMPS)
    courant = (sp.Integer(0), sp.Integer(0), sp.cos(OMEGA * TEMPS))
    champ = champ_electrique_elementaire(ECART, charge, courant, celerite=C)
    retarde = TEMPS - 5 / C
    coulomb = 1 / (4 * sp.pi * PERMITTIVITE_DU_VIDE)
    selon_x = coulomb * (
        sp.Rational(3, 125) * sp.sin(OMEGA * retarde)
        + sp.Rational(3, 25) * OMEGA * sp.cos(OMEGA * retarde) / C
    )
    selon_z = coulomb * (
        sp.Rational(4, 125) * sp.sin(OMEGA * retarde)
        + sp.Rational(4, 25) * OMEGA * sp.cos(OMEGA * retarde) / C
        + OMEGA * sp.sin(OMEGA * retarde) / (5 * C**2)
    )
    assert sp.simplify(champ[0] - selon_x) == 0
    assert champ[1] == 0
    assert sp.simplify(champ[2] - selon_z) == 0


def test_jefimenko_rend_biot_et_savart_en_magnetostatique() -> None:
    """Planche 49 : `B = -(mu_0 / 4 pi) R x J / R^3`, soit `J x R`."""
    courant = (sp.Integer(0), sp.Integer(0), sp.Integer(7))
    champ = champ_magnetique_elementaire(ECART, courant, celerite=C)
    croise = sp.Matrix(courant).cross(sp.Matrix(ECART))
    echelle = PERMEABILITE_DU_VIDE / (4 * math.pi) / 125
    attendu = [echelle * float(composante) for composante in croise]
    assert [float(composante) for composante in champ] == pytest.approx(attendu)
    assert attendu[1] == pytest.approx(echelle * 21)


def test_le_terme_retarde_du_champ_magnetique() -> None:
    """Planche 47 : il s'ajoute `(R / (R^2 c)) x dJ/dt`, au temps retarde."""
    courant = (sp.Integer(0), sp.Integer(0), sp.cos(OMEGA * TEMPS))
    champ = champ_magnetique_elementaire(ECART, courant, celerite=C)
    retarde = TEMPS - 5 / C
    attendu = (
        PERMEABILITE_DU_VIDE
        / (4 * sp.pi)
        * (
            sp.Rational(3, 125) * sp.cos(OMEGA * retarde)
            - sp.Rational(3, 25) * OMEGA * sp.sin(OMEGA * retarde) / C
        )
    )
    assert sp.simplify(champ[1] - attendu) == 0


def test_biot_et_savart_vaut_tant_que_le_circuit_est_petit_devant_l_onde() -> None:
    """Planches 49 et 51 : le terme retarde pese `omega R / c = 2 pi R / lambda`.

    « c / 3 ~ 1e8 m/s is generally a good rule of thumb » pour la vitesse
    de propagation dans un circuit.
    """
    assert pytest.approx(1e8, rel=1e-3) == VITESSE_TYPIQUE_DANS_UN_CIRCUIT
    frequence = 50e6
    onde = longueur_d_onde(VITESSE_TYPIQUE_DANS_UN_CIRCUIT, frequence)
    poids = poids_du_terme_retarde(
        0.01, 2 * math.pi * frequence, VITESSE_TYPIQUE_DANS_UN_CIRCUIT
    )
    assert poids == pytest.approx(2 * math.pi * 0.01 / onde)
    assert poids < 0.05
    assert (
        poids_du_terme_retarde(1.0, 2 * math.pi * 1e9, VITESSE_TYPIQUE_DANS_UN_CIRCUIT)
        > 1
    )
    with pytest.raises(ValueError, match="celerite nulle ou negative"):
        poids_du_terme_retarde(0.01, 1e6, 0.0)
