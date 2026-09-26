"""La table de fleches doit etre publiee comme le sont les macros.

Meme discipline : un fichier genere, jamais edite a la main, et que le
document se contente d'inclure.
"""

from pathlib import Path

import pytest

from polytech.fluides.domain.cinematique.echantillonnage import Fleche
from polytech.fluides.infrastructure.table import FichierDeTable

FLECHES = (
    Fleche(x=0.5, y=0.6, u=0.3521, v=-0.4225),
    Fleche(x=1.5, y=1.8, u=0.3521, v=-0.4225),
)


def test_la_table_porte_l_entete_attendu_par_pgfplots(tmp_path: Path) -> None:
    """Pgfplots lit les colonnes par leur nom : x, y, u, v."""
    chemin = tmp_path / "champ.dat"
    FichierDeTable(chemin=chemin).publier(FLECHES)
    lignes = chemin.read_text(encoding="utf-8").splitlines()
    assert lignes[0].startswith("%")  # avertissement « fichier genere »
    assert lignes[1].split() == ["x", "y", "u", "v"]


def test_chaque_fleche_donne_une_ligne(tmp_path: Path) -> None:
    chemin = tmp_path / "champ.dat"
    FichierDeTable(chemin=chemin).publier(FLECHES)
    corps = chemin.read_text(encoding="utf-8").splitlines()[2:]
    assert len(corps) == len(FLECHES)
    assert [float(v) for v in corps[0].split()] == [0.5, 0.6, 0.3521, -0.4225]


def test_publier_une_table_vide_est_refuse(tmp_path: Path) -> None:
    """Une figure sans fleche serait une figure muette : on leve."""
    with pytest.raises(ValueError, match="aucune fleche"):
        FichierDeTable(chemin=tmp_path / "champ.dat").publier(())
