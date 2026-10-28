"""Les PDF sont produits sur la clé et restent accessibles au chemin habituel."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "outils" / "destination-des-pdf.sh"


def test_copie_compare_et_remplace_par_un_lien(tmp_path: Path) -> None:
    local = tmp_path / "local"
    cle = tmp_path / "cle"
    local.mkdir()
    cle.mkdir()
    pdf = local / "note.pdf"
    pdf.write_bytes(b"%PDF-test")
    fait = subprocess.run(  # noqa: S603 -- script local constant, chemins fournis par pytest
        ["/bin/bash", str(SCRIPT), str(local), str(cle)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fait.returncode == 0, fait.stderr
    assert pdf.is_symlink()
    assert pdf.read_bytes() == b"%PDF-test"
    assert (cle / "note.pdf").read_bytes() == b"%PDF-test"


def test_absence_de_cle_preserve_le_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "note.pdf"
    pdf.write_bytes(b"original")
    fait = subprocess.run(  # noqa: S603 -- script local constant, chemins fournis par pytest
        ["/bin/bash", str(SCRIPT), str(tmp_path), str(tmp_path / "absent")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fait.returncode != 0
    assert "Clé absente" in fait.stderr
    assert pdf.read_bytes() == b"original"


def test_conflit_sur_la_cle_preserve_les_deux_versions(tmp_path: Path) -> None:
    local = tmp_path / "local"
    cle = tmp_path / "cle"
    local.mkdir()
    cle.mkdir()
    pdf = local / "note.pdf"
    pdf.write_bytes(b"original")
    (cle / "note.pdf").write_bytes(b"autre version")
    fait = subprocess.run(  # noqa: S603 -- script local constant, chemins fournis par pytest
        ["/bin/bash", str(SCRIPT), str(local), str(cle)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fait.returncode != 0
    assert "PDF différent" in fait.stderr
    assert not pdf.is_symlink()
    assert pdf.read_bytes() == b"original"
    assert (cle / "note.pdf").read_bytes() == b"autre version"
