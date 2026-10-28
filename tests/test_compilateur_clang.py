"""Le port `Compilateur` et son adaptateur clang++ : contrat et defaillances.

Le contrat est celui que `tests/test_cpp_ecarts_de_compilation.py` exige :
un diagnostic de syntaxe (accepte ou refuse, avec les messages du
compilateur) et une execution qui rend ce que le programme affiche.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from polytech.algorithmique.domain.compilateur import (
    Diagnostic,
    EditionDeLiensImpossible,
)
from polytech.algorithmique.infrastructure.clang import (
    CompilateurClang,
    compilateur_disponible,
)

if TYPE_CHECKING:
    from pathlib import Path

ABSENT = CompilateurClang(candidats=("compilateur-qui-n-existe-pas",))
sans_clang = pytest.mark.skipif(
    not compilateur_disponible(), reason="aucun compilateur C++ sur le poste"
)


def test_un_diagnostic_mentionne_un_fragment_de_ses_messages() -> None:
    diagnostic = Diagnostic(accepte=False, messages="error: expected ';'")
    assert diagnostic.mentionne("expected ';'")
    assert not diagnostic.mentionne("ambiguous")


def test_un_compilateur_absent_leve_au_lieu_de_se_taire() -> None:
    with pytest.raises(FileNotFoundError, match="compilateur-qui-n-existe-pas"):
        ABSENT.diagnostiquer("int main(){}")
    with pytest.raises(FileNotFoundError, match="compilateur-qui-n-existe-pas"):
        ABSENT.executer("int main(){}")


@sans_clang
def test_un_programme_valide_est_accepte() -> None:
    diagnostic = CompilateurClang().diagnostiquer("int main(){ return 0; }")
    assert diagnostic.accepte
    assert diagnostic.messages == ""


@sans_clang
def test_un_programme_invalide_est_refuse_avec_le_message_du_compilateur() -> None:
    diagnostic = CompilateurClang().diagnostiquer("int main(){ auto K; }")
    assert not diagnostic.accepte
    assert diagnostic.mentionne("requires an initializer")


@sans_clang
def test_l_execution_rend_ce_que_le_programme_affiche() -> None:
    source = "#include <iostream>\nint main(){ std::cout << 6 * 7; }"
    try:
        sortie = CompilateurClang().executer(source)
    except EditionDeLiensImpossible as erreur:
        pytest.skip(f"edition de liens impossible sur ce poste : {erreur}")
    assert sortie == "42"


@sans_clang
def test_une_source_qui_ne_compile_pas_n_est_pas_executee() -> None:
    with pytest.raises(RuntimeError, match="compilation refusee"):
        CompilateurClang().executer("int main(){ auto K; }")


def test_une_edition_de_liens_qui_echoue_partout_est_nommee(tmp_path: Path) -> None:
    # Faux compilateur : la syntaxe passe, l'edition de liens echoue toujours,
    # avec ou sans le repli sur le SDK.
    faux = tmp_path / "faux-clang"
    faux.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do [ "$a" = -fsyntax-only ] && exit 0; done\n'
        "echo 'ld: library not found' >&2\nexit 1\n",
        encoding="utf-8",
    )
    faux.chmod(0o755)
    with pytest.raises(EditionDeLiensImpossible, match="library not found"):
        CompilateurClang(candidats=(str(faux),)).executer("int main(){}")
