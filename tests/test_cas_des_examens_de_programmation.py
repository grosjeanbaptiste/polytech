"""Les cas particuliers de la fiche de programmation, verifies en C++.

Chaque cas que la fiche reprend est un comportement de la bibliotheque
standard : un programme les eprouve tous, compile en C++17 avec les options
du depot, et affiche une ligne par cas. Sans compilateur, le test se
suspend, en le disant.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from polytech.algorithmique.infrastructure.compilation import (
    DELAI,
    OPTIONS,
    compilateur,
)

if TYPE_CHECKING:
    from pathlib import Path

PROGRAMME = r"""
#include <algorithm>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <string>
#include <vector>
using namespace std;

int main() {
    map<string, int> m;
    m.insert(make_pair(string("Apple"), 1));
    bool ajoute = m.insert(make_pair(string("Apple"), 2)).second;
    cout << "doublon " << ajoute << " " << m["Apple"] << endl;

    cout << "absent " << (m.find("Zeta") == m.end()) << endl;

    multimap<string, string> pays;
    pays.insert(make_pair(string("OCEANIA"), string("Fiji")));
    auto plage = pays.equal_range("EUROPE");
    cout << "vide " << (plage.first == plage.second) << endl;

    vector<int> region = {4, 7, 1};
    size_t combien = min<size_t>(5, region.size());
    cout << "cinq " << combien << endl;

    vector<int> ages = {30, 25, 30, 19};
    auto plus = max_element(ages.begin(), ages.end());
    cout << "exaequo " << distance(ages.begin(), plus) << " "
         << count(ages.begin(), ages.end(), *plus) << endl;

    list<int> l = {1, 2, 3, 4};
    for (auto it = l.begin(); it != l.end();) {
        if (*it % 2 == 0) { it = l.erase(it); } else { ++it; }
    }
    cout << "erase";
    for (int v : l) { cout << " " << v; }
    cout << endl;

    m.clear();
    cout << "clear " << m.empty() << " " << m.size() << endl;

    list<int> l1 = {1, 2};
    vector<int> v2 = {7, 8, 9};
    list<int> t(v2.begin(), v2.end());
    v2.assign(l1.begin(), l1.end());
    l1 = t;
    cout << "permute " << l1.size() << " " << v2.size() << endl;

    list<int> a = {1, 3};
    vector<int> b = {2, 4, 6, 8};
    vector<int> r;
    auto i = a.begin();
    auto j = b.begin();
    while (i != a.end() || j != b.end()) {
        if (i != a.end()) { r.push_back(*i++); }
        if (j != b.end()) { r.push_back(*j++); }
    }
    cout << "entrelace";
    for (int v : r) { cout << " " << v; }
    cout << endl;
    return 0;
}
"""

ATTENDU = [
    "doublon 0 1",
    "absent 1",
    "vide 1",
    "cinq 3",
    "exaequo 0 2",
    "erase 1 3",
    "clear 1 0",
    "permute 3 2",
    "entrelace 1 2 3 4 6 8",
]


def _lancer(commande: list[str], dossier: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- commande batie ici, sans shell
        commande,
        capture_output=True,
        text=True,
        timeout=DELAI,
        check=False,
        cwd=dossier,
    )


def test_chaque_cas_de_la_stl_se_comporte_comme_la_fiche_le_dit(tmp_path: Path) -> None:
    outil = compilateur()
    if outil is None:
        pytest.skip("aucun compilateur C++ (clang++ ou g++) sur ce poste")
    (tmp_path / "cas.cpp").write_text(PROGRAMME, encoding="ascii")
    compilation = _lancer([outil, *OPTIONS, "cas.cpp", "-o", "cas"], tmp_path)
    assert compilation.returncode == 0, compilation.stderr
    execution = _lancer([str(tmp_path / "cas")], tmp_path)
    assert execution.returncode == 0, execution.stderr
    assert execution.stdout.splitlines() == ATTENDU
