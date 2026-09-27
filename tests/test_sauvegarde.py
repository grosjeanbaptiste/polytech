"""Le script de sauvegarde sur cle USB, eprouve sur un faux volume.

Chaque test ci-dessous correspond a un defaut *rencontre* pendant la mise au
point du script, ou a une garde qu'il doit tenir. Trois d'entre eux sont des
regressions : le script les a reellement eus, et chacun rendait la
sauvegarde soit muette, soit faussement rassurante.

Le « volume » est un simple repertoire temporaire : le script ne demande rien
d'autre a sa cible que d'exister et d'accepter des fichiers.
"""

import os
import shlex
import shutil
import subprocess
import unicodedata
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
SCRIPT = RACINE / "outils" / "sauvegarder-sur-cle.sh"
BASH = shutil.which("bash") or "/bin/bash"
GIT = shutil.which("git") or "/usr/bin/git"


def _git(*args: str, cwd: Path) -> None:
    """Lance git ; leve si la commande echoue (aucun echec silencieux)."""
    subprocess.run([GIT, *args], cwd=cwd, check=True, capture_output=True)  # noqa: S603 -- chemin resolu par shutil.which


def _depot(chemin: Path, noms: tuple[str, ...]) -> Path:
    """Un petit depot git portant les fichiers demandes, deja commite."""
    chemin.mkdir(parents=True, exist_ok=True)
    _git("init", "--quiet", cwd=chemin)
    _git("config", "user.email", "essai@exemple.test", cwd=chemin)
    _git("config", "user.name", "Essai", cwd=chemin)
    for nom in noms:
        cible = chemin / nom
        cible.parent.mkdir(parents=True, exist_ok=True)
        cible.write_text(f"contenu de {nom}\n", encoding="utf-8")
    _git("add", "--all", cwd=chemin)
    _git("commit", "--quiet", "-m", "depot d'essai", cwd=chemin)
    return chemin


def _sauvegarder(
    depot: Path, volume: Path, *options: str
) -> subprocess.CompletedProcess[str]:
    """Execute le script dans la locale de cette machine, comme en vrai."""
    environnement = dict(os.environ, LC_ALL="fr_BE.UTF-8")
    return subprocess.run(  # noqa: S603 -- chemin du script construit ici
        [BASH, str(SCRIPT), *options, str(volume)],
        cwd=depot,
        capture_output=True,
        text=True,
        env=environnement,
        check=False,
    )


@pytest.fixture
def depot_et_volume(tmp_path: Path) -> tuple[Path, Path]:
    """Un depot nomme `Polytech` et un volume vide, comme la vraie cle."""
    volume = tmp_path / "volume"
    volume.mkdir()
    return _depot(tmp_path / "Polytech", ("lisez-moi.md", "cours/planche.md")), volume


def test_la_copie_reussit_et_se_verifie(depot_et_volume: tuple[Path, Path]) -> None:
    depot, volume = depot_et_volume
    fait = _sauvegarder(depot, volume)
    assert fait.returncode == 0, fait.stderr
    assert "tout le depot est sur la cle" in fait.stdout
    assert (volume / "Polytech" / "cours" / "planche.md").is_file()


def test_la_copie_porte_le_meme_nom_que_le_depot(
    depot_et_volume: tuple[Path, Path],
) -> None:
    """« Une copie avec le meme nom » : c'est le nom du dossier du depot."""
    depot, volume = depot_et_volume
    assert _sauvegarder(depot, volume).returncode == 0
    assert [c.name for c in volume.iterdir()] == ["Polytech"]


def test_le_depot_copie_est_relu_par_git(depot_et_volume: tuple[Path, Path]) -> None:
    """Une sauvegarde de depot git qui ne se relit pas ne vaut rien."""
    depot, volume = depot_et_volume
    fait = _sauvegarder(depot, volume)
    assert "git fsck" in fait.stdout
    tetes = subprocess.run(  # noqa: S603 -- chemin resolu par shutil.which
        [GIT, "-C", str(volume / "Polytech"), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert tetes.stdout.strip()


def test_un_volume_absent_leve_avec_un_message(tmp_path: Path) -> None:
    """Aucun echec silencieux : une cle non montee doit se dire."""
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    fait = _sauvegarder(depot, tmp_path / "jamais-monte")
    assert fait.returncode == 1
    assert "volume absent" in fait.stderr


def test_une_option_inconnue_leve(depot_et_volume: tuple[Path, Path]) -> None:
    depot, volume = depot_et_volume
    fait = _sauvegarder(depot, volume, "--inconnue")
    assert fait.returncode == 1
    assert "option inconnue" in fait.stderr


def test_l_essai_n_ecrit_rien(depot_et_volume: tuple[Path, Path]) -> None:
    depot, volume = depot_et_volume
    assert _sauvegarder(depot, volume, "--essai").returncode == 0
    assert not (volume / "Polytech" / "lisez-moi.md").exists()


def test_un_depot_sans_venv_ne_meurt_pas_en_silence(tmp_path: Path) -> None:
    """Regression : `[[ -d .venv ]] && x=y` sortait sous `set -e`.

    Le test etant faux, la liste `&&` rendait 1, et `set -e` arretait le
    script sans un mot -- sur tout depot fraichement clone.
    """
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    volume = tmp_path / "volume"
    volume.mkdir()
    assert not (depot / ".venv").exists()
    fait = _sauvegarder(depot, volume)
    assert fait.returncode == 0, fait.stderr


def test_le_venv_n_est_pas_copie(tmp_path: Path) -> None:
    """456 Mo propres a la machine, et des liens que FAT ne peut pas tenir."""
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    (depot / ".venv" / "bin").mkdir(parents=True)
    (depot / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    volume = tmp_path / "volume"
    volume.mkdir()
    assert _sauvegarder(depot, volume).returncode == 0
    assert not (volume / "Polytech" / ".venv").exists()


def test_les_noms_accentues_ne_sont_pas_declares_absents(tmp_path: Path) -> None:
    """Regression : `comm` et `sort` employaient deux collations differentes.

    `LC_ALL=C` classe « Ex_cote_y » avant « Ex_cote'_x », la locale fr_BE
    l'inverse. `comm` deraillait sur les noms accentues des qu'une seule
    ligne differait, et declarait neuf PDF absents de la cle alors qu'ils y
    etaient, meme taille. Une verification qui crie au loup ne vaut rien.
    """
    compose = unicodedata.normalize("NFC", "Ex_coté_x.md")
    decompose = unicodedata.normalize("NFD", "Ex_coté_z.md")
    depot = _depot(tmp_path / "Polytech", (compose, decompose, "Ex_cote_y.md"))
    volume = tmp_path / "volume"
    volume.mkdir()
    # Un surnumeraire suffit a desynchroniser `comm` : c'est la condition qui
    # revelait le defaut.
    (volume / "Polytech").mkdir()
    (volume / "Polytech" / "vestige.md").write_text("", encoding="utf-8")
    fait = _sauvegarder(depot, volume)
    assert fait.returncode == 0, fait.stderr
    assert "tout le depot est sur la cle" in fait.stdout


def _verifier_la_copie(
    racine: Path, cible: Path, mode: str = "taille"
) -> subprocess.CompletedProcess[str]:
    """Appelle la seule fonction de verification, sans copier quoi que ce soit.

    Le script se laisse sourcer sans s'executer : c'est ce qui rend ses
    gardes eprouvables une a une.
    """
    ordre = (
        f"source {shlex.quote(str(SCRIPT))}; "
        f"verifier_la_copie {shlex.quote(str(racine))} "
        f"{shlex.quote(str(cible))} {shlex.quote(mode)}"
    )
    return subprocess.run(  # noqa: S603 -- chemin resolu par shutil.which
        [BASH, "-c", ordre],
        capture_output=True,
        text=True,
        env=dict(os.environ, LC_ALL="fr_BE.UTF-8"),
        check=False,
    )


def test_la_verification_accepte_deux_arbres_identiques(tmp_path: Path) -> None:
    ici, la = tmp_path / "ici", tmp_path / "la"
    for base in (ici, la):
        (base / "cours").mkdir(parents=True)
        (base / "cours" / "planche.md").write_text("meme contenu", encoding="utf-8")
    fait = _verifier_la_copie(ici, la)
    assert fait.returncode == 0, fait.stderr
    assert "tout le depot est sur la cle" in fait.stdout


def test_la_verification_refuse_un_fichier_absent_de_la_cible(tmp_path: Path) -> None:
    """Une verification qui ne sait pas dire non ne prouve rien."""
    ici, la = tmp_path / "ici", tmp_path / "la"
    (ici / "cours").mkdir(parents=True)
    (ici / "cours" / "planche.md").write_text("contenu", encoding="utf-8")
    la.mkdir()
    fait = _verifier_la_copie(ici, la)
    assert fait.returncode == 1
    assert "absents ou differents sur la cle" in fait.stderr
    assert "planche.md" in fait.stderr


def test_la_verification_refuse_un_fichier_tronque(tmp_path: Path) -> None:
    """Meme nom, taille differente : la copie n'est pas fidele."""
    ici, la = tmp_path / "ici", tmp_path / "la"
    ici.mkdir()
    la.mkdir()
    (ici / "planche.md").write_text("contenu entier", encoding="utf-8")
    (la / "planche.md").write_text("tronq", encoding="utf-8")
    fait = _verifier_la_copie(ici, la)
    assert fait.returncode == 1
    assert "absents ou differents" in fait.stderr


def test_la_verification_profonde_voit_ce_que_la_taille_ne_voit_pas(
    tmp_path: Path,
) -> None:
    """Deux fichiers de meme taille et de contenu different : seul SHA-256 le dit.

    C'est la raison d'etre de `--profond`, et la limite avouee du mode par
    defaut.
    """
    ici, la = tmp_path / "ici", tmp_path / "la"
    ici.mkdir()
    la.mkdir()
    (ici / "planche.md").write_text("aaaa", encoding="utf-8")
    (la / "planche.md").write_text("bbbb", encoding="utf-8")
    assert _verifier_la_copie(ici, la, "taille").returncode == 0
    assert _verifier_la_copie(ici, la, "profond").returncode == 1


def test_les_jumeaux_macos_sont_otes_de_la_copie(tmp_path: Path) -> None:
    """Regression : 566 jumeaux `._x` avaient atterri dans `.git` de la copie.

    Dont `refs/heads/._main`, que git lit comme une reference invalide : le
    depot copie etait declare corrompu alors qu'il etait intact.
    """
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    volume = tmp_path / "volume"
    volume.mkdir()
    assert _sauvegarder(depot, volume).returncode == 0
    intrus = volume / "Polytech" / ".git" / "refs" / "heads" / "._main"
    intrus.write_bytes(b"\x00\x05\x16\x07Mac OS X")
    fait = _sauvegarder(depot, volume)
    assert fait.returncode == 0, fait.stderr
    assert not intrus.exists()
    assert "jumeaux macOS otes" in fait.stdout


def test_les_surnumeraires_sont_nommes_mais_conserves(tmp_path: Path) -> None:
    """Un renommage laisse l'ancien nom sur la cle : on le dit, on ne l'ote pas.

    Supprimer d'office serait dangereux : l'ecart des formes Unicode fait
    passer un fichier present pour un fichier disparu.
    """
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    volume = tmp_path / "volume"
    volume.mkdir()
    assert _sauvegarder(depot, volume).returncode == 0
    vestige = volume / "Polytech" / "ancien-nom.md"
    vestige.write_text("garde-moi", encoding="utf-8")
    fait = _sauvegarder(depot, volume)
    assert fait.returncode == 0, fait.stderr
    assert "ancien-nom.md" in fait.stdout
    assert vestige.is_file()


def test_purger_ote_les_surnumeraires_quand_on_le_demande(tmp_path: Path) -> None:
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    volume = tmp_path / "volume"
    volume.mkdir()
    assert _sauvegarder(depot, volume).returncode == 0
    vestige = volume / "Polytech" / "ancien-nom.md"
    vestige.write_text("ote-moi", encoding="utf-8")
    fait = _sauvegarder(depot, volume, "--purger")
    assert fait.returncode == 0, fait.stderr
    assert not vestige.exists()


def test_le_script_abandonne_si_le_depot_ne_cesse_de_changer(tmp_path: Path) -> None:
    """Regression : une copie a cheval sur un commit perdait six objets git.

    On simule un depot qui bouge sans cesse par un faux `git` qui fait
    varier `status --porcelain` a chaque appel. Le script doit epuiser ses
    trois tentatives, puis refuser -- et non livrer une sauvegarde prise au
    milieu d'un commit.
    """
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    volume = tmp_path / "volume"
    volume.mkdir()
    faux = tmp_path / "bin-git-mouvant"
    faux.mkdir()
    # Le compteur vit dans le repertoire du test : deux executions
    # simultanees ne doivent pas se marcher dessus.
    compteur = tmp_path / "compteur-git"
    (faux / "git").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "status" ]; then\n'
        f"  n=$(cat {shlex.quote(str(compteur))} 2>/dev/null || echo 0)\n"
        "  n=$((n + 1))\n"
        f'  echo "$n" > {shlex.quote(str(compteur))}\n'
        '  echo " M fichier-$n"\n'
        "  exit 0\n"
        "fi\n"
        f'exec {shlex.quote(GIT)} "$@"\n',
        encoding="utf-8",
    )
    (faux / "git").chmod(0o755)
    fait = subprocess.run(  # noqa: S603 -- chemin resolu par shutil.which
        [BASH, str(SCRIPT), str(volume)],
        cwd=depot,
        capture_output=True,
        text=True,
        env=dict(os.environ, PATH=f"{faux}:{os.environ['PATH']}", LC_ALL="fr_BE.UTF-8"),
        check=False,
    )
    assert fait.returncode == 1
    assert "change sans cesse" in fait.stderr
    assert fait.stderr.count("on recommence") == 3


def test_le_script_declare_ses_outils_requis(tmp_path: Path) -> None:
    """Un outil manquant sortait en 127, sans dire lequel.

    Le PATH ne porte que `bash` : il faut bien que le script demarre pour
    pouvoir se plaindre, mais `rsync` doit lui manquer.
    """
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    volume = tmp_path / "volume"
    volume.mkdir()
    presque_vide = tmp_path / "bin-minimal"
    presque_vide.mkdir()
    (presque_vide / "bash").symlink_to(BASH)
    environnement = dict(os.environ, PATH=str(presque_vide), LC_ALL="fr_BE.UTF-8")
    fait = subprocess.run(  # noqa: S603 -- chemin du script construit ici
        [BASH, str(SCRIPT), str(volume)],
        cwd=depot,
        capture_output=True,
        text=True,
        env=environnement,
        check=False,
    )
    assert fait.returncode == 1
    assert "outil absent du systeme" in fait.stderr


def test_le_script_passe_shellcheck_et_shfmt() -> None:
    """La doctrine : formateur et linter en erreur, sans exception."""
    for outil, options in (
        ("shellcheck", ["--severity=warning"]),
        ("shfmt", ["-i", "2", "-bn", "-ci", "-sr", "-d"]),
    ):
        chemin = shutil.which(outil)
        if chemin is None:
            pytest.skip(f"{outil} absent : conformite non verifiable ici")
        fait = subprocess.run(  # noqa: S603 -- chemin resolu par shutil.which
            [chemin, *options, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert fait.returncode == 0, f"{outil} :\n{fait.stdout}{fait.stderr}"


def test_une_divergence_sur_depot_stable_est_un_defaut_reel(tmp_path: Path) -> None:
    """Regression : la verification tombait hors de la fenetre de stabilite.

    L'empreinte du depot n'encadrait que la copie. Une edition survenue
    entre la copie et le controle se lisait donc comme une copie infidele,
    et le script echouait sur un depot pourtant bien sauvegarde. Elle
    encadre desormais les deux -- mais une divergence a depot *stable*
    doit rester fatale, sinon le controle ne sert plus a rien.
    """
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    volume = tmp_path / "volume"
    volume.mkdir()
    assert _sauvegarder(depot, volume).returncode == 0
    # On mutile la copie *apres* coup, sans toucher au depot : rsync la
    # repare, donc pour eprouver le verdict on appelle le controle seul.
    (volume / "Polytech" / "a.md").write_text("tronque", encoding="utf-8")
    fait = _verifier_la_copie(depot, volume / "Polytech")
    assert fait.returncode == 1
    assert "absents ou differents" in fait.stderr


def test_la_purge_epargne_un_nom_ecrit_sous_l_autre_forme_unicode(
    tmp_path: Path,
) -> None:
    """Regression : `rsync --delete` effacait les fichiers accentues de la copie.

    rsync compare des octets. Sur la cle, il ecrivait « Séance.tex » puis
    relisait le nom sous l'autre forme Unicode, le tenait pour un fichier
    disparu du depot, et le supprimait -- apres l'avoir ecrit. Mesure sur
    la vraie cle : tout `travaux/MécaniqueFluides/` et trente fichiers
    accentues otes de la sauvegarde. La purge se fait donc sur la
    comparaison normalisee, jamais par rsync.
    """
    depot = _depot(tmp_path / "Polytech", (unicodedata.normalize("NFC", "Séance.md"),))
    volume = tmp_path / "volume"
    volume.mkdir()
    copie = volume / "Polytech"
    copie.mkdir()
    # La copie porte le meme fichier sous la forme decomposee, comme le fait
    # le pilote FAT, plus un vrai surnumeraire.
    (copie / unicodedata.normalize("NFD", "Séance.md")).write_text(
        "x", encoding="utf-8"
    )
    (copie / "vestige.md").write_text("x", encoding="utf-8")
    fait = _sauvegarder(depot, volume, "--purger")
    assert fait.returncode == 0, fait.stderr
    assert not (copie / "vestige.md").exists()
    restants = [unicodedata.normalize("NFC", c.name) for c in copie.iterdir()]
    assert "Séance.md" in restants


def test_les_pdf_partent_sur_la_cle_bien_qu_ils_ne_soient_pas_dans_git(
    tmp_path: Path,
) -> None:
    """Aucun PDF dans git, tous sur la cle : les deux moities de la meme regle.

    Un PDF ignore par git reste un fichier du depot, et la sauvegarde n'a pas
    a reproduire les exclusions de git -- c'est meme l'inverse : ce que git
    ne garde pas, seule la cle le conserve.
    """
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    (depot / ".gitignore").write_text("*.pdf\n", encoding="utf-8")
    (depot / "compose.pdf").write_bytes(b"%PDF-1.7 faux mais suffisant\n")
    volume = tmp_path / "volume"
    volume.mkdir()
    fait = _sauvegarder(depot, volume, "--sans-pdf")
    assert fait.returncode == 0, fait.stderr
    assert (volume / "Polytech" / "compose.pdf").is_file()


def test_aucun_outil_ne_cherche_les_sources_dans_passpolytech() -> None:
    """Le dossier `PASSPOLYTECH` est abandonne : la copie du depot le remplace.

    Ses 114 fichiers sont tous dans `<cle>/Polytech/`, verifie contenu par
    contenu, et celle-ci porte en plus les 132 PDF composes et l'historique
    git. Un outil qui irait encore y lire designerait une source de verite
    qui n'en est plus une ; un outil qui y ecrirait deposerait un fichier
    que ni git ni la sauvegarde ne reprendraient.
    """
    outils = (*RACINE.glob("outils/*.sh"), *RACINE.rglob("outils/**/*.py"))
    fautifs = [
        str(chemin.relative_to(RACINE))
        for chemin in outils
        if "PASSPOLYTECH" in chemin.read_text(encoding="utf-8", errors="replace")
    ]
    assert fautifs == []


def test_les_consignes_disent_ou_vivent_les_sources_hors_du_depot() -> None:
    """Un agent qui cherche une source absente doit savoir ou regarder.

    La consigne doit nommer les deux faces de la decision : `Polytech/` est
    le bon endroit, `PASSPOLYTECH` ne l'est plus. Nommer seulement le bon
    laisserait l'ancien reflexe intact.
    """
    consignes = (RACINE / "AGENTS.md").read_text(encoding="utf-8")
    assert "PASSPOLYTECH" in consignes
    assert "abandonné" in consignes
    assert "<clé>/Polytech/" in consignes


def _empreinte(depot: Path) -> str:
    """L'empreinte que le script releve avant et apres chaque passe."""
    ordre = (
        f"source {shlex.quote(str(SCRIPT))}; "
        f"empreinte_du_depot {shlex.quote(str(depot))}"
    )
    fait = subprocess.run(  # noqa: S603 -- chemin resolu par shutil.which
        [BASH, "-c", ordre],
        cwd=depot,
        capture_output=True,
        text=True,
        check=True,
    )
    return fait.stdout


def test_l_empreinte_voit_un_fichier_que_git_ignore(tmp_path: Path) -> None:
    """Regression : l'empreinte ne regardait que l'arbre suivi par git.

    L'essentiel de ce que la sauvegarde emporte est justement ignore par git
    -- les PDF sources, les 132 documents composes. Des cours arrives dans
    `ProgrammationAlgorithmique/` pendant une passe laissaient donc le depot
    « stable » au sens de git, et la verification lisait cela comme une copie
    infidele : un defaut annonce la ou il n'y avait qu'une source mouvante.
    """
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    (depot / ".gitignore").write_text("sources/\n", encoding="utf-8")
    (depot / "sources").mkdir()
    avant = _empreinte(depot)
    (depot / "sources" / "cours.pdf").write_bytes(b"%PDF nouveau\n")
    assert _empreinte(depot) != avant


def test_un_cache_n_est_ni_copie_ni_reclame(tmp_path: Path) -> None:
    """Les deux listes d'exclusion doivent dire la meme chose.

    rsync et `find` ont des syntaxes differentes, donc le script ecrit ses
    exclusions deux fois. Si elles divergent, la copie est declaree infidele
    a jamais : rsync n'emporte pas le fichier, et la verification le
    reclame. Ce test les confronte sur un cas reel.

    Ces caches sont exclus parce qu'ils changent a chaque execution des
    tests : `.pytest_cache/v/cache/nodeids` faisait osciller l'empreinte du
    depot d'une passe a l'autre, pour 21 Mo sans valeur de sauvegarde.
    """
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    for cache in (".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__"):
        (depot / cache).mkdir()
        (depot / cache / "trace").write_text("volatil", encoding="utf-8")
    volume = tmp_path / "volume"
    volume.mkdir()
    fait = _sauvegarder(depot, volume, "--sans-pdf")
    assert fait.returncode == 0, fait.stdout + fait.stderr
    assert "tout le depot est sur la cle" in fait.stdout
    for cache in (".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__"):
        assert not (volume / "Polytech" / cache).exists(), cache


def test_un_cache_profond_n_est_pas_reclame_non_plus(tmp_path: Path) -> None:
    """`__pycache__` vit a toute profondeur, pas seulement a la racine."""
    depot = _depot(tmp_path / "Polytech", ("paquet/module.py",))
    (depot / "paquet" / "__pycache__").mkdir()
    (depot / "paquet" / "__pycache__" / "module.pyc").write_bytes(b"\x00")
    volume = tmp_path / "volume"
    volume.mkdir()
    fait = _sauvegarder(depot, volume, "--sans-pdf", "--profond")
    assert fait.returncode == 0, fait.stdout + fait.stderr
    assert not (volume / "Polytech" / "paquet" / "__pycache__").exists()


def test_un_cache_perime_est_ote_de_la_copie(tmp_path: Path) -> None:
    """Un cache laisse par une copie ancienne ne doit pas y rester indefiniment.

    Depuis qu'ils sont exclus, ni la verification ni la purge ne les voient :
    toutes deux les elaguent. Sans ce nettoyage, 21 Mo de caches perimes
    seraient restes sur la cle, invisibles a l'outillage qui les ignore.
    """
    depot = _depot(tmp_path / "Polytech", ("a.md",))
    volume = tmp_path / "volume"
    volume.mkdir()
    assert _sauvegarder(depot, volume, "--sans-pdf").returncode == 0
    perime = volume / "Polytech" / ".mypy_cache" / "3.14"
    perime.mkdir(parents=True)
    (perime / "cache.db").write_bytes(b"vieux")
    profond = volume / "Polytech" / "paquet" / "__pycache__"
    profond.mkdir(parents=True)
    (profond / "x.pyc").write_bytes(b"\x00")
    fait = _sauvegarder(depot, volume, "--sans-pdf")
    assert fait.returncode == 0, fait.stdout + fait.stderr
    assert not (volume / "Polytech" / ".mypy_cache").exists()
    assert not profond.exists()
    assert "dossiers jetables otes" in fait.stdout
