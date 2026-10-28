"""Le noyau numerique natif (crate `polytech-core`), ou une erreur claire.

Aucun repli en Python : si l'extension manque, importer ce module leve.
"""

from __future__ import annotations

try:
    import polytech_core as core
except ImportError as erreur:
    msg = (
        "extension native `polytech_core` introuvable : construire le noyau avec "
        "`dotfiles heavy run -- maturin develop --release -m polytech-core/Cargo.toml` "
        "(voir polytech-core/README.md)"
    )
    raise ImportError(msg) from erreur

__all__ = ["core"]
