# polytech-core

Noyaux numeriques purs (flottants) de `polytech/`, en Rust, exposes a Python
par PyO3 (abi3, Python >= 3.12) sous le module `polytech_core`.

Construire et installer dans `.venv` (prerequis : `dotfiles add --manager pipx maturin`) :

    VIRTUAL_ENV=$PWD/.venv dotfiles heavy run -- maturin develop --release -m polytech-core/Cargo.toml

Verifier : `dotfiles heavy run --quick -- cargo test --manifest-path polytech-core/Cargo.toml`,
puis `cargo fmt --check` et `cargo clippy --all-targets -- -D warnings`.

Les facades Python importent `polytech.noyau`, qui leve `ImportError` (avec cette
commande) si l'extension manque : jamais de repli silencieux en Python.
Les constantes physiques restent dans le Python et sont passees en argument.
Stubs : `polytech_core.pyi` (mypy --strict).
