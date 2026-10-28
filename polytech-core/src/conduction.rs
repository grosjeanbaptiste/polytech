//! Rayon critique d'un isolant autour d'un tube : `r_c = lambda / K_e`.

use crate::texte::repr;

fn exiger_strictement_positif(valeur: f64, grandeur: &str) -> Result<(), String> {
    if !valeur.is_finite() {
        return Err(format!("{grandeur} : valeur non finie ({})", repr(valeur)));
    }
    if valeur <= 0.0 {
        return Err(format!(
            "{grandeur} nulle ou negative interdite : {}",
            repr(valeur)
        ));
    }
    Ok(())
}

/// # Errors
/// Conductivite ou coefficient d'echange non fini ou non strictement positif.
pub fn rayon_critique(conductivite: f64, echange: f64) -> Result<f64, String> {
    exiger_strictement_positif(conductivite, "conductivite de l'isolant")?;
    exiger_strictement_positif(echange, "coefficient d'echange exterieur")?;
    Ok(conductivite / echange)
}

#[cfg(test)]
mod tests {
    use super::rayon_critique;

    #[test]
    fn nominal() {
        assert!((rayon_critique(0.04, 8.0).unwrap() - 0.005).abs() < 1e-15);
    }

    #[test]
    fn conductivite_invalide() {
        let e = rayon_critique(0.0, 8.0).unwrap_err();
        assert_eq!(
            e,
            "conductivite de l'isolant nulle ou negative interdite : 0.0"
        );
        assert!(rayon_critique(-1.0, 8.0).is_err());
    }

    #[test]
    fn echange_invalide() {
        let e = rayon_critique(0.04, -3.0).unwrap_err();
        assert_eq!(
            e,
            "coefficient d'echange exterieur nulle ou negative interdite : -3.0"
        );
    }

    #[test]
    fn non_fini_refuse() {
        let e = rayon_critique(f64::NAN, 1.0).unwrap_err();
        assert_eq!(e, "conductivite de l'isolant : valeur non finie (nan)");
        assert!(rayon_critique(1.0, f64::INFINITY).is_err());
    }
}
