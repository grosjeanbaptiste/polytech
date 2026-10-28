//! Facteur spectral en marches : validation, puis moyenne par part d'energie.

use crate::texte::liste;

/// # Errors
/// Longueurs incoherentes, facteur hors de [0, 1], coupures non croissantes.
pub fn valider_bandes(facteurs: &[f64], coupures: &[f64]) -> Result<(), String> {
    if facteurs.len() != coupures.len() + 1 {
        return Err(format!(
            "{} facteurs, {} coupures (un de plus attendu)",
            facteurs.len(),
            coupures.len()
        ));
    }
    if facteurs.iter().any(|f| !(0.0..=1.0).contains(f)) {
        return Err(format!("facteur hors de [0, 1] : {}", liste(facteurs)));
    }
    if coupures.windows(2).any(|p| p[1] <= p[0]) {
        return Err(format!("coupures non croissantes : {}", liste(coupures)));
    }
    Ok(())
}

/// `parts` : fraction d'energie sous chaque coupure, calculee par l'appelant.
///
/// # Errors
/// Un facteur de plus que de parts est attendu.
pub fn moyenne_par_bandes(facteurs: &[f64], parts: &[f64]) -> Result<f64, String> {
    if facteurs.len() != parts.len() + 1 {
        return Err(format!(
            "{} facteurs, {} parts (un de plus attendu)",
            facteurs.len(),
            parts.len()
        ));
    }
    let mut bornes = vec![0.0];
    bornes.extend_from_slice(parts);
    bornes.push(1.0);
    Ok(facteurs
        .iter()
        .enumerate()
        .fold(0.0, |s, (i, f)| s + f * (bornes[i + 1] - bornes[i])))
}

#[cfg(test)]
mod tests {
    use super::{moyenne_par_bandes, valider_bandes};

    #[test]
    fn moyenne_ponderee() {
        let v = moyenne_par_bandes(&[0.2, 0.8], &[0.25]).unwrap();
        assert!((v - (0.2 * 0.25 + 0.8 * 0.75)).abs() < 1e-15);
    }

    #[test]
    fn une_seule_bande() {
        assert!((moyenne_par_bandes(&[0.3], &[]).unwrap() - 0.3).abs() < 1e-15);
    }

    #[test]
    fn longueurs_incoherentes() {
        assert_eq!(
            valider_bandes(&[0.1], &[1.0, 2.0]).unwrap_err(),
            "1 facteurs, 2 coupures (un de plus attendu)"
        );
        assert!(moyenne_par_bandes(&[0.1], &[0.5]).is_err());
    }

    #[test]
    fn facteur_hors_bornes_ou_nan() {
        assert_eq!(
            valider_bandes(&[0.5, 1.5], &[1.0]).unwrap_err(),
            "facteur hors de [0, 1] : [0.5, 1.5]"
        );
        assert!(valider_bandes(&[f64::NAN, 0.5], &[1.0]).is_err());
        assert!(valider_bandes(&[0.0, 1.0], &[1.0]).is_ok());
    }

    #[test]
    fn coupures_non_croissantes() {
        assert_eq!(
            valider_bandes(&[0.1, 0.2, 0.3], &[2.0, 2.0]).unwrap_err(),
            "coupures non croissantes : [2.0, 2.0]"
        );
        assert!(valider_bandes(&[0.1, 0.2, 0.3], &[3.0, 2.0]).is_err());
    }
}
