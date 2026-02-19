def compute_state(blink, lip, brow, nod, symmetry):

    # ---- Eye Precision Enhancement ----
    # Eye deviation amplification
    eye_amplifier = 1.4
    
    # Small eye spikes become detectable
    blink_adjusted = blink ** 1.2

    # ---- Balanced Weights (Total ≈ 1.0) ----
    weights = {
        "blink": 0.25,      # Strong eye focus
        "lip": 0.25,
        "brow": 0.15,
        "nod": 0.15,
        "symmetry": 0.20
    }

    weighted_score = (
        (blink_adjusted * eye_amplifier * weights["blink"]) +
        (lip * weights["lip"]) +
        (brow * weights["brow"]) +
        (nod * weights["nod"]) +
        (symmetry * weights["symmetry"])
    )

    # Clamp to 0–1
    weighted_score = min(weighted_score, 1.0)

    # ---- More Responsive Thresholds ----
    if weighted_score < 0.25:
        return "CALM", weighted_score
    elif weighted_score < 0.50:
        return "STRESS", weighted_score
    else:
        return "HIGH STRESS", weighted_score
