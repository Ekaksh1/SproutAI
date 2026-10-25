"""
SPROUT AI - Plant-Care Recommendation Engine
Delivers grounded, practical, and botanically responsible care advice
tailored to model predictions and uncertainty thresholds.
"""

from typing import Dict, Any, List

BOTANICAL_DISCLAIMER = (
    "Botanical Integrity Notice: A camera image provides visual screening only. "
    "A photograph cannot measure subterranean soil moisture, detect root rot, "
    "or diagnose specific microscopic pathogens. Always perform physical checks before taking drastic action."
)

CARE_GUIDELINES: Dict[str, Dict[str, Any]] = {
    "healthy": {
        "status_title": "Healthy-Looking Plant",
        "badge_color": "#2E7D32",
        "bg_color": "#E8F5E9",
        "summary": "Your plant displays vibrant foliage, uniform chlorophyll distribution, and good cell turgor.",
        "primary_actions": [
            "Continue routine care schedule tailored to this specific species.",
            "Maintain current lighting and watering frequency without sudden changes.",
            "Gently wipe dust off foliage with a damp cloth once a month to maximize photosynthesis.",
            "Regularly rotate the pot a quarter turn every week for balanced, symmetrical growth."
        ],
        "monitoring_checklist": [
            "Inspect new leaf shoots for uniform shape and color.",
            "Check soil moisture before each scheduled watering (dry top inch for most tropicals).",
            "Periodically check under leaves during weekly maintenance as a pest prevention measure."
        ],
        "what_not_to_do": [
            "Do not over-fertilize during dormant seasons (autumn/winter).",
            "Do not move the plant abruptly to direct, harsh midday sun."
        ],
        "diagnostic_context": "Visual indicators suggest healthy metabolic activity. Continue preventive care."
    },
    "stressed": {
        "status_title": "Visibly Stressed Plant",
        "badge_color": "#C62828",
        "bg_color": "#FFEBEE",
        "summary": "Visible symptoms of stress detected (such as chlorosis/yellowing, leaf tip browning, wilting, or spotting).",
        "primary_actions": [
            "Perform the 2-Inch Knuckle Test: Insert your finger 2 inches into the soil. If it feels wet, DO NOT water. If completely dry and pulling from pot walls, the plant needs hydration.",
            "Review drainage: Ensure the planter has active drainage holes and empty any stagnant saucer water immediately.",
            "Assess sunlight exposure: Browning crispy edges often indicate direct sunburn or extremely dry air, while pale limp leaves indicate light deprivation.",
            "Inspect foliage undersides: Look closely for spider mite webbing, sticky scale, or aphid clusters.",
            "Prune necrotic tissue: Trim fully dead or fungal-spotted leaves with sanitized scissors to conserve plant energy."
        ],
        "monitoring_checklist": [
            "Soil moisture status (wet / moist / bone-dry).",
            "Sunlight intensity (direct / bright indirect / dim).",
            "Pot drainage state (clear / clogged / no holes).",
            "Presence of insects or powdery mildew on stems."
        ],
        "what_not_to_do": [
            "DO NOT automatically pour water! Overwatering causes root asphyxiation and root rot, presenting identical wilting symptoms to underwatering.",
            "DO NOT apply heavy chemical fertilizer to a stressed plant; it burns vulnerable roots.",
            "DO NOT repot immediately while the plant is undergoing acute environmental shock."
        ],
        "diagnostic_context": "The model noticed optical distress signatures. Follow the triage checklist above to pinpoint the environmental cause."
    },
    "uncertain": {
        "status_title": "Uncertain / Insufficient Confidence",
        "badge_color": "#E65100",
        "bg_color": "#FFF3E0",
        "summary": "The prediction confidence fell below the safe threshold, or the image features were ambiguous.",
        "primary_actions": [
            "Retake the photograph under bright, indirect natural daylight.",
            "Position the camera closer to the affected foliage, ensuring sharp focus without motion blur.",
            "Capture both a close-up of representative leaves and a wider frame showing the full plant posture.",
            "Perform a direct physical examination: feel leaf texture (crisp vs mushy) and smell the soil for sour root rot odor."
        ],
        "monitoring_checklist": [
            "Ensure the camera lens is clean and the lighting is even without glare.",
            "Check if background clutter or indoor shadows are confusing the model.",
            "Compare visual appearance against known healthy specimens of this species."
        ],
        "what_not_to_do": [
            "Do not treat an uncertain classification as a definitive health diagnosis.",
            "Do not administer pesticide or fungicide treatments without verifying the underlying issue."
        ],
        "diagnostic_context": "Responsible AI principle: SPROUT AI declines to guess when confidence is low to avoid misleading plant owners."
    }
}


def get_recommendation(predicted_class: str, confidence: float, threshold: float = 0.70) -> Dict[str, Any]:
    """
    Returns structured recommendations and diagnostics based on prediction and threshold.
    """
    # Enforce uncertainty if confidence is below threshold or class is uncertain
    if confidence < threshold or predicted_class.lower() not in ["healthy", "stressed"]:
        key = "uncertain"
        effective_class = "uncertain"
    else:
        key = predicted_class.lower()
        effective_class = key

    data = CARE_GUIDELINES.get(key, CARE_GUIDELINES["uncertain"]).copy()
    data["effective_class"] = effective_class
    data["confidence"] = confidence
    data["threshold"] = threshold
    data["disclaimer"] = BOTANICAL_DISCLAIMER
    return data
