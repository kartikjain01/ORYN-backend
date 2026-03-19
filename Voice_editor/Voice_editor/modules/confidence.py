def calculate_confidence(delta):
    """
    Higher mismatch = lower confidence
    """

    score = 1.0

    score -= abs(delta["rms_diff"]) * 0.02
    score -= abs(delta["centroid_diff"]) * 0.0005
    score -= abs(delta["lufs_diff"]) * 0.03

    score = max(0.0, min(1.0, score))

    return round(score, 3)

