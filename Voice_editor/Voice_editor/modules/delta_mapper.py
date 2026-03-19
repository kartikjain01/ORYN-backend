def generate_delta(raw_profile, studio_profile):
    """
    Compare raw vs studio profile
    """

    delta = {}

    delta["rms_diff"] = studio_profile["rms_db"] - raw_profile["rms_db"]
    delta["centroid_diff"] = studio_profile["centroid"] - raw_profile["centroid"]
    delta["lufs_diff"] = studio_profile["lufs"] - raw_profile["lufs"]
    delta["peak_diff"] = studio_profile["peak_db"] - raw_profile["peak_db"]

    return delta
