import json
import os

def save_profile(profile_data, name, save_dir="profiles"):
    """
    Save studio profile as JSON
    """

    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"{name}.json")

    with open(path, "w") as f:
        json.dump(profile_data, f, indent=4)

    return path


def load_profile(name, save_dir="profiles"):
    """
    Load saved studio profile
    """

    path = os.path.join(save_dir, f"{name}.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile {name} not found")

    with open(path, "r") as f:
        profile = json.load(f)

    return profile
