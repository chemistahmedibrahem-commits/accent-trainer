"""Static practice content. The adaptive engine (build step 7) will select from this."""

from __future__ import annotations

MINIMAL_PAIRS: dict[str, list[tuple[str, str]]] = {
    "P_B": [
        ("pat", "bat"),
        ("pear", "bear"),
        ("cap", "cab"),
        ("pig", "big"),
        ("rope", "robe"),
        ("pull", "bull"),
        ("simple", "symbol"),
    ],
    "V_F": [
        ("van", "fan"),
        ("very", "fairy"),
        ("vine", "fine"),
        ("leave", "leaf"),
        ("save", "safe"),
        ("vast", "fast"),
    ],
    "TH_SUB": [
        ("think", "sink"),
        ("thank", "tank"),
        ("three", "tree"),
        ("path", "pass"),
        ("thought", "taught"),
        ("bath", "bat"),
    ],
    "DH_SUB": [("they", "day"), ("then", "den"), ("breathe", "breeze"), ("those", "doze"), ("than", "Dan")],
    "IH_IY": [
        ("ship", "sheep"),
        ("live", "leave"),
        ("bit", "beat"),
        ("fill", "feel"),
        ("sit", "seat"),
        ("still", "steal"),
    ],
    "UH_UW": [("full", "fool"), ("pull", "pool"), ("look", "Luke"), ("should", "shoed"), ("could", "cooed")],
    "AE_EH": [
        ("bad", "bed"),
        ("man", "men"),
        ("sat", "set"),
        ("had", "head"),
        ("pan", "pen"),
        ("laughed", "left"),
    ],
    "FINAL_DEVOICING": [
        ("bag", "back"),
        ("dog", "dock"),
        ("prize", "price"),
        ("rib", "rip"),
        ("raised", "raced"),
    ],
}

SENTENCES: dict[str, list[str]] = {
    "FLAP_T": ["I'd like a bottle of water.", "Better late than never.", "Put it on the city map."],
    "RHOTIC": [
        "The early bird gets the worm.",
        "Her brother works in a car factory.",
        "Read the report from the research team.",
    ],
    "EPENTHESIS": [
        "Students study in the street.",
        "Spring is a special season.",
        "The script was strictly structured.",
    ],
    "SCHWA_REDUCTION": ["I want a cup of coffee.", "Give it to the doctor.", "Can you see the problem?"],
    "DARK_L": ["I feel well, I'll call you.", "Fill the bottle with milk.", "The whole school was full."],
    "stress": ["I want to record a record.", "Please present the present.", "They object to the object."],
    "lab": [
        "We ran the titration three times.",
        "The reaction mixture turned yellow.",
        "Please label the beakers clearly.",
    ],
}
