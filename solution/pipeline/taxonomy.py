"""Canonical service taxonomy shared by the database build and the proposal pipeline.

Domain constants only: validation lives in create_db.py and matching logic in
the pipeline modules that import these names.
"""

SERVICE_TYPES = ("activity", "food", "transport", "venue")
DIETARY_OPTIONS = ("vegan", "vegetarian")

# Controlled tag vocabulary, grouped by the dimension each tag describes.
# Every tag a service uses must be listed here, so typos and synonym variants
# (e.g. "karting" next to "go_kart") fail validation.
TAG_VOCABULARY = {
    "activity": (
        "go_kart", "escape_room", "kayak", "sailing", "boat_trip", "padel", "cycling",
        "guided_tour", "flamenco", "wine_tasting", "cocktails", "cooking", "workshop",
        "culture",
    ),
    "food": ("tapas", "dinner", "lunch", "catering", "seafood"),
    "setting": ("private_room", "rooftop", "beach", "meeting_space", "indoor", "outdoor"),
    "transport": ("coach", "minibus"),
    "experience": ("active", "relaxed", "competitive", "fun", "casual", "premium", "memorable"),
}
ALL_TAGS = frozenset(tag for tags in TAG_VOCABULARY.values() for tag in tags)

# Meanings of tags whose name alone is ambiguous. The experience tags are
# separate axes, not synonyms: a service can be both active and competitive.
TAG_DESCRIPTIONS = {
    "sailing": "participants sail or crew the boat themselves",
    "boat_trip": "a cruise or charter where the crew does the sailing",
    "guided_tour": "a visit led by a guide, on foot, by bike or inside a museum",
    "workshop": "a hands-on session where participants make or learn something",
    "culture": "local heritage, art or tradition",
    "catering": "food delivered and served at the client's own venue",
    "private_room": "a dining room or salon reserved exclusively for the group",
    "meeting_space": "a venue equipped for presentations, training or meetings",
    "indoor": "takes place mainly indoors (a service may be both indoor and outdoor)",
    "outdoor": "takes place mainly outdoors (a service may be both indoor and outdoor)",
    "coach": "a full-size coach with driver, for larger groups",
    "minibus": "a small minibus with driver, for smaller groups",
    "active": "physically demanding or sporty pace",
    "relaxed": "unhurried, low-effort pace",
    "competitive": "a race, match or contest with a winner",
    "fun": "a light-hearted, game-like format",
    "casual": "informal, no-frills dining; the opposite of a formal meal",
    "premium": "an upscale price and quality tier",
    "memorable": "a distinctive, out-of-the-ordinary experience",
}
