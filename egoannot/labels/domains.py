r"""
Universal rule core + per-domain packs.

The vocabulary is a SINGLE SOURCE OF TRUTH shared by the prompt and the linter.
Extending a pack's verbs without extending the linter's would just move the
rejection from the model to the scorer - which is what produced the six A4
failures on the diverse set.
"""
from __future__ import annotations

import re

# verbs every domain shares - the manipulation primitives
CORE_VERBS = ["reach", "grasp", "pinch", "lift", "transport", "place", "release",
              "withdraw", "align", "rotate", "twist", "flip", "push", "pull",
              "poke", "tap", "steady", "wrap", "adjust", "open", "close", "slide"]
# pinch/poke/tap/twist/flip/wrap/adjust were added after the model reached for
# them unprompted on fine-motor footage: they are manipulation primitives the
# original shelf-stocking set simply lacked, not domain vocabulary.

PACKS = {
    "retail_shelf": dict(
        label="retail shelf stocking",
        verbs=[],
        nouns=["shampoo bottle", "conditioner bottle", "snack package", "cup noodles",
               "cardboard box", "plastic basket", "shopping trolley", "shelf",
               "shelf riser", "price label", "display rack", "pen"],
        exemplars=[
            "Grasp the Schwarzkopf shampoo bottle lying inside the cardboard box with both hands",
            "Lift the Schwarzkopf shampoo bottle up out of the cardboard box toward the shelf",
            "Place the Schwarzkopf shampoo bottle upright on the middle shelf with both hands",
            "Align the Lux shampoo bottle flush with the front edge of the shelf",
            "Push the Clear shampoo bottle back against the shelf riser with the right hand",
        ]),
    "beverage": dict(
        label="beverage preparation",
        verbs=["pour", "stir", "press", "dispense", "scoop", "tear", "squeeze"],
        nouns=["ceramic cup", "paper cup", "syrup bottle", "syrup dispenser", "stirrer",
               "sugar sachet", "milk jug", "saucer", "condiment station", "counter"],
        exemplars=[
            "Grasp the white ceramic cup sitting on the dark counter with the right hand",
            "Press the syrup dispenser pump down over the white ceramic cup firmly",
            "Pour the vanilla syrup from the glass bottle into the ceramic cup",
            "Stir the coffee inside the white ceramic cup with the wooden stirrer",
            "Place the white ceramic cup back down onto the saucer with both hands",
        ]),
    "kitchen": dict(
        label="kitchen cleaning",
        verbs=["scrub", "rinse", "wash", "dry", "wipe", "pour", "scrape",
               "tilt", "drain", "remove"],
        nouns=["frying pan", "sponge", "dish brush", "sink", "tap", "draining rack",
               "washing-up liquid", "plate", "lid", "worktop"],
        exemplars=[
            "Grasp the black frying pan by its handle inside the steel sink",
            "Scrub the inside of the black frying pan with the yellow sponge",
            "Rinse the black frying pan under running water from the chrome tap",
            "Tilt the black frying pan downward over the steel sink with both hands",
            "Place the clean frying pan upside down onto the draining rack carefully",
        ]),
    "plant_care": dict(
        label="plant care",
        verbs=["pour", "unscrew", "squeeze", "tilt", "insert", "remove"],
        nouns=["nutrient bottle", "watering can", "flower pot", "plant saucer",
               "potted plant", "soil", "cap", "measuring cap", "table"],
        exemplars=[
            "Grasp the small nutrient solution bottle standing on the wooden table",
            "Unscrew the white cap from the nutrient solution bottle with the right hand",
            "Squeeze the nutrient solution bottle gently over the small transparent measuring cap",
            "Pour the nutrient solution from the measuring cap into the flower pot",
            "Place the nutrient solution bottle back upright onto the wooden table",
        ]),
    "small_assembly": dict(
        label="small-part assembly",
        verbs=["insert", "twist", "unscrew", "snap", "screw", "seat", "press"],
        nouns=["toothbrush handle", "brush head", "charger base", "battery cover",
               "connector", "cap", "housing", "small part", "table"],
        exemplars=[
            "Grasp the white toothbrush handle lying on the wooden table with both hands",
            "Insert the brush head onto the top of the white toothbrush handle",
            "Twist the brush head clockwise until it seats onto the toothbrush handle",
            "Press the brush head firmly down onto the white toothbrush handle shaft",
            "Place the assembled electric toothbrush upright onto the charger base carefully",
        ]),
    "personal_care": dict(
        label="personal care",
        verbs=["unscrew", "squeeze", "pour", "tilt", "insert", "press", "remove"],
        nouns=["solution bottle", "contact lens case", "lens case lid", "cap",
               "tissue box", "cotton pad", "bedside table", "mirror"],
        exemplars=[
            "Grasp the white contact lens solution bottle standing on the bedside table",
            "Unscrew the lid from the small contact lens case with the left hand",
            "Squeeze the solution bottle to fill the left contact lens case well",
            "Close the lid back onto the contact lens case with the right hand",
            "Place the solution bottle upright beside the lens case on the table",
        ]),
}


def verbs_for(key):
    p = PACKS.get(key) or {}
    return CORE_VERBS + [v for v in p.get("verbs", []) if v not in CORE_VERBS]


def nouns_for(key):
    return list((PACKS.get(key) or {}).get("nouns", []))


# scene / task -> pack. Keyed on what the episode metadata actually carries.
_RULES = [
    (r"coffee|syrup|espresso|latte|barista|condiment|beverage", "beverage"),
    (r"frying pan|sink|dish|wash|kitchen|pot\b|utensil|cutting board", "kitchen"),
    (r"plant|nutrient|flower pot|soil|garden|water the", "plant_care"),
    (r"assembl|toothbrush|connector|screw|snap|disassembl", "small_assembly"),
    (r"contact lens|skincare|shav|cotton|lotion|personal", "personal_care"),
    (r"shelf|stock|shampoo|snack|noodle|trolley|pen|label|belt|bin|tissue|cabinet",
     "retail_shelf"),
]


def pack_for(task=None, scene=None, cls=None):
    hay = " ".join(str(x or "") for x in (task, scene, cls)).lower()
    for pat, key in _RULES:
        if re.search(pat, hay):
            return key
    return "retail_shelf"


PROMPT_EXEMPLARS = [
    'Grasp the Schwarzkopf shampoo bottle lying inside the cardboard box with both hands',
    'Lift the Schwarzkopf shampoo bottle up out of the cardboard box toward the shelf',
    'Place the Schwarzkopf shampoo bottle upright on the middle shelf with both hands',
    'Reach down into the cardboard box toward the next shampoo bottle again',
    'Grasp another Schwarzkopf shampoo bottle inside the cardboard box using both hands',
    'Release the Schwarzkopf shampoo bottle standing upright on the middle shelf',
    'Align the Lux shampoo bottle flush with the front edge of the shelf',
    'Steady the Rejoice shampoo bottle on the shelf with the left hand',
    'Rotate the Rejoice shampoo bottle so its printed label faces the aisle',
    'Push the Clear shampoo bottle back against the shelf riser with the right hand',
]
"""The prompt's format anchor. Deliberately NOT domain-specific: measured, putting
domain vocabulary and object lists into the prompt costs 10 points of uniqueness
(88.7% -> 79.0%) for zero atomicity gain, because the model reaches for the listed
words instead of describing what it sees. Vocabulary belongs in the LINTER."""

FREE_TEXT_R2 = """R2. `verb` MUST be one of the listed verbs, spelled exactly. It is the coarse,
    machine-readable label for this action - nothing more.
    `text` is FREE. Use whatever words describe the action most precisely, including
    verbs outside the list (poke, pinch, twist, flick, dab, tap, jab, prise, tilt),
    named digits ("the right index finger"), and rotation direction. Begin `text`
    with a capitalised verb. The verb in `text` need NOT match the `verb` field."""

DETAIL_RULES = """DETAIL YOU ARE GIVEN, AND MUST USE.
Each span may carry measured facts from motion capture: hand rotation direction,
and finger state (pinch, index extended, whole hand wrapped). When a span gives
you one, put it in the sentence in those terms - "counter-clockwise", "with the
right index finger", "pinched between thumb and index". These are measurements,
not guesses; do not contradict them and do not invent them when absent.

Express them as MODIFIERS OF THE ONE ACTION, never as an extra clause:
  YES  "Pinch the ceramic cup handle between thumb and index with the right hand"
  YES  "Rotate the white bottle cap counter-clockwise with the right hand"
  NO   "Pinch the cup with the right hand while holding the saucer"   (two actions)
  NO   "Pinch the handle and lift it upward"                          (two actions)
What the non-acting hand is doing is NOT part of this caption. Never write
"while", "as" or "and" to attach a second thing that is happening."""

CORE_RULES = """HARD FORMAT RULES. A caption breaking any of these is rejected automatically.

R1. `text` MUST be 10 to 15 words. Count them. Nine words is a rejection.
    If you are under 10, add the acting hand and the specific destination or
    surface - never padding words.
R2. `text` MUST BEGIN with the verb, capitalised, spelled EXACTLY as it appears
    in the verb list below, and that same lowercase spelling goes in `verb`.
    Do not write "grab", "pick", "put", "take", "move", "set" or "hold" - they are
    not in the list. The nearest legal choices are grasp, lift, place, transport
    and steady.
R3. Exactly ONE action verb in the whole sentence. Never join two actions with
    "and", "then", "while", "before", "after", "until" or a comma. One span,
    one action. If you catch yourself describing a sequence, describe only the
    part that occupies THIS span.
    Also never state the PURPOSE of the action with "to <verb>" - write
    "Scrub the pan with the yellow sponge", not "Grasp the sponge to scrub the
    pan". Name the action happening now, not what it is for.
R4. `noun` is the single manipulated object, and its head word must appear in
    `text` (noun "shampoo bottle" -> the word "bottle" must be in the sentence).
R5. `visibility` is FULL, PARTIAL, OCCLUDED or OUT_OF_FRAME - describing the
    object and the acting hand.
R6. Never write: "continue", "etc", "neatly", "properly", "them", "those",
    "items", "second", "third", "fourth", "finish" or "complete".
R7. Describe what you see specifically - object colour, brand, material, and the
    exact surface or container involved. When consecutive spans repeat a similar
    action, individuate them by naming the specific destination, the position, or
    the object's state - NEVER by counting or numbering them.
R8. Consecutive spans are usually different STAGES of one sequence, not repeats.
    Do not reuse the previous caption's wording.

UNCERTAINTY. R7 asks for specifics; it does NOT ask you to invent them.
If the frames do not show you an object's colour, brand or identity - it is
occluded, blurred, out of frame, or your view is of the back of the hand - then:
  * set "uncertain": true
  * set "visibility" to PARTIAL, OCCLUDED or OUT_OF_FRAME as applicable
  * name the object at whatever level you actually can ("the bottle", "the
    small part", "an object in the sink") and drop the modifiers
  * you may fall below 10 words when uncertain is true
A caption marked uncertain and vague is CORRECT. A caption that guesses a colour
or a brand you cannot see is WRONG, and worse than vague, because it cannot be
told apart from a caption that got it right."""
