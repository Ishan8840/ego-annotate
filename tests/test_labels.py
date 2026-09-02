"""The atomicity linter, including the vocabulary collision it used to miss."""
from egoannot.labels import atomicity as AL
from egoannot.labels import domains as DM


def label(text, **kw):
    base = dict(start_ts=0.0, end_ts=2.0, text=text, verb="grasp",
                noun="shampoo bottle", hand="RIGHT", visibility="FULL")
    base.update(kw)
    return base


def errors(lab):
    return {code for sev, code, _ in AL.lint(lab) if sev == "ERROR"}


def test_selftest_passes():
    """Curated positives pass, curated negatives fail, exemplars lint clean."""
    assert AL.selftest()


def test_noun_that_is_also_a_verb_is_not_a_second_action():
    """
    `tap` is a CORE_VERB and the kitchen pack's own noun, so this sentence --
    shipped in the prompt as a worked example -- used to fail A3.
    """
    AL.use_domain("kitchen")
    lab = label("Rinse the black frying pan under running water from the chrome tap",
                verb="rinse", noun="frying pan")
    assert "A3" not in errors(lab)
    AL.use_domain("retail_shelf")


def test_a_real_second_verb_is_still_caught():
    AL.use_domain("retail_shelf")
    lab = label("Grasp the shampoo bottle and place the shampoo bottle on the shelf")
    assert "A3" in errors(lab)


def test_leading_verb_counts_even_when_it_is_also_a_noun_head():
    AL.use_domain("kitchen")
    lab = label("Tap the frying pan handle gently against the steel sink edge twice",
                verb="tap", noun="frying pan")
    assert "A3" not in errors(lab)      # exactly one core, the leading verb
    AL.use_domain("retail_shelf")


def test_verb_match_is_word_bounded():
    """Substring matching accepted `place` inside `placemat`."""
    AL.use_domain("retail_shelf")
    lab = label("Steady the shampoo bottle upright against the placemat with one hand",
                verb="place")
    assert "A4" in errors(lab)


def test_span_band_rules():
    AL.use_domain("retail_shelf")
    assert "A1" in errors(label("Grasp the shampoo bottle on the shelf with the "
                                "right hand", start_ts=0.0, end_ts=0.9))
    assert "A1" in errors(label("Grasp the shampoo bottle on the shelf with the "
                                "right hand", start_ts=0.0, end_ts=9.0))
    assert "A1" not in errors(label("Grasp the shampoo bottle on the shelf with "
                                    "the right hand", start_ts=0.0, end_ts=2.0))


def test_word_count_bounds_and_the_uncertain_escape_hatch():
    AL.use_domain("retail_shelf")
    short = label("Grasp the bottle", noun="shampoo bottle")
    assert "A2" in errors(short)
    assert "A2" not in errors(dict(short, uncertain=True))


def test_a9_fires_only_against_a_measured_rotation():
    AL.use_domain("retail_shelf")
    text = ("Rotate the shampoo bottle counter-clockwise on the shelf with the "
            "right hand")
    lab = label(text, verb="rotate")
    assert "A9" not in errors(lab)
    assert "A9" in errors(dict(lab, rotation="clockwise"))
    assert "A9" not in errors(dict(lab, rotation="counter-clockwise"))


def test_overlapping_labels_are_rejected_but_gaps_are_allowed():
    AL.use_domain("retail_shelf")
    text = "Grasp the shampoo bottle on the shelf with the right hand"
    a = label(text, start_ts=0.0, end_ts=2.0, episode="e")
    b = label(text, start_ts=1.0, end_ts=3.0, episode="e")
    res = AL.lint_stream([a, b])
    assert any(code == "A8" for _, errs in res for _, code, _ in errs)
    c = label(text, start_ts=5.0, end_ts=7.0, episode="e")
    res = AL.lint_stream([a, c])
    assert not any(code == "A8" for _, errs in res for _, code, _ in errs)


def test_hand_neither_requires_a_contactless_verb():
    AL.use_domain("retail_shelf")
    text = "Reach toward the shampoo bottle standing on the middle shelf slowly"
    assert "A5" not in errors(label(text, verb="reach", hand="NEITHER"))
    text2 = "Grasp the shampoo bottle standing on the middle shelf quite firmly"
    assert "A5" in errors(label(text2, verb="grasp", hand="NEITHER"))


def test_every_pack_shares_the_core_verbs():
    for pack in DM.PACKS:
        verbs = DM.verbs_for(pack)
        assert set(DM.CORE_VERBS) <= set(verbs)
        assert len(verbs) == len(set(verbs))
