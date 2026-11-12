from accent_engine.g2p import build_target, g2p, lexicon_for, lookup, tokenize


def phones(text: str) -> list[str]:
    return [p.phone for p in g2p(text).phones]


def test_cmudict_stress_shift_variants() -> None:
    rec = lookup("record")
    patterns = {tuple(s for _, s in v if s is not None) for v in rec.variants}
    assert (1, 0) in patterns  # REcord (noun)
    assert (0, 1) in patterns  # reCORD (verb)


def test_flap_within_word_and_across_words() -> None:
    w = g2p("water").phones
    assert w[2].phone == "DX" and w[2].underlying == "T" and w[2].allophone == "flap"
    assert "T" in w[2].alternatives
    # "at all": word-final T before a vowel-initial word flaps even before a stressed vowel
    assert "DX" in phones("look at all")
    # no flap before a stressed vowel inside a word, nor word-initially
    assert "DX" not in phones("italian")
    assert "DX" not in phones("a toy")


def test_glottal_t_before_syllabic_n() -> None:
    t = g2p("button").phones
    assert t[2].phone == "T" and t[2].allophone == "glottal" and "Q" in t[2].alternatives


def test_dark_l_marked_in_coda_only() -> None:
    feel = g2p("feel").phones
    assert feel[-1].allophone == "dark_l"
    leaf = g2p("leaf").phones
    assert leaf[0].allophone is None


def test_weak_forms_default_and_strong_accepted() -> None:
    to = lookup("to")
    assert [p for p, _ in to.default] == ["T", "AH"]
    assert any([p for p, _ in v] == ["T", "UW"] for v in to.variants)


def test_context_reductions() -> None:
    lex = lexicon_for("I am going to go")
    going = lex[2]
    assert any([p for p, _ in v] == ["G", "AH", "N"] for v in going.variants)
    assert any([p for p, _ in v] == ["AH"] for v in lex[3].variants)
    # default target stays the careful form
    assert [p for p, _ in going.default] != ["G", "AH", "N"]


def test_positions() -> None:
    t = g2p("street").phones
    assert [p.position for p in t] == ["initial", "medial", "medial", "medial", "final"]


def test_oov_fallback_and_numbers() -> None:
    wp = lookup("blorptastic")
    assert wp.source in ("rules", "g2p_model")
    assert len(wp.default) > 4
    assert tokenize("I have 21 cats") == ["i", "have", "twenty", "one", "cats"]


def test_variant_choice() -> None:
    lex = lexicon_for("to")
    strong = build_target(lex, [1])
    assert [p.phone for p in strong.phones] == ["T", "UW"]
