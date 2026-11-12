from accent_engine.alignment import align


def kinds(ops):  # type: ignore[no-untyped-def]
    return [(o.kind, o.ref, o.hyp) for o in ops]


def test_identity() -> None:
    assert all(o.kind == "match" for o in align(["TH", "IH", "NG", "K"], ["TH", "IH", "NG", "K"]))


def test_substitution_preferred_over_indel() -> None:
    ops = align(["TH", "IH", "NG", "K"], ["S", "IH", "NG", "K"])
    assert kinds(ops)[0] == ("sub", "TH", "S")


def test_epenthesis_is_insertion() -> None:
    ops = align(["S", "T", "R", "IY", "T"], ["IH", "S", "T", "R", "IY", "T"])
    assert kinds(ops)[0] == ("ins", None, "IH")
    ops = align(["S", "T", "R", "IY", "T"], ["S", "IH", "T", "R", "IY", "T"])
    assert ("ins", None, "IH") in kinds(ops)


def test_deletion() -> None:
    ops = align(["K", "AA", "R"], ["K", "AA"])
    assert kinds(ops)[-1] == ("del", "R", None)


def test_accepted_alternative_is_match() -> None:
    ops = align(["B", "AH", "T", "AH", "N"], ["B", "AH", "Q", "AH", "N"], [(), (), ("Q",), (), ()])
    assert all(o.kind == "match" for o in ops)


def test_vowel_never_substitutes_consonant() -> None:
    ops = align(["P", "AA"], ["IY", "AA"])
    assert {o.kind for o in ops} == {"del", "ins", "match"}
