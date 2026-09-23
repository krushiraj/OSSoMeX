from research import schema as s


def test_valid_record():
    rec = {
        "name": "NumPy",
        "version": "1.24",
        "context_sentence": "We used NumPy 1.24.",
        "intents": ["used"],
        "sentiment": "not_expressed",
    }
    assert s.validate_public_record(rec) == []


def test_null_version_ok():
    rec = {
        "name": "NumPy",
        "version": None,
        "context_sentence": "We used NumPy.",
        "intents": ["used"],
        "sentiment": "not_expressed",
    }
    assert s.validate_public_record(rec) == []


def test_sentinel_version_rejected():
    for bad in ["null", "NA", "N/A", "none"]:
        rec = {
            "name": "X",
            "version": bad,
            "context_sentence": "t",
            "intents": ["mentioned"],
            "sentiment": "not_expressed",
        }
        assert "version" in " ".join(s.validate_public_record(rec))


def test_mentioned_alone():
    rec = {
        "name": "X",
        "version": None,
        "context_sentence": "t",
        "intents": ["mentioned"],
        "sentiment": "not_expressed",
    }
    assert s.validate_public_record(rec) == []


def test_mentioned_with_other_rejected():
    rec = {
        "name": "X",
        "version": None,
        "context_sentence": "t",
        "intents": ["mentioned", "used"],
        "sentiment": "not_expressed",
    }
    assert any("mentioned must appear alone" in p for p in s.validate_public_record(rec))


def test_split_valid_invalid():
    good = {"name": "A", "version": None, "context_sentence": "t", "intents": ["used"], "sentiment": "not_expressed"}
    bad = {"name": " ", "version": None, "context_sentence": "t", "intents": ["used"], "sentiment": "not_expressed"}
    valid, invalid = s.validate_public_records([good, bad])
    assert len(valid) == 1 and len(invalid) == 1