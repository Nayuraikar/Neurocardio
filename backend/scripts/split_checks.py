"""Leakage assertions for the multi-database split. No dependency beyond the standard library (runs on Kaggle)."""

# de Chazal et al. 2004, DS2
DS2_RECORDS = "100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234".split()
PARTS = ("train", "validation", "test_ds2", "test_incart")


def check_split(split: dict) -> None:
    patients = split["patients"]
    people = {p: {patients[r] for r in split[p]} for p in PARTS}
    for i, a in enumerate(PARTS):
        assert len(split[a]) == len(set(split[a])), f"duplicate record in {a}"
        for b in PARTS[i + 1:]:
            assert not set(split[a]) & set(split[b]), f"record overlap {a}/{b}"
            assert not people[a] & people[b], f"patient overlap {a}/{b}: {sorted(people[a] & people[b])}"
    ds2 = {f"mitdb/{r}" for r in DS2_RECORDS}
    assert set(split["test_ds2"]) == ds2, "test_ds2 is not exactly DS2"
    assert all(r.startswith("incartdb/") for r in split["test_incart"]), "external test holds a non-INCART record"
    dev = set(split["train"]) | set(split["validation"])
    assert not dev & ds2, "a DS2 record is in training or validation"
    dev_people = people["train"] | people["validation"]
    assert not {"mitdb:201", "mitdb:202"} & dev_people, "DS2 subject 201/202 is in training or validation"


def check_arrays(split: dict, data_records, data_patients, parts) -> None:
    """The data file must hold exactly the records of `parts`, with the patient labels the split declares."""
    want = {r for p in parts for r in split[p]}
    have = set(data_records.tolist())
    assert have == want, f"data file records differ from split parts {parts}: extra {sorted(have - want)[:5]}, missing {sorted(want - have)[:5]}"
    for r, p in set(zip(data_records.tolist(), data_patients.tolist())):
        assert split["patients"][r] == p, f"patient label mismatch for {r}"
