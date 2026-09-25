"""
LTR entries and the form of an LTR insertion: full, solo, internal, partial.
"""

import pytest

from placer.core import ltr_pairs as P
from placer.core.seqtools import parse_te_name_parts

pytestmark = pytest.mark.invariant


@pytest.mark.parametrize("name, internal, stem", [
    ("MER41-int", True, "MER41"), ("MLT1J-int", True, "MLT1J"),
    ("Gypsy-12_I", True, "Gypsy-12"), ("Gypsy-12_INT", True, "Gypsy-12"),
    ("Gypsy-12_LTR", False, "Gypsy-12"), ("MER41A", False, "MER41A"),
    ("MLT1J1", False, "MLT1J1"), ("Pintail", False, "Pintail"),
])
def test_internal_entries_are_recognised_by_suffix_only(name, internal, stem):
    assert P.is_internal_entry(name) is internal
    assert P.stem(name) == stem


class Hit:
    def __init__(self, name, start, end):
        self.name_parts = parse_te_name_parts(name + "#LTR/ERV1")
        self.query_start, self.query_end = start, end


def test_the_form_is_read_off_where_the_ltr_and_internal_hits_fall():
    full = [Hit("MER41A", 0, 500), Hit("MER41-int", 500, 5500), Hit("MER41B", 5500, 6000)]
    assert P.insert_form(full, 6000) == "full"
    assert P.insert_form([Hit("MER41A", 3, 495)], 500) == "solo"
    assert P.insert_form([Hit("MER41-int", 0, 3000)], 3000) == "internal"
    assert P.insert_form([Hit("MER41A", 0, 200)], 900) == "partial"
    assert P.insert_form([], 900) == "NA"
