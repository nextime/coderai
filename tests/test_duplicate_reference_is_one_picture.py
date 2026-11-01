"""A source repeated in `images` is one picture, not two.

Reported 2026-11-01 from a Motus run: its combine step sends the photo as
``image`` and again as the first entry of ``images`` together with the second
picture, so two pictures arrived as three references and an edit within the
model's capacity was refused:

    'Krea-2-Edit' was sent 3 reference images ...

Counting a repeat is wrong whatever the model's capacity is, and it also made
the source get encoded twice.
"""
import codai.api.images as images


class _Req:
    def __init__(self, image=None, images_=None):
        self.image = image
        self.images = images_


def test_the_repeated_source_is_counted_once():
    r = _Req("A", ["A", "B"])
    assert images._distinct_reference_refs(r) == ["A", "B"]


def test_order_is_preserved_with_the_source_first():
    r = _Req("A", ["B", "C"])
    assert images._distinct_reference_refs(r) == ["A", "B", "C"]


def test_a_lone_image_is_one_reference():
    assert images._distinct_reference_refs(_Req("A")) == ["A"]
    assert images._distinct_reference_refs(_Req("A", [])) == ["A"]


def test_repeats_inside_images_are_also_collapsed():
    r = _Req("A", ["B", "B", "A"])
    assert images._distinct_reference_refs(r) == ["A", "B"]


def test_empty_entries_are_dropped_not_counted():
    r = _Req("A", ["", None, "B"])
    assert images._distinct_reference_refs(r) == ["A", "B"]


def test_the_edit_path_counts_and_decodes_the_distinct_list():
    import inspect
    src = inspect.getsource(images.create_image_edit)
    assert "_distinct_refs = _distinct_reference_refs(request)" in src
    assert "_n_refs = len(_distinct_refs)" in src, (
        "the capacity check must see the de-duplicated count"
    )
    assert "refs = _distinct_refs" in src, (
        "and the decode must use the same list, or the source is encoded twice"
    )
