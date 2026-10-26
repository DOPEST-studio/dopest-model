from dopest_model.reason_batch import Item, groups_of


def test_groups_are_sorted_longest_first_and_padded_to_the_batch_size():
    items = [Item("a", [1] * 5, 2), Item("b", [1] * 9, 3), Item("c", [1] * 7, 2), Item("d", [1] * 1, 4), Item("e", [1] * 3, 2)]
    groups = groups_of(items, 2)
    assert [[it.id for it in g] for g in groups] == [["b", "c"], ["a", "e"], ["d", "d"]]
    assert all(len(g) == 2 for g in groups)
    assert groups_of([], 4) == []
