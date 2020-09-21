import random

import pytest

from mtree import MTree


def distance(p1, p2):
    return abs(p1 - p2)


def create_empty_mtree():
    return MTree(distance_measure=distance, leaf_node_capacity=10, inner_node_capacity=5)


def base_values():
    values = list(range(1000))
    random.shuffle(values)
    return values


def create_full_mtree():
    mtree = create_empty_mtree()
    for value in base_values():
        mtree.insert(value, value)
    return mtree


class TestMTree:

    def test_insert(self):
        mtree = create_empty_mtree()
        for value in base_values():
            mtree.insert(value, value)
        assert len(mtree.values()) == len(base_values())
        mtree_values = [x.value for x in mtree.values()]
        assert set(mtree_values) == set(base_values())

    def test_remove_by_id(self):
        mtree = create_full_mtree()

        for value in base_values():
            mtree.remove_by_id(value)
        assert len(mtree.values()) == 0

    def test_inner_node_capacity_greater_than_2(self):
        with pytest.raises(ValueError) as _:
            MTree(distance_measure=None, inner_node_capacity=1)

    def test_leaf_node_capacity_greater_than_2(self):
        with pytest.raises(ValueError) as _:
            MTree(distance_measure=None, leaf_node_capacity=1)
