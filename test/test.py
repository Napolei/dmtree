import random

import pytest

from dmtree import DMTree, RangeElement, MTreeElement


def distance(p1, p2):
    return abs(p1 - p2)


def create_empty_mtree():
    return DMTree(distance_measure=distance, leaf_node_capacity=5, inner_node_capacity=3)


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
        for i in range(10):
            mtree = create_empty_mtree()
            for value in base_values():
                mtree.insert(value, value)
            assert len(mtree.values()) == len(base_values())
            mtree_values = [x.value for x in mtree.values()]
            assert set(mtree_values) == set(base_values())

    def test_batch_insert(self):
        mtree = create_empty_mtree()
        batch = [(x, x) for x in base_values()]
        mtree.insert_batch(batch)
        assert len(mtree.values()) == len(batch)

    def test_remove(self):
        mtree = create_full_mtree()

        for value in base_values():
            mtree.remove(value)
        assert len(mtree.values()) == 0

    def test_remove_batch(self):
        for i in range(10):
            mtree = create_full_mtree()
            identifiers = set([x for x in base_values()[:100]])
            assert len(identifiers) > 0
            mtree.remove_batch(identifiers)
            assert len(mtree.values()) == len(base_values()) - len(identifiers)
            mtree_identifiers = set([x.identifier for x in mtree.values()])
            for iden in identifiers:
                assert iden not in mtree_identifiers

    def test_inner_node_capacity_greater_than_2(self):
        with pytest.raises(ValueError) as _:
            DMTree(distance_measure=None, inner_node_capacity=1)

    def test_leaf_node_capacity_greater_than_2(self):
        with pytest.raises(ValueError) as _:
            DMTree(distance_measure=None, leaf_node_capacity=1)

    def test_knn_result_elements(self):
        mtree = create_full_mtree()
        nn = mtree.knn(value=50.5, k=2, truncate=True)
        assert len(nn) == 2
        expected_element_1 = RangeElement(MTreeElement(50, 50), 0.5)
        expected_element_2 = RangeElement(MTreeElement(51, 51), 0.5)
        assert expected_element_1 in nn
        assert expected_element_2 in nn
        unexpected_element_1 = RangeElement(MTreeElement(52, 50), 1.5)
        assert unexpected_element_1 not in nn

    def test_knn_truncate(self):
        mtree = create_full_mtree()
        nn = mtree.knn(value=50, k=2, truncate=True)
        assert len(nn) == 2
        nn = mtree.knn(value=50, k=2, truncate=False)
        assert len(nn) == 3

    def test_find_in_radius(self):
        mtree = create_full_mtree()
        results = mtree.find_in_radius(value=50.5, radius=1)
        assert len(results) == 2
        expected_element_1 = RangeElement(MTreeElement(50, 50), 0.5)
        expected_element_2 = RangeElement(MTreeElement(51, 51), 0.5)
        assert expected_element_1 in results
        assert expected_element_2 in results
        unexpected_element_1 = RangeElement(MTreeElement(52, 50), 1.5)
        assert unexpected_element_1 not in results

    def test_no_endless_loop_leafnode_elements_all_the_same(self):
        mtree = create_empty_mtree()
        for i in range(500):
            mtree.insert(i, 42)
