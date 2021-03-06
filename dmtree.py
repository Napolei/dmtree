import abc
import copy
import heapq
import itertools
import math
import operator
import random
from numbers import Number
from typing import Optional, Set, Dict, List, Iterator, Tuple, Any


class RangeElement:
    def __init__(self, element, r):
        self.element = element
        self.r = r

    def __eq__(self, other):
        if other is None or not isinstance(other, RangeElement):
            return False
        return self.element == other.element and self.r == other.r

    def __repr__(self):
        return f'RangeItem[value={self.element}, range={self.r}]'


class MTreeElement:
    def __init__(self, identifier, value):
        self.identifier = identifier
        self.value = value

    def __eq__(self, other):
        if other is None or not isinstance(other, MTreeElement):
            return False
        return self.identifier == other.identifier and self.value == other.value

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f'MtreeElement[id={self.identifier}, value={self.value}]'


class LeafNodeElement:
    # objects that belong to leaf nodes
    def __init__(self, identifier, value, distance_to_parent_routing_object, mtree_pointer):
        self.identifier = identifier
        self.value = value
        self.distance_to_parent_routing_object = distance_to_parent_routing_object
        self.mtree_pointer = mtree_pointer

    def __repr__(self):
        return f'{self.value}'


class HeapObject:
    next_id = 0

    def __init__(self, distance, obj):
        self.distance = distance
        self.unique_sorting_id = HeapObject.next_id
        self.obj = obj
        HeapObject.next_id += 1

    def heap_object(self):
        return self.distance, self.unique_sorting_id, self.obj


class Node:
    def __init__(self, mtree_pointer):
        self.mtree_pointer: DMTree = mtree_pointer
        self.distance_measure = mtree_pointer.distance_measure
        self.parent_node: Optional[NonLeafNode] = None  # holder of parent_routing_object

    def parent_routing_object(self):
        if self.parent_node is not None:
            candidates = [x for x in self.parent_node.routing_objects if x.covering_tree == self]
            assert len(candidates) == 1, f'length was {len(candidates)}, should be 1 out of {len(self.parent_node.routing_objects)}'
            return candidates[0]

    def is_root(self):
        return self == self.mtree_pointer._root_node

    @abc.abstractmethod
    def insert(self, mtree_object) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_full(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def values(self) -> Set[MTreeElement]:
        raise NotImplementedError()

    @abc.abstractmethod
    def leafnode_values(self) -> Set[MTreeElement]:
        raise NotImplementedError()

    @abc.abstractmethod
    def find_in_radius(self, value, radius, distance_to_parent) -> List[RangeElement]:
        raise NotImplementedError()

    @abc.abstractmethod
    def knn(self, value, distance_to_value, k, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        raise NotImplementedError()

    @abc.abstractmethod
    def remove(self, identifier):
        raise NotImplementedError()

    @abc.abstractmethod
    def height(self):
        raise NotImplementedError()


class NonLeafNode(Node):
    def __init__(self, mtree_pointer):
        super().__init__(mtree_pointer)
        self.routing_objects: Set[RoutingObject] = set()

    def is_full(self) -> bool:
        return self.mtree_pointer.inner_node_capacity < len(self.routing_objects)

    def insert(self, mtree_object: LeafNodeElement) -> None:

        candidates = sorted(
            [RangeElement(x, self.mtree_pointer.distance_measure(mtree_object.value, x.routing_value)) for x in
             self.routing_objects], key=lambda x: x.r)

        if len(candidates) == 0:
            # we are in the root node
            new_routing_object = RoutingObject(mtree_object.value, 0, None, self, self.mtree_pointer)
            self.insert_routing_object(new_routing_object)
            mtree_object = LeafNodeElement(mtree_object.identifier, mtree_object.value,
                                           self.mtree_pointer.distance_measure(mtree_object.value, mtree_object.value),
                                           self.mtree_pointer)
            new_routing_object.insert(mtree_object)
        else:
            # insert it into the closest candidate
            closest_routing_object: RoutingObject = candidates[0].element
            closest_routing_object_distance = candidates[0].r
            mtree_object.distance_to_parent_routing_object = closest_routing_object_distance
            closest_routing_object.insert(mtree_object)

    def _choose_new_routing_value_for_split_minimum_sum_distances(self, rv_1, rv_2, candidates: List[
        MTreeElement]):  # (self, rv_1: RoutingObject, rv_2: RoutingObject, candidates: List[MtreeElement]):
        minimum_sum = math.inf
        element = None
        for candidate in candidates:
            dist_1 = self.mtree_pointer.distance_measure(rv_1.routing_value, candidate.value)
            dist_2 = self.mtree_pointer.distance_measure(rv_2.routing_value, candidate.value)
            sum_dist = dist_1 + dist_2
            if sum_dist < minimum_sum:
                minimum_sum = sum_dist
                element = candidate
        return element

    def _choose_split_nodes_and_routing_object_value(self, ro_leaf_nodes):  # -> (LeafNode, LeafNode, MTreeObject)
        assert len(ro_leaf_nodes) >= 2
        # choose the two routing_objects that are closest together
        smallest_distance = math.inf
        c_1 = None
        c_2 = None
        for child_1 in ro_leaf_nodes:
            for child_2 in ro_leaf_nodes:
                if child_1 != child_2:
                    dist = self.mtree_pointer.distance_measure(child_1.routing_value, child_1.routing_value)
                    if dist < smallest_distance:
                        c_1 = child_1
                        c_2 = child_2

        all_children = list(c_1.values().union(c_2.values()))
        random.shuffle(all_children)
        new_routing_value = self._choose_new_routing_value_for_split_minimum_sum_distances(c_1, c_2, all_children)
        return c_1, c_2, new_routing_value

    def split_and_promote(self):
        candidates = [x for x in self.routing_objects if isinstance(x.covering_tree, LeafNode)]
        assert len(candidates) >= 2
        random.shuffle(candidates)
        leafnode_ro_1, leafnode_ro_2, rv = self._choose_split_nodes_and_routing_object_value(candidates)

        all_children = leafnode_ro_1.values().union(leafnode_ro_2.values())
        if len(all_children) == 0:
            self.remove_routing_object(leafnode_ro_1)
            # TODO optimization
        else:
            self.remove_routing_object(leafnode_ro_1)
            self.remove_routing_object(leafnode_ro_2)
            covering_radius = max([self.mtree_pointer.distance_measure(x.value, rv.value) for x in all_children])
            covering_tree = NonLeafNode(self.mtree_pointer)
            covering_tree.parent_node = self

            covering_tree.insert_routing_object(leafnode_ro_1)
            covering_tree.insert_routing_object(leafnode_ro_2)

            leafnode_ro_1.parent_node = self
            leafnode_ro_2.parent_node = self

            leafnode_ro_1.assigned_node = covering_tree
            leafnode_ro_2.assigned_node = covering_tree

            leafnode_ro_1.covering_tree.parent_node = covering_tree
            leafnode_ro_2.covering_tree.parent_node = covering_tree

            new_ro = RoutingObject(rv.value, covering_radius, covering_tree, self, self.mtree_pointer)
            self.insert_routing_object(new_ro)

    def insert_routing_object(self, new_routing_object):
        self.routing_objects.add(new_routing_object)
        new_routing_object.assigned_node = self
        if new_routing_object.covering_tree is not None:
            new_routing_object.covering_tree.parent_node = self
        if self.is_full():
            self.split_and_promote()

    def remove_routing_object(self, routing_object):
        self.routing_objects.remove(routing_object)

    def leafnode_values(self) -> Set[MTreeElement]:
        return set().union(*([x.leafnode_values() for x in self.routing_objects]))

    def values(self) -> Set[MTreeElement]:
        return set().union(*([x.values() for x in self.routing_objects]))

    def remove(self, identifier):
        for ro in list(self.routing_objects):
            ro.remove(identifier)

    def find_in_radius(self, value, radius, distance_to_parent) -> Iterator[RangeElement]:
        return itertools.chain.from_iterable([x.find_in_radius(value, radius, None) for x in self.routing_objects])

    def knn(self, value, distance_to_value, k, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        for ro in self.routing_objects:
            distance_to_value = self.distance_measure(value, ro.routing_value)
            lower_bound = max([0, distance_to_value - ro.covering_radius])
            heapq.heappush(heap, HeapObject(lower_bound, ro).heap_object())
        return heap, current_elements

    def do_fn(self, value, _, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        for ro in self.routing_objects:
            distance_to_value = self.distance_measure(value, ro.routing_value)
            upper_bound = max([0, distance_to_value + ro.covering_radius])
            new_heap_object = HeapObject(-1 * upper_bound, ro).heap_object()
            heapq.heappush(heap, new_heap_object)
        return heap, current_elements

    def height(self):
        candidates = [x.covering_tree for x in self.routing_objects if x.covering_tree is not None]
        if len(candidates) == 0:
            return 0
        return max([x.height() for x in candidates]) + 1

    def __repr__(self):
        return f'NonLeafNode[routing_objects={self.routing_objects}]'


class LeafNode(Node):
    def __init__(self, mtree_pointer):
        super().__init__(mtree_pointer)
        self.mtree_objects: Dict[str, LeafNodeElement] = {}
        self.radius = 0
        self.mtree_pointer._leaf_nodes.add(self)

    def update_radius(self):
        if len(self.mtree_objects) == 0:
            self.radius = 0
        else:
            self.radius = max([x.distance_to_parent_routing_object for x in self.mtree_objects.values()])

    def is_full(self) -> bool:
        # if the radius is 0, it means all objects are at the same position as the routing value, there is no split that will help us
        if self.radius <= self.mtree_pointer._leaf_nodes_minimum_radius_for_split:
            return False
        return self.mtree_pointer.leaf_node_capacity < len(self.mtree_objects.keys())

    def _assign_points_to_nearest_routing_object(self, ro_1: LeafNodeElement, ro_2: LeafNodeElement,
                                                 points: List[LeafNodeElement]) -> (
            List[LeafNodeElement], List[LeafNodeElement]):
        ro_1_children = []
        ro_2_children = []
        for point in points:
            dist_1 = self.mtree_pointer.distance_measure(ro_1.value, point.value)
            dist_2 = self.mtree_pointer.distance_measure(ro_2.value, point.value)
            if dist_1 <= dist_2:
                ro_1_children.append(LeafNodeElement(point.identifier, point.value, dist_1, self.mtree_pointer))
            else:
                ro_2_children.append(LeafNodeElement(point.identifier, point.value, dist_2, self.mtree_pointer))

        return ro_1_children, ro_2_children

    def _precompute_distances(self, candidates) -> Dict[Tuple[str, str], Number]:
        dist_cache: Dict[Tuple[str, str], Number] = {}
        for m1 in candidates:
            for m2 in candidates:
                if (m1.identifier, m2.identifier) not in dist_cache.keys():
                    dist = self.mtree_pointer.distance_measure(m1.value, m2.value)
                    dist_cache[(m1.identifier, m2.identifier)] = dist
                    dist_cache[(m2.identifier, m1.identifier)] = dist
        return dist_cache

    @staticmethod
    def _assign_values_to_closest_ro(ro_1, ro_2, candidates, dist_cache):
        list_1 = []
        list_2 = []
        for candidate in candidates:
            dist_1 = dist_cache[(ro_1.identifier, candidate.identifier)]
            dist_2 = dist_cache[(ro_2.identifier, candidate.identifier)]
            if dist_1 < dist_2:
                list_1.append(LeafNodeElement(candidate.identifier, candidate.value, dist_1, candidate.mtree_pointer))
            else:
                list_2.append(LeafNodeElement(candidate.identifier, candidate.value, dist_2, candidate.mtree_pointer))
        return list_1, list_2

    def _split_and_promote_max_distance(self) -> (
            LeafNodeElement, List[LeafNodeElement], LeafNodeElement, List[LeafNodeElement]):
        members = set(self.mtree_objects.values())

        id_mapper = {}
        for el in members:
            id_mapper[el.identifier] = el
        dist_cache = self._precompute_distances(members)
        identifier_ro_1, identifier_ro_2 = max(dist_cache.items(), key=operator.itemgetter(1))[0]
        ro_1 = id_mapper[identifier_ro_1]
        ro_2 = id_mapper[identifier_ro_2]

        children_1, children_2 = self._assign_values_to_closest_ro(ro_1, ro_2, members, dist_cache)
        return ro_1, children_1, ro_2, children_2

    def _split_and_promote_min_sum_radii(self) -> (
            LeafNodeElement, List[LeafNodeElement], LeafNodeElement, List[LeafNodeElement]):
        members = set(self.mtree_objects.values())

        dist_cache = self._precompute_distances(members)

        ro_1 = None
        ro_2 = None
        r_1 = math.inf
        r_2 = math.inf
        children_1 = []
        children_2 = []
        members_list = list(members)
        for i in range(len(members_list) - 1):
            for j in range(i + 1, len(members_list)):
                ro_candidate_1 = members_list[i]
                ro_candidate_2 = members_list[j]
                if ro_candidate_1.identifier != ro_candidate_2.identifier:
                    temp_children_1 = []
                    temp_children_2 = []
                    max_1 = 0
                    max_2 = 0
                    for member in members:
                        dist_1 = dist_cache[(ro_candidate_1.identifier, member.identifier)]
                        dist_2 = dist_cache[(ro_candidate_2.identifier, member.identifier)]
                        if dist_1 < dist_2:
                            temp_children_1.append(member)
                            max_1 = max(max_1, dist_1)
                        else:
                            temp_children_2.append(member)
                            max_2 = max(max_2, dist_2)
                    if max_1 + max_2 < r_1 + r_2:
                        ro_1 = ro_candidate_1
                        ro_2 = ro_candidate_2
                        r_1 = max_1
                        r_2 = max_2
                        children_1 = temp_children_1
                        children_2 = temp_children_2
        children_1 = [
            LeafNodeElement(x.identifier, x.value, dist_cache[(x.identifier, ro_1.identifier)], self.mtree_pointer) for
            x in children_1]
        children_2 = [
            LeafNodeElement(x.identifier, x.value, dist_cache[(x.identifier, ro_2.identifier)], self.mtree_pointer) for
            x in children_2]
        return ro_1, children_1, ro_2, children_2

    def _split_and_promote_random(self) -> (
            LeafNodeElement, List[LeafNodeElement], LeafNodeElement, List[LeafNodeElement]):
        members = list(self.mtree_objects.values())
        random.shuffle(members)
        assert len(members) >= 2
        ro_1 = members[0]
        ro_2 = members[1]
        ro_1_members, ro_2_members = self._assign_points_to_nearest_routing_object(ro_1, ro_2, members)
        return ro_1, ro_1_members, ro_2, ro_2_members

    def split_and_promote(self):
        if self.mtree_pointer.split_method == 'random':
            rv_1, ro_1_members, rv_2, ro_2_members = self._split_and_promote_random()
        elif self.mtree_pointer.split_method == 'min_sum_radii':
            rv_1, ro_1_members, rv_2, ro_2_members = self._split_and_promote_min_sum_radii()
        elif self.mtree_pointer.split_method == 'max_distance':
            rv_1, ro_1_members, rv_2, ro_2_members = self._split_and_promote_max_distance()
        else:
            raise Exception('Splitting method not supported')

        assert len(ro_1_members) + len(ro_2_members) == len(self.mtree_objects)
        assert self.parent_routing_object() in self.parent_node.routing_objects
        ro_1 = self._create_routing_object_from_points(rv_1, ro_1_members)
        ro_2 = self._create_routing_object_from_points(rv_2, ro_2_members)
        assert self.parent_routing_object() is not None
        self.parent_node.remove_routing_object(self.parent_routing_object())
        self.mtree_pointer._leaf_nodes.remove(self)
        self.parent_node.insert_routing_object(ro_1)
        self.parent_node.insert_routing_object(ro_2)
        assert ro_1.assigned_node is not None
        assert ro_2.assigned_node is not None

        if len(ro_1.covering_tree.mtree_objects)>0:
            assert ro_1 in [x for x in ro_1.assigned_node.routing_objects]
            assert ro_1.covering_tree.parent_node == self.parent_node or ro_1.covering_tree.parent_node.parent_node == self.parent_node
        if len(ro_2.covering_tree.mtree_objects) > 0:
            assert ro_2 in [x for x in ro_2.assigned_node.routing_objects]
            assert ro_2.covering_tree.parent_node == self.parent_node or ro_2.covering_tree.parent_node.parent_node == self.parent_node

    def _create_routing_object_from_points(self, routing_value: LeafNodeElement, points: List[LeafNodeElement]):

        new_leaf_node = LeafNode(self.mtree_pointer)
        new_leaf_node.parent_node = self.parent_node

        for element in points:
            new_leaf_node.do_insert(element, can_trigger_split=False)

        new_routing_object = RoutingObject(routing_value.value, new_leaf_node.radius, new_leaf_node, self.parent_node,
                                           self.mtree_pointer)

        new_routing_object.assigned_node = self.parent_node
        return new_routing_object

    def insert(self, mtree_object: LeafNodeElement) -> None:
        self.do_insert(mtree_object, True)

    def do_insert(self, mtree_object: LeafNodeElement, can_trigger_split=True) -> None:
        self.mtree_objects[mtree_object.identifier] = mtree_object
        if mtree_object.distance_to_parent_routing_object >= self.radius:
            self.radius = mtree_object.distance_to_parent_routing_object

        if can_trigger_split and self.is_full():
            self.split_and_promote()

    def remove(self, identifier):
        old_radius = self.radius
        self.mtree_objects.pop(identifier, None)
        if len(self.mtree_objects) == 0:
            pro = self.parent_routing_object()
            pro.covering_tree = None
            pro.covering_radius = 0
        else:
            self.update_radius()
            new_radius = self.radius
            if new_radius != old_radius:
                self.parent_routing_object().update_radius()

    def remove_batch(self, iterable: Set):
        old_radius = self.radius
        intersection = iterable.intersection(self.mtree_objects.keys())
        did_remove_elements = True
        for identifier in intersection:
            self.mtree_objects.pop(identifier)
            did_remove_elements = True
        if did_remove_elements:
            self.update_radius()
            new_radius = self.radius
            if new_radius != old_radius:
                self.parent_routing_object().update_radius()

    def remove_by_value(self, value):
        if value in self.mtree_objects.values():
            self.mtree_objects = {key: val for key, val in self.mtree_objects.items() if val != value}
            self.update_radius()

    def leafnode_values(self) -> Set[MTreeElement]:
        return self.values()

    def values(self) -> Set[MTreeElement]:
        return set([MTreeElement(x.identifier, x.value) for x in self.mtree_objects.values()])

    def _is_candidate_search_in_radius(self, search_radius, distance_to_routing_object, element):
        if search_radius < distance_to_routing_object:
            return element.distance_to_parent_routing_object >= distance_to_routing_object - search_radius
        return element.distance_to_parent_routing_object <= search_radius + distance_to_routing_object

    def find_in_radius(self, value, radius, distance_to_parent) -> List[RangeElement]:
        candidates = [x for x in self.mtree_objects.values() if
                      self._is_candidate_search_in_radius(radius, distance_to_parent, x)]
        ranges = [RangeElement(MTreeElement(x.identifier, x.value), self.distance_measure(x.value, value)) for x in
                  candidates]
        return [x for x in ranges if x.r <= radius]

    def knn(self, value, distance_to_value, k, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        for element in self.mtree_objects.values():
            distance = self.mtree_pointer.distance_measure(element.value, value)
            heapq.heappush(current_elements, HeapObject(distance,
                                                        RangeElement(MTreeElement(element.identifier, element.value),
                                                                     distance)).heap_object())
        return heap, current_elements

    def do_fn(self, value, _, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        for element in self.mtree_objects.values():
            distance = self.mtree_pointer.distance_measure(element.value, value)
            heapq.heappush(current_elements, HeapObject(-1 * distance, RangeElement(element, distance)).heap_object())
        return heap, current_elements

    def height(self):
        return 0

    def __repr__(self):
        return f'LeafNode[values={list(self.mtree_objects.values())}]'


class RoutingObject:
    def __init__(self, routing_value, covering_radius, covering_tree, assigned_node, mtree_pointer):
        self.routing_value = copy.copy(routing_value)
        self.assigned_node: Node = assigned_node
        self.covering_radius: Number = covering_radius
        self.covering_tree: Optional[Node] = covering_tree  # child node
        self.mtree_pointer = mtree_pointer

    def parent_node(self):
        return self.assigned_node.parent_node

    def update_radius(self):
        old_covering_radius = self.covering_radius
        parent_routing_object = self.assigned_node.parent_routing_object()
        furthest_child = self.fn(self.routing_value)
        if furthest_child is None:
            self.covering_radius = 0
            if parent_routing_object is not None:
                parent_routing_object.update_radius()
        else:
            new_covering_radius = furthest_child.r
            if new_covering_radius < old_covering_radius:
                self.covering_radius = new_covering_radius
                if parent_routing_object is not None:
                    parent_routing_object.update_radius()

    def insert(self, mtree_object):
        if self.covering_radius < mtree_object.distance_to_parent_routing_object:
            self.covering_radius = mtree_object.distance_to_parent_routing_object
        if self.covering_tree:
            self.covering_tree.insert(mtree_object)
        else:
            # routing object only has a routing value, but no subtree yet
            new_leafnode = LeafNode(self.mtree_pointer)
            new_leafnode.parent_node = self.assigned_node
            self.covering_tree = new_leafnode
            self.covering_tree.insert(mtree_object)
            assert new_leafnode in [x.covering_tree for x in new_leafnode.parent_node.routing_objects]

    def leafnode_values(self) -> Set[MTreeElement]:
        if self.covering_tree:
            return self.covering_tree.leafnode_values()
        else:
            return set()

    def values(self) -> Set[MTreeElement]:
        if self.covering_tree is not None:
            return self.covering_tree.values()
        else:
            return set()

    def remove(self, identifier):
        if self.covering_tree is not None:
            self.covering_tree.remove(identifier)

    def find_in_radius(self, value, radius, distance_to_parent) -> List[RangeElement]:
        # TODO optimize
        distance_to_routing_object_value = self.mtree_pointer.distance_measure(value, self.routing_value)

        if self.covering_tree is None:
            return []

        if distance_to_routing_object_value > radius + self.covering_radius:  # no overlap
            return []

        children_in_radius = self.covering_tree.find_in_radius(value, radius, distance_to_routing_object_value)
        return list(children_in_radius)

    def knn(self, value, _1, _2, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        # TODO optimize
        distance_to_value = self.mtree_pointer.distance_measure(value, self.routing_value)
        lower_bound = max([0, distance_to_value - self.covering_radius])

        if self.covering_tree is not None:
            heapq.heappush(heap, HeapObject(lower_bound, self.covering_tree).heap_object())
        return heap, current_elements

    @staticmethod
    def _continue_fn_search(k: int, heap: heapq, current_elements: heapq):
        if len(heap) == 0:
            return False
        if len(current_elements) < 1:
            return True
        upper_bound, _, highest_distance_node = heapq.nsmallest(1, heap)[0]
        upper_bound *= -1
        distance_element, _, highest_distance_element = heapq.nlargest(1, current_elements)[0]
        if distance_element <= upper_bound:
            return True
        else:
            return False

    def do_fn(self, value, _, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        if self.covering_tree is None:
            return heap, current_elements
        else:
            distance_to_value = self.mtree_pointer.distance_measure(value, self.routing_value)
            upper_bound = max([0, distance_to_value + self.covering_radius])
            heapq.heappush(heap, HeapObject(-1 * upper_bound, self.covering_tree).heap_object())
        return heap, current_elements

    def fn(self, value) -> Optional[RangeElement]:
        heap: heapq = []
        current_elements = []

        heap, current_elements = self.do_fn(value, math.inf, heap, current_elements)
        while self._continue_fn_search(1, heap, current_elements):
            _, _, next_candidate = heapq.heappop(heap)
            heap, current_elements = next_candidate.do_fn(value, math.inf, heap, current_elements)

        if len(current_elements) == 0:
            return None
        else:
            element = heapq.heappop(current_elements)
            return RangeElement(element[2].element, element[2].r)

    def __repr__(self):
        return f'RoutingObject[value={self.routing_value}, covering_tree={self.covering_tree}]'


class DMTree:
    _supported_split_methods = {'random', 'min_sum_radii', 'max_distance'}

    def __init__(self, distance_measure, inner_node_capacity=20, leaf_node_capacity=50, split_method='max_distance'):
        if inner_node_capacity < 2:
            raise ValueError(f'inner_node_capacity must be >=2, got {inner_node_capacity}')
        if leaf_node_capacity < 2:
            raise ValueError(f'leaf_node_capacity must be >=2, got {leaf_node_capacity}')
        self._leaf_nodes_minimum_radius_for_split = 0.00001
        self.inner_node_capacity = inner_node_capacity
        self.leaf_node_capacity = leaf_node_capacity
        self._root_node: Optional[Node] = None
        self.distance_measure = distance_measure
        self.split_method = split_method
        if split_method not in self._supported_split_methods:
            raise Exception(f'split method not supported, supported options are: {self._supported_split_methods}')
        self._leaf_nodes: Set[LeafNode] = set()
        self._values = {}

    def contains(self, identifier):
        for ln in self._leaf_nodes:
            if identifier in ln.mtree_objects.keys():
                return True
        return False

    def insert(self, identifier, obj):
        new_mtree_object = LeafNodeElement(identifier, obj, math.inf, self)
        if self.contains(identifier):
            return

        self._values[identifier] = MTreeElement(identifier, obj)
        if not self._root_node:
            # empty tree
            self._root_node = NonLeafNode(self)
            self._root_node.insert(new_mtree_object)
        else:
            self._root_node.insert(new_mtree_object)

    def insert_batch(self, iterable: (Any, Any)):
        temp = [x for x in iterable if not self.contains(x[0])]
        for identifier, value in sorted(temp, key=lambda _: random.random()):
            self.insert(identifier, value)

    def remove(self, identifier):
        self._values.pop(identifier, None)
        for leafnode in self._leaf_nodes:
            if identifier in leafnode.mtree_objects.keys():
                leafnode.remove(identifier)

    def remove_batch(self, iterable):
        for element in iterable:
            self._values.pop(element, None)
        for leafnode in self._leaf_nodes:
            leafnode.remove_batch(set(iterable))

    def values(self) -> Set[MTreeElement]:
        return self._values.values()

    def find_in_radius(self, value, radius) -> List[RangeElement]:
        if not self._root_node:
            return []
        return sorted(self._root_node.find_in_radius(value, radius, math.inf), key=lambda x: x.r)

    def height(self):
        if not self._root_node:
            return 0
        else:
            return self._root_node.height()

    @staticmethod
    def _continue_knn_search(k: int, heap: heapq, current_elements: heapq):
        if len(heap) == 0:
            return False
        if len(current_elements) < k:
            return True
        lower_bound, _, highest_distance_node = heapq.nsmallest(1, heap)[0]
        distance_element, _, highest_distance_element = heapq.nlargest(1, current_elements)[0]
        if distance_element >= lower_bound:
            return True
        else:
            return False

    def knn(self, value, k, truncate=True) -> List[RangeElement]:
        heap: heapq = []
        current_elements = []
        if not self._root_node:
            return current_elements

        heap, current_elements = self._root_node.knn(value, math.inf, k, heap, current_elements)
        while self._continue_knn_search(k, heap, current_elements):
            _, _, next_candidate = heapq.heappop(heap)
            heap, current_elements = next_candidate.knn(value, math.inf, k, heap, current_elements)

        return_elements = []
        max_distance = 0
        for index in range(len(current_elements)):
            element = heapq.heappop(current_elements)
            if index == k - 1:
                max_distance = element[0]
            if index <= k - 1:
                return_elements.append(element[2])
            if index > k - 1:
                if truncate:
                    return return_elements
                else:
                    if element[0] == max_distance:
                        return_elements.append(element[2])
                    else:
                        return return_elements
        return return_elements
