import abc
import copy
import heapq
import itertools
import math
import random
from numbers import Number
from typing import Optional, Set, Dict, List, Iterator


class RangeItem:
    def __init__(self, obj, r):
        self.obj = obj
        self.r = r

    def __eq__(self, other):
        if other is None or not isinstance(other, RangeItem):
            return False
        return self.obj == other.obj and self.r == other.r

    def __repr__(self):
        return f'RangeItem[value={self.obj}, range={self.r}]'


class MtreeElement:
    def __init__(self, identifier, value):
        self.identifier = identifier
        self.value = value

    def __repr__(self):
        return f'MtreeElement[id={self.identifier}, value={self.value}]'


class MTreeObject:
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
        self.mtree_pointer: MTree = mtree_pointer
        self.distance_measure = mtree_pointer.distance_measure
        self.parent_routing_object: Optional[RoutingObject] = None
        self.parent_node: Optional[NonLeafNode] = None  # holder of parent_routing_object

    def is_root(self):
        return self == self.mtree_pointer._root_node

    @abc.abstractmethod
    def insert(self, mtree_object) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_full(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def values(self) -> Set[MtreeElement]:
        raise NotImplementedError()

    @abc.abstractmethod
    def leafnode_values(self) -> Set[MtreeElement]:
        raise NotImplementedError()

    @abc.abstractmethod
    def find_in_radius(self, value, radius) -> List[RangeItem]:
        raise NotImplementedError()

    @abc.abstractmethod
    def knn(self, value, distance_to_value, k, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_by_id(self, identifier):
        raise NotImplementedError()


class NonLeafNode(Node):
    def __init__(self, mtree_pointer):
        super().__init__(mtree_pointer)
        self.routing_objects: Set[RoutingObject] = set()

    def is_full(self) -> bool:
        return self.mtree_pointer.inner_node_capacity < len(self.routing_objects)

    def insert(self, mtree_object: MTreeObject) -> None:
        candidates = sorted(
            [RangeItem(x, self.mtree_pointer.distance_measure(mtree_object.value, x.routing_value.value)) for x in
             self.routing_objects], key=lambda x: x.r)

        if len(candidates) == 0:
            # we are in the root node
            new_routing_object = RoutingObject(mtree_object, 0, None, self, self.mtree_pointer)
            self.insert_routing_object(new_routing_object)
            new_routing_object.insert(mtree_object)
        else:
            # insert it into the closest candidate
            closest_routing_object: RoutingObject = candidates[0].obj
            closest_routing_object_distance = candidates[0].r
            mtree_object.distance_to_parent_routing_object = closest_routing_object_distance
            closest_routing_object.insert(mtree_object)

    def _choose_new_routing_value_for_split_minimum_sum_distances(self, rv_1, rv_2, candidates: List[
        MtreeElement]):  # (self, rv_1: RoutingObject, rv_2: RoutingObject, candidates: List[MtreeElement]):
        minimum_sum = math.inf
        element = None
        for candidate in candidates:
            dist_1 = self.mtree_pointer.distance_measure(rv_1.routing_value.value, candidate.value)
            dist_2 = self.mtree_pointer.distance_measure(rv_2.routing_value.value, candidate.value)
            sum_dist = dist_1 + dist_2
            if sum_dist < minimum_sum:
                minimum_sum = sum_dist
                element = candidate
        return element

    def _choose_split_nodes_and_routing_object_value(self, ro_leaf_nodes):  # -> (LeafNode, LeafNode, MTreeObject)
        # assert type(ro_leaf_nodes, List[RoutingObject])
        assert len(ro_leaf_nodes) >= 2
        # choose the two routing_objects that are closest together
        smallest_distance = math.inf
        c_1 = None
        c_2 = None
        for child_1 in ro_leaf_nodes:
            for child_2 in ro_leaf_nodes:
                if child_1 != child_2:
                    dist = self.mtree_pointer.distance_measure(child_1.routing_value.value, child_1.routing_value.value)
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
        ro_1, ro_2, rv = self._choose_split_nodes_and_routing_object_value(candidates)
        self.remove_routing_object(ro_1)
        self.remove_routing_object(ro_2)

        all_children = ro_1.values().union(ro_2.values())
        covering_radius = max([self.mtree_pointer.distance_measure(x.value, rv.value) for x in all_children])
        covering_tree = NonLeafNode(self.mtree_pointer)
        covering_tree.insert_routing_object(ro_1)
        covering_tree.insert_routing_object(ro_2)

        ro_1.parent_node = self
        ro_2.parent_node = self

        ro_1.assigned_node = covering_tree
        ro_2.assigned_node = covering_tree

        ro_1.covering_tree.parent_node = covering_tree
        ro_2.covering_tree.parent_node = covering_tree

        covering_tree.parent_node = self

        new_ro = RoutingObject(rv, covering_radius, covering_tree, self, self.mtree_pointer)
        ro_1.parent_routing_object = new_ro
        ro_2.parent_routing_object = new_ro
        self.insert_routing_object(new_ro)

    def insert_routing_object(self, new_routing_object):
        self.routing_objects.add(new_routing_object)
        new_routing_object.assigned_node = self

        if self.is_full():
            self.split_and_promote()

    def remove_routing_object(self, routing_object):
        self.routing_objects.discard(routing_object)

    def leafnode_values(self) -> Set[MtreeElement]:
        return set().union(*([x.leafnode_values() for x in self.routing_objects]))

    def values(self) -> Set[MtreeElement]:
        return set().union(*([x.values() for x in self.routing_objects]))

    def remove_by_id(self, identifier):
        for ro in list(self.routing_objects):
            ro.remove_by_id(identifier)

    def find_in_radius(self, value, radius) -> Iterator[RangeItem]:
        return itertools.chain.from_iterable([x.find_in_radius(value, radius) for x in self.routing_objects])

    def knn(self, value, distance_to_value, k, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        for ro in self.routing_objects:
            distance_to_value = self.distance_measure(value, ro.routing_value.value)
            lower_bound = max([0, distance_to_value - ro.covering_radius])
            heapq.heappush(heap, HeapObject(lower_bound, ro).heap_object())
        return heap, current_elements

    def do_fn(self, value, _, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        for ro in self.routing_objects:
            distance_to_value = self.distance_measure(value, ro.routing_value.value)
            upper_bound = max([0, distance_to_value + ro.covering_radius])
            new_heap_object = HeapObject(-1 * upper_bound, ro).heap_object()
            heapq.heappush(heap, new_heap_object)
        return heap, current_elements

    def __repr__(self):
        return f'NonLeafNode[routing_objects={self.routing_objects}]'


class LeafNode(Node):
    def __init__(self, mtree_pointer):
        super().__init__(mtree_pointer)
        self.mtree_objects: Dict[str, MTreeObject] = {}
        self.radius = 0
        self.mtree_pointer._leaf_nodes.add(self)

    def update_radius(self):
        if len(self.mtree_objects) == 0:
            self.radius = 0
        else:
            self.radius = max([x.distance_to_parent_routing_object for x in self.mtree_objects.values()])

    def is_full(self) -> bool:
        return self.mtree_pointer.leaf_node_capacity < len(self.mtree_objects.keys())

    def _assign_points_to_nearest_routing_object(self, ro_1: MTreeObject, ro_2: MTreeObject,
                                                 points: List[MTreeObject]) -> (List[MTreeObject], List[MTreeObject]):
        ro_1_children = []
        ro_2_children = []
        for point in points:
            dist_1 = self.mtree_pointer.distance_measure(ro_1.value, point.value)
            dist_2 = self.mtree_pointer.distance_measure(ro_2.value, point.value)
            if dist_1 <= dist_2:
                ro_1_children.append(MTreeObject(point.identifier, point.value, dist_1, self.mtree_pointer))
            else:
                ro_2_children.append(MTreeObject(point.identifier, point.value, dist_2, self.mtree_pointer))

        return ro_1_children, ro_2_children

    def _split_and_promote_random(self) -> (MTreeObject, List[MTreeObject], MTreeObject, List[MTreeObject]):
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
        else:
            raise Exception('Splitting method not supported')

        ro_1 = self._create_routing_object_from_points(rv_1, ro_1_members)
        ro_2 = self._create_routing_object_from_points(rv_2, ro_2_members)
        self.parent_node.remove_routing_object(self.parent_routing_object)
        self.parent_node.insert_routing_object(ro_1)
        self.parent_node.insert_routing_object(ro_2)

    def _create_routing_object_from_points(self, routing_value: MTreeObject, points: List[MTreeObject]):

        new_leaf_node = LeafNode(self.mtree_pointer)

        for element in points:
            new_leaf_node.insert(element)

        new_routing_object = RoutingObject(routing_value, new_leaf_node.radius, new_leaf_node, self.parent_node,
                                           self.mtree_pointer)

        new_leaf_node.parent_node = self.parent_node
        new_leaf_node.parent_routing_object = new_routing_object
        return new_routing_object

    def insert(self, mtree_object: MTreeObject) -> None:
        self.mtree_objects[mtree_object.identifier] = mtree_object
        if mtree_object.distance_to_parent_routing_object >= self.radius:
            self.radius = mtree_object.distance_to_parent_routing_object

        if self.is_full():
            self.split_and_promote()

    def remove_by_id(self, identifier):
        old_radius = self.radius
        self.mtree_objects.pop(identifier, None)
        if len(self.mtree_objects) == 0:
            self.parent_routing_object.covering_tree = None
            self.parent_routing_object.covering_radius = 0
        else:
            self.update_radius()
            new_radius = self.radius
            if new_radius != old_radius:
                self.parent_routing_object.update_radius()

    def remove_by_value(self, value):
        if value in self.mtree_objects.values():
            self.mtree_objects = {key: val for key, val in self.mtree_objects.items() if val != value}
            self.update_radius()

    def leafnode_values(self) -> Set[MtreeElement]:
        return self.values()

    def values(self) -> Set[MtreeElement]:
        return set([MtreeElement(x.identifier, x.value) for x in self.mtree_objects.values()])

    def find_in_radius(self, value, radius) -> List[RangeItem]:
        ranges = [RangeItem(x.value, self.distance_measure(x.value, value)) for x in self.mtree_objects.values()]
        return [x for x in ranges if x.r <= radius]

    def knn(self, value, distance_to_value, k, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        for element in self.mtree_objects.values():
            distance = self.mtree_pointer.distance_measure(element.value, value)
            heapq.heappush(current_elements, HeapObject(distance, RangeItem(element, distance)).heap_object())
        return heap, current_elements

    def do_fn(self, value, _, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        for element in self.mtree_objects.values():
            distance = self.mtree_pointer.distance_measure(element.value, value)
            heapq.heappush(current_elements, HeapObject(-1 * distance, RangeItem(element, distance)).heap_object())
        return heap, current_elements

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
        parent_routing_object = self.assigned_node.parent_routing_object
        furthest_child = self.fn(self.routing_value.value)
        if furthest_child is None:
            self.covering_radius = 0
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
            new_leafnode.parent_routing_object = self
            new_leafnode.parent_node = self.assigned_node
            self.covering_tree = new_leafnode
            self.covering_tree.insert(mtree_object)

    def leafnode_values(self) -> Set[MtreeElement]:
        if self.covering_tree:
            return self.covering_tree.leafnode_values()
        else:
            return set()

    def values(self) -> Set[MtreeElement]:
        if self.covering_tree is not None:
            return self.covering_tree.values()
        else:
            return set()

    def remove_by_id(self, identifier):
        if self.covering_tree is not None:
            self.covering_tree.remove_by_id(identifier)

    def find_in_radius(self, value, radius) -> List[RangeItem]:
        # TODO optimize
        distance_to_routing_object_value = self.mtree_pointer.distance_measure(value, self.routing_value.value)

        if self.covering_tree is None:
            return []

        if distance_to_routing_object_value > radius + self.covering_radius:  # no overlap
            return []

        children_in_radius = self.covering_tree.find_in_radius(value, radius)
        return list(children_in_radius)

    def knn(self, value, _1, _2, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        # TODO optimize
        distance_to_value = self.mtree_pointer.distance_measure(value, self.routing_value.value)
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
            distance_to_value = self.mtree_pointer.distance_measure(value, self.routing_value.value)
            upper_bound = max([0, distance_to_value + self.covering_radius])
            heapq.heappush(heap, HeapObject(-1 * upper_bound, self.covering_tree).heap_object())
        return heap, current_elements

    def fn(self, value) -> Optional[RangeItem]:
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
            return RangeItem(element[2].obj, element[2].r)

    def __repr__(self):
        return f'RoutingObject[value={self.routing_value}, covering_tree={self.covering_tree}]'


class MTree:
    def __init__(self, distance_measure, inner_node_capacity=20, leaf_node_capacity=20):
        if inner_node_capacity < 2:
            raise ValueError(f'inner_node_capacity must be >=2, got {inner_node_capacity}')
        if leaf_node_capacity < 2:
            raise ValueError(f'leaf_node_capacity must be >=2, got {leaf_node_capacity}')
        self.inner_node_capacity = inner_node_capacity
        self.leaf_node_capacity = leaf_node_capacity
        self._root_node: Optional[Node] = None
        self.distance_measure = distance_measure
        self.split_method = 'random'
        self._leaf_nodes = set()

    def insert(self, identifier, obj):
        new_mtree_object = MTreeObject(identifier, obj, math.inf, self)
        if not self._root_node:
            # empty tree
            self._root_node = NonLeafNode(self)
            self._root_node.insert(new_mtree_object)
        else:
            self._root_node.insert(new_mtree_object)

    def remove_by_id(self, identifier):
        for leafnode in self._leaf_nodes:
            if identifier in leafnode.mtree_objects.keys():
                leafnode.remove_by_id(identifier)

    def values(self) -> Set[MtreeElement]:
        if not self._root_node:
            return set()
        else:
            return self._root_node.values()

    def find_in_radius(self, value, radius) -> List[RangeItem]:
        if not self._root_node:
            return []
        return sorted(self._root_node.find_in_radius(value, radius), key=lambda x: x.r)

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

    def knn(self, value, k, truncate=True) -> List[RangeItem]:
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
