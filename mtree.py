import abc
import heapq
import itertools
import math
from numbers import Number
from typing import Optional, Set, Dict, List, Iterator


class RangeItem:
    def __init__(self, obj, r):
        self.obj = obj
        self.r = r

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
        self.unique_sorting_id = id(self)
        self.obj = obj

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
    def find_in_radius(self, value, radius) -> List[RangeItem]:
        raise NotImplementedError()

    @abc.abstractmethod
    def knn(self, value, distance_to_value, k, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_by_id(self, identifier):
        raise NotImplementedError()

    # @abc.abstractmethod
    # def remove_by_value(self, value):
    #     raise NotImplementedError()


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
        else:
            # insert it into the closest candidate
            closest_routing_object: RoutingObject = candidates[0].obj
            closest_routing_object_distance = candidates[0].r
            mtree_object.distance_to_parent_routing_object = closest_routing_object_distance
            closest_routing_object.insert(mtree_object)
        pass

    def split_and_promote(self):
        # at least 2 of the routing objects need to point to leaf nodes!
        candidates = [x for x in self.routing_objects if isinstance(x.covering_tree, LeafNode)]
        assert len(candidates) >= 2
        candidates = sorted(candidates,
                            key=lambda x: len(x.covering_tree.mtree_objects.values()),
                            reverse=True)

        assert len(candidates) >= 2
        fullest_routing_object = candidates[0]
        fullest_leafnode = fullest_routing_object.covering_tree
        assert fullest_leafnode is not None
        assert isinstance(fullest_leafnode, LeafNode)
        fullest_leafnode: LeafNode = fullest_leafnode

        # choose a random object inside the leafnode
        identifier, new_routing_object_value = fullest_leafnode.mtree_objects.popitem()
        fullest_leafnode.update_radius()
        for mtree_obj in fullest_leafnode.mtree_objects.values():
            mtree_obj.distance_to_parent_routing_object = self.mtree_pointer.distance_measure(mtree_obj.value,
                                                                                              new_routing_object_value.value)
        fullest_leafnode.update_radius()

        fullest_routing_object_2 = candidates[1]
        subtree_2: LeafNode = fullest_routing_object_2.covering_tree
        assert subtree_2 is not None
        assert isinstance(subtree_2, LeafNode)
        subtree_2 = subtree_2
        for mtree_obj in subtree_2.mtree_objects.values():
            mtree_obj.distance_to_parent_routing_object = self.mtree_pointer.distance_measure(mtree_obj.value,
                                                                                              new_routing_object_value.value)
        subtree_2.update_radius()

        all_children = [x for x in fullest_leafnode.mtree_objects.values()] + [x for x in
                                                                               subtree_2.mtree_objects.values()]
        new_routing_object_radius = max(x.distance_to_parent_routing_object for x in all_children)
        new_covering_tree = NonLeafNode(self.mtree_pointer)
        new_covering_tree.routing_objects.add(fullest_routing_object)
        new_covering_tree.routing_objects.add(fullest_routing_object_2)
        new_covering_tree.parent_node = self

        fullest_leafnode.parent_node = new_covering_tree
        subtree_2.parent_node = new_covering_tree

        new_routing_object = RoutingObject(new_routing_object_value, new_routing_object_radius, new_covering_tree, self,
                                           self.mtree_pointer)
        new_covering_tree.parent_routing_object = new_routing_object

        # remove the two candidates from current node
        self.routing_objects.remove(fullest_routing_object)
        self.routing_objects.remove(fullest_routing_object_2)

        # add the new routing element
        self.insert_routing_object(new_routing_object)

    def insert_routing_object(self, new_routing_object):
        self.routing_objects.add(new_routing_object)
        new_routing_object.assigned_node = self

        if self.is_full():
            self.split_and_promote()

    def remove_routing_object(self, routing_object):
        self.routing_objects.discard(routing_object)

    def values(self) -> Set[MtreeElement]:
        return set().union(*([x.values() for x in self.routing_objects]))

    def remove_by_id(self, identifier):
        for ro in self.routing_objects:
            ro.remove_by_id(identifier)

    def find_in_radius(self, value, radius) -> Iterator[RangeItem]:
        return itertools.chain.from_iterable([x.find_in_radius(value, radius) for x in self.routing_objects])

    def knn(self, value, distance_to_value, k, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        # add routing objects to heap
        for ro in self.routing_objects:
            distance_to_value = self.distance_measure(value, ro.routing_value.value)
            lower_bound = max([0, distance_to_value - ro.covering_radius])
            heapq.heappush(heap, HeapObject(lower_bound, ro).heap_object())
        return heap, current_elements

    def __repr__(self):
        return f'NonLeafNode[routing_objects={self.routing_objects}]'


class LeafNode(Node):
    def __init__(self, mtree_pointer):
        super().__init__(mtree_pointer)
        self.mtree_objects: Dict[str, MTreeObject] = {}
        self.radius = 0

    def update_radius(self):
        if len(self.mtree_objects) == 0:
            self.radius = 0
        else:
            self.radius = max([x.distance_to_parent_routing_object for x in self.mtree_objects.values()])

    def is_full(self) -> bool:
        return self.mtree_pointer.leaf_node_capacity < len(self.mtree_objects.keys())

    def split_and_promote(self):
        # select a new point to create a new routing object
        # select the furthest point
        new_routing_object_value: MTreeObject = \
            sorted(self.mtree_objects.values(), key=lambda item: item.distance_to_parent_routing_object)[-1]

        self.remove_by_id(new_routing_object_value.identifier)
        # assign points that will move to that new routing object
        objects_to_move: Set[MTreeObject] = set()
        for element in self.mtree_objects.values():
            if element != new_routing_object_value:
                dist_to_new_routing_object_value = self.mtree_pointer.distance_measure(new_routing_object_value.value,
                                                                                       element.value)
                if dist_to_new_routing_object_value < element.distance_to_parent_routing_object:
                    element.distance_to_parent_routing_object = dist_to_new_routing_object_value
                    objects_to_move.add(element)

        new_routing_object_radius = 0 if len(objects_to_move) == 0 else max(
            [x.distance_to_parent_routing_object for x in objects_to_move])
        new_leaf_node = LeafNode(self.mtree_pointer)

        for element in objects_to_move:
            new_leaf_node.insert(element)
            self.mtree_objects.pop(
                element.identifier)  # remove objects from current node if they got assigned to the new leaf

        new_routing_object = RoutingObject(new_routing_object_value, new_routing_object_radius, new_leaf_node,
                                           self.parent_node, self.mtree_pointer)

        new_leaf_node.parent_node = self.parent_node
        new_leaf_node.parent_routing_object = new_routing_object
        # add the routing object to the parent node and split it if necessary

        # update the current leafnode radius
        self.radius = 0 if len(self.mtree_objects.values()) == 0 else max(
            [x.distance_to_parent_routing_object for x in self.mtree_objects.values()])

        self.parent_node.insert_routing_object(new_routing_object)

    def insert(self, mtree_object: MTreeObject) -> None:
        self.mtree_objects[mtree_object.identifier] = mtree_object
        if mtree_object.distance_to_parent_routing_object >= self.radius:
            self.radius = mtree_object.distance_to_parent_routing_object

        if self.is_full():
            self.split_and_promote()

    def remove_by_id(self, identifier):
        self.mtree_objects.pop(identifier, None)
        self.update_radius()

    def remove_by_value(self, value):
        if value in self.mtree_objects.values():
            self.mtree_objects = {key: val for key, val in self.mtree_objects.items() if val != value}
            self.update_radius()

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

    def __repr__(self):
        return f'LeafNode[values={list(self.mtree_objects.values())}]'


class RoutingObject:
    def __init__(self, routing_value, covering_radius, covering_tree, assigned_node, mtree_pointer):
        # self.obj = obj
        self.routing_value = routing_value
        self.assigned_node: Node = assigned_node
        self.covering_radius: Number = covering_radius
        self.covering_tree: Optional[Node] = covering_tree  # child node
        self.mtree_pointer = mtree_pointer

    def parent_node(self):
        return self.assigned_node.parent_node

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

    def values(self) -> Set[MtreeElement]:
        if self.covering_tree:
            return {MtreeElement(self.routing_value.identifier, self.routing_value.value)}.union(
                self.covering_tree.values())
        else:
            return {MtreeElement(self.routing_value.identifier, self.routing_value.value)}

    def remove_by_id(self, identifier):
        if self.covering_tree is None and self.routing_value.identifier == identifier:
            pn = self.parent_node()
            if pn is not None:
                self.parent_node().remove_routing_object(self)
            return
            # TODO potentially change all ancestor covering tree radii
        if self.routing_value.identifier != identifier and self.covering_tree is not None:
            self.covering_tree.remove_by_id(identifier)
            return
        else:
            # delete the routing_value
            # idea: select a child element, make it the new routing value, and re-insert all the children

            child_elements = [x for x in list(self.values()) if x.identifier != identifier]
            if len(child_elements) == 0:
                if self.parent_node() is not None:
                    self.parent_node().remove_routing_object(self)
                return
            self.routing_value = child_elements[0]
            self.covering_tree = None
            self.covering_radius = 0
            for child in child_elements[1:]:
                mtree_obj = MTreeObject(child.identifier, child.value,
                                        self.mtree_pointer.distance_measure(child.value, self.routing_value.value),
                                        self.mtree_pointer)
                self.insert(mtree_obj)

    def find_in_radius(self, value, radius) -> List[RangeItem]:
        distance_to_routing_object_value = self.mtree_pointer.distance_measure(value, self.routing_value.value)

        if distance_to_routing_object_value > radius + self.covering_radius:  # no overlap
            return []
        children_in_radius = self.covering_tree.find_in_radius(value, radius)
        if distance_to_routing_object_value <= radius:
            return list(children_in_radius) + [RangeItem(self.routing_value.value, distance_to_routing_object_value)]
        else:
            return children_in_radius

    def knn(self, value, _, k, heap: heapq, current_elements: heapq) -> (heapq, heapq):
        distance_to_value = self.mtree_pointer.distance_measure(value, self.routing_value.value)
        lower_bound = max([0, distance_to_value - self.covering_radius])
        heapq.heappush(current_elements,
                       HeapObject(distance_to_value, RangeItem(self.routing_value, distance_to_value)).heap_object())

        if self.covering_tree is not None:
            heapq.heappush(heap, HeapObject(lower_bound, self.covering_tree).heap_object())
        return heap, current_elements

    def __repr__(self):
        return f'RoutingObject[value={self.routing_value}, covering_tree={self.covering_tree}]'


class MTree:
    def __init__(self, distance_measure, inner_node_capacity=20, leaf_node_capacity=20):
        self.inner_node_capacity = inner_node_capacity
        self.leaf_node_capacity = leaf_node_capacity
        self._root_node: Optional[Node] = None
        self.distance_measure = distance_measure

    def insert(self, identifier, obj):
        new_mtree_object = MTreeObject(identifier, obj, math.inf, self)
        if not self._root_node:
            # empty tree
            self._root_node = NonLeafNode(self)
            self._root_node.insert(new_mtree_object)
        else:
            self._root_node.insert(new_mtree_object)

    def remove_by_id(self, identifier):
        if self._root_node is not None:
            self._root_node.remove_by_id(identifier)
        pass

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
        distance_element, _, highest_distance_element = heapq.nlargest(k, current_elements)[-1]
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
