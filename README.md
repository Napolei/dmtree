# D-MTree (Downward-MTree)

This library implements a datastructure very similar to the MTree datastructure in pure python (for more information, see https://en.wikipedia.org/wiki/M-tree)

The main difference is that the split of a NonLeafNode (inner node) will not propagate to the parent node, but instead 
creates a new subtree with the two closest routing objects as descendants. 
* This has the advantage that the covering-radii of the NonLeafNodes have the least overlap.
* The disadvantage is that if the points are added in an unlucky order, the depth of the tree can grow very fast!


## Usage

```python
def distance(p1, p2):
    return abs(p1 - p2)

dmtree = DMTree(distance_measure=distance, leaf_node_capacity=10, inner_node_capacity=5)

# inserting 1000 values with equal identifier:
for i in range(1000):
    dmtree.insert(identifier=i, value=i)     

# search in radius and order them by distance (ascending):
elements = dmtree.find_in_radius(value=50.5, radius=1)

# find the 10 nearest neighbours and order them by distance (ascending):
# the parameter 'truncate' decides if the results is trimmed to exactly k if there are multiple elements with the same distance
elements = dmtree.knn(value=50.5, k=10, truncate=True)

# deleting entries
for identifier in range(1000):
    dmtree.remove(identifier)
```
* identifier should be a datatype that can be compared with ==, <=, >=, <, >
* mtree.knn(value, k, truncate=True) is equal to mtree.knn(value, k, truncate=False)[:k]

## Parameters
| Parameter                 | Description                                                                           | 
| -------------:            |:-------------                                                                         | 
| distance_measure          | Custom distance method that fulfills the  triangle inequality                         | 
| leaf_node_capacity        | The number of children a NonLeafNode can hold before it is split (>=2) Default=50     |  
| inner_node_capacity       | The number of objects a LeafNode can hold before it is split  (>=2)  Default=20       |
| split_method              | The algorithm that splits overflowing LeafNodes Default='max_distance'                |    

## Split methods
| Parameter                 | Description                                                                           | 
| -------------:            |:-------------                                                                         | 
| random        | Choose two random elements of the LeafNode as new routing objects                                 | 
| min_sum_radii | Choose the two elements of the LeafNode that minimizes the sum of the resulting radii             |  
| max_distance  | Choose the two elements of the LeafNode that maximizes the distance between the routing values    |

## Best practice
* When inserting the values, do so in a randomized order, or the resulting tree can very quickly become heavily unbalanced