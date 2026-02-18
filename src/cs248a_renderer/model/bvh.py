import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import numpy as np
import slangpy as spy

from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.primitive import Primitive
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class BVHNode:
    # The bounding box of this node.
    bound: BoundingBox3D = field(default_factory=BoundingBox3D)
    # The index of the left child node, or -1 if this is a leaf node.
    left: int = -1
    # The index of the right child node, or -1 if this is a leaf node.
    right: int = -1
    # The starting index of the primitives in the primitives array.
    prim_left: int = 0
    # The ending index (exclusive) of the primitives in the primitives array.
    prim_right: int = 0
    # The depth of this node in the BVH tree.
    depth: int = 0

    def get_this(self) -> Dict:
        return {
            "bound": self.bound.get_this(),
            "left": self.left,
            "right": self.right,
            "primLeft": self.prim_left,
            "primRight": self.prim_right,
            "depth": self.depth,
        }

    @property
    def is_leaf(self) -> bool:
        """Checks if this node is a leaf node."""
        return self.left == -1 and self.right == -1


class BVH:
    def __init__(
        self,
        primitives: List[Primitive],
        max_nodes: int,
        min_prim_per_node: int = 1,
        num_thresholds: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Builds the BVH from the given list of primitives. The build algorithm should
        reorder the primitives in-place to align with the BVH node structure.
        The algorithm will start from the root node and recursively partition the primitives
        into child nodes until the maximum number of nodes is reached or the primitives
        cannot be further subdivided.
        At each node, the splitting axis and threshold should be chosen using the Surface Area Heuristic (SAH)
        to minimize the expected cost of traversing the BVH during ray intersection tests.

        :param primitives: the list of primitives to build the BVH from
        :type primitives: List[Primitive]
        :param max_nodes: the maximum number of nodes in the BVH
        :type max_nodes: int
        :param min_prim_per_node: the minimum number of primitives per leaf node
        :type min_prim_per_node: int
        :param num_thresholds: the number of thresholds per axis to consider when splitting
        :type num_thresholds: int
        """
        self.nodes: List[BVHNode] = []

        # TODO: Student implementation starts here.
        self.nodes = build_bvh(primitives, num_thresholds, min_prim_per_node, max_nodes)
        # TODO: Student implementation ends here.


def build_bvh(primitives, num_thresholds, min_prim_per_node, max_nodes):
    nodes = []

    # make a node
    node = make_node(primitives, 0, len(primitives), 0)
    nodes.append(node)


    # for each node in nodes, but less than max node, start branching out BFS algo style
    i = 0
    while i < len(nodes) and len(nodes) + 2 <= max_nodes:
        node = nodes[i]
        start, end = node.prim_left, node.prim_right
        num_prims = end - start

        # node is a leaf if min prims
        if num_prims <= min_prim_per_node:
            i += 1
            continue
        
        # use sah heuristic to find the best split on an axis and bucket
        axis, bucket = sah(primitives, start, end, node.bound, num_thresholds)
        if axis is None:
            i += 1
            continue

        # partition by the node
        axis_min = node.bound.min[axis]
        axis_max = node.bound.max[axis]
        threshold = axis_min + (axis_max - axis_min) * bucket / num_thresholds

        # primitive reordering in place in the array based on partitioning
        l, r = start, end - 1
        while l <= r:
            while l <= r and primitives[l].bounding_box.center[axis] < threshold:
                l += 1
            while l <= r and primitives[r].bounding_box.center[axis] >= threshold:
                r -= 1
            if l < r:
                primitives[l], primitives[r] = primitives[r], primitives[l]
                l += 1
                r -= 1

        mid = l
        if mid == start or mid == end:
            i += 1
            continue

        # recurse and find the children of the node based on the partitioning
        left_idx = len(nodes)
        nodes.append(make_node(primitives, start, mid, node.depth + 1))

        right_idx = len(nodes)
        nodes.append(make_node(primitives, mid, end, node.depth + 1))

        node.left = left_idx
        node.right = right_idx

        i += 1

    return nodes

def make_node(primitives, start, end, depth):
    node = BVHNode()
    node.prim_left = start
    node.prim_right = end
    node.bound = find_bounding_box(primitives, start, end)
    node.depth = depth
    node.left = -1
    node.right = -1
    return node

def sah(primitives, start, end, bb, num_thresholds):
    best_cost = float("inf")
    best_axis = None
    best_bucket = None

    parent_area = bb.area

    # edge case
    if parent_area == 0:
        return None, None
    
    # for each axis (x, y, z)
    for axis in range(3):
        # array of empty buckets, use a dict to represent one bucket with count and bounding box
        buckets = [{"count": 0, "bb": None} for _ in range(num_thresholds)]

        # for each primitive in list
        # compute where primitive goes in bucket
        # then add a count and primitive to the bucket and union bounding boxes within a bucket together
        for i in range(start, end):
            
            p = primitives[i]
            b = bucket_index(p.bounding_box.center[axis], bb, num_thresholds, axis)

            buckets[b]["count"] += 1
            if buckets[b]["bb"] is None:
                buckets[b]["bb"] = p.bounding_box
            else:
                buckets[b]["bb"] = BoundingBox3D.union(buckets[b]["bb"], p.bounding_box)

        # for each of the B-1 partitions planes evaluate SAH for the right and for the left
        # [0 ... split-1] and [split ... B-1]
        for split in range(1, num_thresholds): 
            left_bb = None
            right_bb = None
            left = 0
            right = 0

            # left
            for i in range(0, split):
                if buckets[i]["bb"] is None:
                    continue
                if left_bb is None:
                    left_bb = buckets[i]["bb"]
                else:
                    left_bb = BoundingBox3D.union(left_bb, buckets[i]["bb"])
                left += buckets[i]["count"]
            
            # right 
            for i in range(split, num_thresholds):
                if buckets[i]["bb"] is None:
                    continue
                if right_bb is None:
                    right_bb = buckets[i]["bb"]
                else:
                    right_bb = BoundingBox3D.union(right_bb, buckets[i]["bb"])
                right += buckets[i]["count"]

            if left_bb is None or right_bb is None:
                continue
            
            # compute SAH cost surface_left/surface_parent * num left + surface_right/surface_parent * num right
            cost = (left_bb.area / parent_area) * left + (right_bb.area / parent_area) * right

            # compare costs
            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_bucket = split

    return best_axis, best_bucket

def bucket_index(center, bb, num_thresholds, axis):
    # if bounding box has no size
    if bb.max[axis] == bb.min[axis]:
        return 0
    # normalize primitive on bounding box
    t = (center - bb.min[axis]) / (bb.max[axis] - bb.min[axis])
    # return bucket placement
    return min(num_thresholds - 1, max(0, int(t * num_thresholds)))

def find_bounding_box(primitives, start, end):
    # Compute the bounding box that encloses a set of primitives.
    bb = primitives[start].bounding_box
    for i in range(start + 1, end):
        bb = BoundingBox3D.union(bb, primitives[i].bounding_box)
    return bb

def create_bvh_node_buf(module: spy.Module, bvh_nodes: List[BVHNode]) -> spy.NDBuffer:
    device = module.device
    node_buf = spy.NDBuffer(
        device=device, dtype=module.BVHNode.as_struct(), shape=(max(len(bvh_nodes), 1),)
    )
    cursor = node_buf.cursor()
    for idx, node in enumerate(bvh_nodes):
        cursor[idx].write(node.get_this())
    cursor.apply()
    return node_buf
