from dataclasses import dataclass
from typing import Self
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class BoundingBox:
    # cannot be a dataclass as they are not hashable?
    xmin: np.float64
    ymin: np.float64
    zmin: np.float64
    xmax: np.float64
    ymax: np.float64
    zmax: np.float64

    def is_colliding(self, other: Self):
        """Check if this bounding box collides with another."""
        return (
            self.xmin <= other.xmax
            and self.xmax >= other.xmin
            and self.ymin <= other.ymax
            and self.ymax >= other.ymin
            and self.zmin <= other.zmax
            and self.zmax >= other.zmin
        )


def generate_random_bounding_boxes(
    n, space_size: int = 10
) -> list[BoundingBox]:
    """Generate `n` random bounding boxes within the given space."""
    boxes = []
    for _ in range(n):
        x, y, z = np.random.uniform(0, space_size, 3)
        size = np.random.uniform(1, 3)
        boxes.append(BoundingBox(x, y, z, x + size, y + size, z + size))
    return boxes


def draw_bounding_box(ax: Axes3D, box: BoundingBox, color):
    """Draw a 3D bounding box."""
    vertices = np.array(
        [
            [box.xmin, box.ymin, box.zmin],
            [box.xmax, box.ymin, box.zmin],
            [box.xmax, box.ymax, box.zmin],
            [box.xmin, box.ymax, box.zmin],
            [box.xmin, box.ymin, box.zmax],
            [box.xmax, box.ymin, box.zmax],
            [box.xmax, box.ymax, box.zmax],
            [box.xmin, box.ymax, box.zmax],
        ]
    )

    faces = [
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]],
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [2, 3, 7, 6]],
        [vertices[j] for j in [1, 2, 6, 5]],
        [vertices[j] for j in [4, 7, 3, 0]],
    ]

    ax.add_collection3d(
        Poly3DCollection(
            faces, alpha=0.3, linewidths=1, edgecolors="k", facecolors=color
        )
    )


class NodeBVH:
    """Node for the Bounding Volume Hierarchy (BVH) Tree."""

    box_index: int
    left_index: int | None
    right_index: int | None

    def __init__(
        self,
        box_index: int,
        left: Self | None = None,
        right: Self | None = None,
    ):
        self.box_index = box_index
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        return self.right is None and self.left is None


class TreeBVH:
    boxes: list[BoundingBox]
    leaf_count: int
    root: NodeBVH | None

    def __init__(self, boxes: list[BoundingBox]):
        self.boxes = boxes
        self.leaf_count = len(boxes)
        self.gen()

    def min(self, arg1: np.float64, arg2: np.float64):
        if arg1 > arg2:
            return arg2
        return arg1

    def max(self, arg1: np.float64, arg2: np.float64):
        if arg1 < arg2:
            return arg2
        return arg1

    def recurse_gen(self, start_i: int, stop_i: int) -> NodeBVH:
        length = stop_i - start_i
        if length == 1:
            return NodeBVH(box_index=start_i)
        mid = start_i + length // 2
        return self.merge(
            self.recurse_gen(start_i, mid), self.recurse_gen(mid, stop_i)
        )

    def merge(self, left: NodeBVH, right: NodeBVH) -> NodeBVH:
        box_i: int = len(self.boxes)
        self.boxes.append(
            BoundingBox(
                xmin=self.min(
                    self.boxes[left.box_index].xmin,
                    self.boxes[right.box_index].xmin,
                ),
                xmax=self.max(
                    self.boxes[left.box_index].xmax,
                    self.boxes[right.box_index].xmax,
                ),
                ymin=self.min(
                    self.boxes[left.box_index].ymin,
                    self.boxes[right.box_index].ymin,
                ),
                ymax=self.max(
                    self.boxes[left.box_index].ymax,
                    self.boxes[right.box_index].ymax,
                ),
                zmin=self.min(
                    self.boxes[left.box_index].zmin,
                    self.boxes[right.box_index].zmin,
                ),
                zmax=self.max(
                    self.boxes[left.box_index].zmax,
                    self.boxes[right.box_index].zmax,
                ),
            )
        )
        return NodeBVH(box_index=box_i, left=left, right=right)

    def gen(self) -> None:
        if len(self.boxes) == 0:
            self.root = None
            return
        self.root = self.recurse_gen(0, self.leaf_count)

    def collide_r(self, node: NodeBVH, index: int) -> bool:
        if node.box_index == index:
            return False
        if node.box_index < self.leaf_count:
            return bool(
                self.boxes[node.box_index].is_colliding(self.boxes[index])
            )
        if not self.boxes[node.box_index].is_colliding(self.boxes[index]):
            return False
        if node.left is None or node.right is None:
            print("serious error")
            exit(-1)
        return self.collide_r(node.left, index) or self.collide_r(
            node.right, index
        )

    def collisions(self) -> set[int]:
        if self.root == None:
            return set()
        collisions = set()

        for i in range(self.leaf_count):
            if self.collide_r(self.root, i):
                collisions.add(i)

        print(f"Collisions Detected: {len(collisions)} pairs")
        for box_i in collisions:
            print(
                f"Collision: Box({self.boxes[box_i].xmin},"
                f" {self.boxes[box_i].ymin}, {self.boxes[box_i].zmin})"
            )
        return collisions

    def visualize(self) -> None:
        """Plot bounding boxes and highlight colliding ones, with coordinates."""
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore
        # Extract all boxes involved in collisions
        collisions = self.collisions()

        # Draw each bounding box with the correct color and add coordinate labels
        for box_i in range(self.leaf_count):
            color = "red" if box_i in collisions else "blue"
            draw_bounding_box(ax, self.boxes[box_i], color)

        # Set axis labels
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")

        plt.title("Bounding Box Collision Visualization (BVH)")
        plt.show()

    def dump_tree(self) -> None:
        print(self.leaf_count, len(self.boxes))


if __name__ == "__main__":
    num_meshes = 5
    bounding_boxes = generate_random_bounding_boxes(num_meshes)

    bvh = TreeBVH(boxes=bounding_boxes)
    bvh.dump_tree()
    bvh.visualize()
