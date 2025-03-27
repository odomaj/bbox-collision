from typing import Self
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


class BoundingBox:
    # cannot be a dataclass as they are not hashable?
    xmin: np.float64
    ymin: np.float64
    zmin: np.float64
    xmax: np.float64
    ymax: np.float64
    zmax: np.float64

    def __init__(
        self,
        xmin: np.float64,
        ymin: np.float64,
        zmin: np.float64,
        xmax: np.float64,
        ymax: np.float64,
        zmax: np.float64,
    ) -> None:
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = ymax

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


class BVHNode:
    """Node for the Bounding Volume Hierarchy (BVH) Tree."""

    box: BoundingBox
    left: Self | None
    right: Self | None

    def __init__(
        self,
        box: BoundingBox,
        left: Self | None = None,
        right: Self | None = None,
    ):
        self.box = box
        self.left = left
        self.right = right


def np_min(arg1: np.float64, arg2: np.float64):
    if arg1 > arg2:
        return arg2
    return arg1


def np_max(arg1: np.float64, arg2: np.float64):
    if arg1 < arg2:
        return arg2
    return arg1


def merge_bvh(left: BVHNode, right: BVHNode) -> BVHNode:
    return BVHNode(
        box=BoundingBox(
            xmin=np_min(left.box.xmin, right.box.xmin),
            xmax=np_max(left.box.xmax, right.box.xmax),
            ymin=np_min(left.box.ymin, right.box.ymin),
            ymax=np_max(left.box.ymax, right.box.ymax),
            zmin=np_min(left.box.zmin, right.box.zmin),
            zmax=np_max(left.box.zmax, right.box.zmax),
        ),
        left=left,
        right=right,
    )


def build_bvh(boxes) -> BVHNode | None:
    """Recursively build a BVH tree."""

    if len(boxes) == 0:
        return None

    if len(boxes) == 1:
        return BVHNode(box=boxes[0])

    mid: int = len(boxes) // 2
    left: BVHNode | None = build_bvh(boxes=boxes[:mid])
    right: BVHNode | None = build_bvh(boxes=boxes[mid:])
    if left == None:
        return right
    if right == None:
        return left
    return merge_bvh(left, right)


def detect_collisions_bvh(root: BVHNode) -> set:
    """Detect collisions using BVH by comparing all leaf nodes."""
    return set()


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


def visualize_bounding_boxes(boxes, collisions):
    """Plot bounding boxes and highlight colliding ones, with coordinates."""
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore
    # Extract all boxes involved in collisions
    collided_boxes: set = set()
    for box1, box2 in collisions:
        collided_boxes.add(box1)
        collided_boxes.add(box2)

    # Draw each bounding box with the correct color and add coordinate labels
    for box in boxes:
        color = "red" if box in collided_boxes else "blue"
        draw_bounding_box(ax, box, color)

    # Set axis labels
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    plt.title("Bounding Box Collision Visualization (BVH)")
    plt.show()


if __name__ == "__main__":
    num_meshes = 5
    bounding_boxes = generate_random_bounding_boxes(num_meshes)

    bvh: BVHNode | None = build_bvh(bounding_boxes)
    if bvh is None:
        exit()

    # Step2: Detect collisions using BVH
    collisions = detect_collisions_bvh(bvh)

    # Print detected collisions
    print(f"Collisions Detected: {len(collisions)} pairs")
    for box1, box2 in collisions:
        print(
            f"Collision: Box({box1.xmin}, {box1.ymin}, {box1.zmin}) ↔"
            f" Box({box2.xmin}, {box2.ymin}, {box2.zmin})"
        )

    # Visualize the bounding boxes
    visualize_bounding_boxes(bounding_boxes, collisions)
