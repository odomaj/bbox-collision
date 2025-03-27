import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class BoundingBox:
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self.xmin, self.ymin, self.zmin = xmin, ymin, zmin
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax

    def is_colliding(self, other):
        """Check if this bounding box collides with another."""
        return (
            self.xmin <= other.xmax
            and self.xmax >= other.xmin
            and self.ymin <= other.ymax
            and self.ymax >= other.ymin
            and self.zmin <= other.zmax
            and self.zmax >= other.zmin
        )


def generate_random_bounding_boxes(n, space_size=10):
    """Generate `n` random bounding boxes within the given space."""
    boxes = []
    for _ in range(n):
        x, y, z = np.random.uniform(0, space_size, 3)
        size = np.random.uniform(1, 3)
        boxes.append(BoundingBox(x, y, z, x + size, y + size, z + size))
    return boxes


""" TODO """


class BVHNode:
    """Node for the Bounding Volume Hierarchy (BVH) Tree."""

    pass


""" TODO """


def build_bvh(boxes):
    """Recursively build a BVH tree."""
    pass


""" TODO """


def detect_collisions_bvh(root):
    """Detect collisions using BVH by comparing all leaf nodes."""
    pass


def draw_bounding_box(ax, box, color):
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
        Poly3DCollection(faces, alpha=0.3, linewidths=1, edgecolors="k", facecolors=color)
    )


def visualize_bounding_boxes(boxes, collisions):
    """Plot bounding boxes and highlight colliding ones, with coordinates."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract all boxes involved in collisions
    collided_boxes = set()
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

    """ TODO stars """
    # Step1: Build BVH

    # Step2: Detect collisions using BVH
    collisions = set()  # call function to get this. Now, set it to empty for visualization
    """ TODO ends """

    # Print detected collisions
    print(f"Collisions Detected: {len(collisions)} pairs")
    for box1, box2 in collisions:
        print(
            f"Collision: Box({box1.xmin}, {box1.ymin}, {box1.zmin}) ↔ Box({box2.xmin},"
            f" {box2.ymin}, {box2.zmin})"
        )

    # Visualize the bounding boxes
    visualize_bounding_boxes(bounding_boxes, collisions)
