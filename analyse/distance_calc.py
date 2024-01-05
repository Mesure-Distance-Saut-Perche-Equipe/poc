import math


def get_distance_between_point_and_box(point, box):
    # Extracting the rectangle corner coordinates
    corner1 = (box[0], box[1])
    corner2 = (box[0], box[3])
    corner3 = (box[2], box[1])
    corner4 = (box[2], box[3])

    # Calculate distances from the given point to each corner
    distances = [
        math.sqrt((point['x'] - corner[0]) ** 2 + (point['y'] - corner[1]) ** 2)
        for corner in [corner1, corner2, corner3, corner4]
    ]

    # Find the minimum distance and corresponding corner
    min_distance, closest_corner = min((dist, corner) for dist, corner in zip(distances, [corner1, corner2, corner3, corner4]))

    # Find the minimum distance
    return round(min_distance, 2), closest_corner
