def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
    center = point
    radius = 10
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


def draw_distance_line(draw, point, corner, distance):
    # Unpack point and corner tuples
    x1, y1 = point
    x2, y2 = corner

    # Draw the line
    draw.line([x1, y1, x2, y2], fill="blue", width=2)

    # Calculate the midpoint of the line
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Adjust the position to avoid overlapping with the line
    text_x = mid_x + 10  # Move the text to the right
    text_y = mid_y - 10  # Move the text up

    # Display the distance near the adjusted position with a bigger font
    draw.text((text_x, text_y), f"{distance:.2f}", fill="blue")
