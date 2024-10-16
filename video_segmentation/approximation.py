import cv2
import matplotlib.pyplot as plt


def approximate_bar(img, mask_barre):
    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask_barre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("No contours found.")
        return

    # Get image dimensions
    _rows, cols = img.shape[:2]

    # Use the first contour (assuming there's at least one)
    contour = contours[0]
    contour_length = len(contour)

    if contour_length < 4:
        print("Contour has insufficient points for line fitting.")
        return

    # Split the contour into two sections for line fitting
    section_len = contour_length // 4

    # Fit a line to the first quarter of the contour
    [vx1, vy1, x1, y1] = cv2.fitLine(
        contour[0:section_len], cv2.DIST_L2, 0, 0.001, 0.001
    )

    # Fit a line to the last quarter of the contour
    [vx2, vy2, x2, y2] = cv2.fitLine(
        contour[3 * section_len :], cv2.DIST_L2, 0, 0.001, 0.001
    )

    # Calculate the average slope (coefficient)
    slope_avg = (vy1 / vx1 + vy2 / vx2) / 2

    # Calculate the midpoint between the two line segments
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    # Calculate intersection points for drawing the line
    left_y = int((-x_mid * slope_avg) + y_mid)
    right_y = int(((cols - x_mid) * slope_avg) + y_mid)

    # Draw the approximated line on the image
    cv2.line(img, (cols - 1, right_y), (0, left_y), (0, 255, 0), 2)

    # Display the result
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
