import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


class ImageROISelector:
    def __init__(self, image_name):
        self.image_name = image_name
        self.rgb_image = None
        self.roi = None

    def load_image(self):
        """Load an image using OpenCV and convert it to RGB format."""
        frame = cv2.imread(self.image_name)
        self.rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def select_roi(self, eclick, erelease):
        """Capture the selected ROI from the rectangle selector."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.roi = (x1, y1, x2, y2)
        print(f"Selected ROI: ({x1}, {y1}) to ({x2}, {y2})")

    def plot_image(self):
        """Plot the image and allow user to select a ROI."""
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.imshow(self.rgb_image)

        # Initialize the RectangleSelector
        _selector = RectangleSelector(
            ax,
            self.select_roi,
            useblit=True,
            button=[1, 3],  # disable the middle button
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        # Display the plot
        plt.show()

    def get_roi_image(self):
        """Return the ROI image if a valid ROI was selected."""
        if self.roi:
            x1, y1, x2, y2 = self.roi
            print(f"Selected ROI coordinates: {x1}, {y1}, {x2}, {y2}")
            return self.rgb_image[y1:y2, x1:x2]
        else:
            print("No ROI selected.")
            return None


def select(image_name):
    """Main function to execute the ROI selection and process the image."""
    roi_selector = ImageROISelector(image_name)
    roi_selector.load_image()
    roi_selector.plot_image()
    return roi_selector.get_roi_image()
