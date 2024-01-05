import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw
from analyse.image_processor import image_processor
from analyse.distance_calc import get_distance_between_point_and_box
from ui.utils.utils import get_ellipse_coords, draw_distance_line

# Initialise point
if "point" not in st.session_state:
    # Set -10, -10 as the default values to not display the point in the image
    st.session_state["point"] = (-10, -10)

# Initialise threshold
if "threshold" not in st.session_state:
    st.session_state.threshold = 90


def main():
    st.title('Pole Vault Distance Measurement 🏋️‍')
    st.subheader('Welcome to the project dedicated to the measuring distance to bar for pole vault')

    # Get file from user
    file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

    # File was uploaded
    if file is not None:
        st.title("Here is the image you've uploaded")
        image = Image.open(file)

        # Modify the size to display correctly with the lib
        image = image.resize((int(image.width / 1.7), int(image.height / 1.7))) # TODO: refactor

        # Draw an ellipse at each coordinate in point
        draw = ImageDraw.Draw(image)
        point = st.session_state["point"]
        coords = get_ellipse_coords(point)
        draw.ellipse(coords, fill="red")

        # Display the image with ability to get the location of click
        value = streamlit_image_coordinates(image, key="pil")
        st.markdown("**Click on the image** to select the point that represents the bar. It will be used to calculate "
                    "the distance.  \n"
                    "You can click **multiple** times to change the location of the point: the distance will be "
                    "recalculated automatically.")

        # The point was selected
        if value is not None:
            point = value["x"], value["y"]
            if point != st.session_state["point"]:
                st.session_state["point"] = point
                st.rerun()

            # Run image processing
            with st.spinner("Analysing..."):
                threshold = st.session_state.threshold / 100
                resulted_image, results = image_processor.detect_objects(
                    image=image,
                    draw_most_score=True,
                    threshold=threshold
                )

                # Show results
                st.title("Here is the result of analyse")

                # Nothing is detected
                if len(results) == 0:
                    st.subheader("No objects were detected")
                else:
                    # Calculate the distance between the selected point and the closest rect corner
                    _max_score, _max_score_label, max_score_box = results[0]
                    distance, closest_corner = get_distance_between_point_and_box(value, max_score_box)

                    # Draw the line between the corner and the point
                    draw = ImageDraw.Draw(resulted_image)
                    draw_distance_line(draw, point, closest_corner, distance)

                    # Display image with the detected objects
                    st.subheader("Processed image")
                    st.image(resulted_image, use_column_width=True, caption="Processed image with distance line")

                    # Display information about the distances
                    st.subheader("Calculated distance")
                    st.markdown(f"Distance: **{distance}**")

                    # Display information about the objects detection
                    st.subheader("Detected objects")
                    st.markdown(f"Only the object with the **maximum** score is used to calculate the distance.")
                    for _, (score, label, box) in enumerate(results):
                        st.markdown(f"**{label}** with confidence **{round(score.item(), 3)}** at location **{box}**")

                    # Threshold info and modification
                    st.markdown(f"The threshold value is **{threshold}**.",
                                help="The **threshold** is a confidence score used to filter out less reliable "
                                     "detections. Objects with confidence scores below the threshold may not be "
                                     "considered. Adjust the threshold to control the balance between sensitivity and "
                                     "precision."
                                )
                    st.session_state.threshold = st.slider(label="Adjust threshold value", value=90, min_value=30, max_value=99)


if __name__ == "__main__":
    main()
