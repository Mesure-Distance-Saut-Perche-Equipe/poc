import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw
from analyse.image_processor import image_processor
from analyse.distance_calc import get_distance_between_point_and_box
from analyse.example_manager import example_manager
from ui.utils.utils import get_ellipse_coords, draw_distance_line


def main():
    initialize_session_state()

    st.title("Pole Vault Distance Measurement 🏋️‍")
    st.subheader(
        "Welcome to the project dedicated to measuring the distance to the bar for pole vault!"
    )

    st.subheader("To get started, upload an image")
    selected_image = get_selected_image()

    st.subheader("Or use the examples")
    option = st.selectbox(
        label_visibility="collapsed",
        label="Examples",
        options=example_manager.get_example_names(),
        index=None,
    )

    if selected_image is not None:
        st.title("Here is the image you've uploaded")
        value = streamlit_image_coordinates(selected_image, key="pil")
        st.markdown(
            "**Click on the image** to select the point that represents the bar. It will be used to calculate "
            "the distance.  \n"
            "You can click **multiple** times to change the location of the point: the distance will be "
            "recalculated automatically."
        )

        # The point was selected
        if value is not None:
            point = value["x"], value["y"]
            if point != st.session_state["point"]:
                st.session_state["point"] = point
                st.rerun()

            analyse(selected_image, point, value)

    elif option is not None:
        example = example_manager.get_example_by_name(option)
        image, point, value = display_example_image(example)
        analyse(image, point, value)


def initialize_session_state():
    if "point" not in st.session_state:
        st.session_state["point"] = (-10, -10)

    if "threshold" not in st.session_state:
        st.session_state.threshold = 90


def point_to_value(point):
    return {"x": point[0], "y": point[1]}


def draw_point_on_image(image, point):
    draw = ImageDraw.Draw(image)
    coords = get_ellipse_coords(point)
    draw.ellipse(coords, fill="red")


def get_selected_image():
    file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
    if file is not None:
        image = Image.open(file)
        image = image.resize((int(image.width / 1.7), int(image.height / 1.7)))

        # Draw an ellipse at each coordinate in point
        point = st.session_state["point"]
        draw_point_on_image(image, point)

        # Done
        return image
    return None


def display_example_image(example):
    image = Image.open(example["image"])
    image = image.resize((int(image.width / 1.7), int(image.height / 1.7)))

    value = example["value"]
    point = (value["x"], value["y"])
    draw_point_on_image(image, point)

    st.title("Here is the example you've selected")
    st.image(image)
    return image, point, value


def analyse(image, point, value):
    with st.spinner("Analysing..."):
        threshold = st.session_state.threshold / 100
        resulted_image, results = image_processor.detect_objects(
            image=image, draw_most_score=True, threshold=threshold
        )

        st.title("Here is the result of analyse")
        if len(results) == 0:
            st.subheader("No objects were detected")
        else:
            _max_score, _max_score_label, max_score_box = results[0]
            distance, closest_corner = get_distance_between_point_and_box(
                value, max_score_box
            )

            draw = ImageDraw.Draw(resulted_image)
            draw_distance_line(draw, point, closest_corner, distance)

            st.subheader("Processed image")
            st.image(
                resulted_image,
                use_column_width=True,
                caption="Processed image with distance line",
            )

            st.subheader("Calculated distance")
            st.markdown(f"Distance: **{distance}**")

            st.subheader("Detected objects")
            st.markdown(
                f"Only the object with the **maximum** score is used to calculate the distance."
            )
            for _, (score, label, box) in enumerate(results):
                st.markdown(
                    f"**{label}** with confidence **{round(score.item(), 3)}** at location **{box}**"
                )

            st.markdown(
                f"The threshold value is **{threshold}**.", help="..."
            )  # Your existing help content
            st.session_state.threshold = st.slider(
                label="Adjust threshold value", value=90, min_value=30, max_value=99
            )


if __name__ == "__main__":
    main()
