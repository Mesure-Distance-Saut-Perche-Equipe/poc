import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


def segment_video(
    input_video="video_saut.mp4",
    output_video="instance-segmentation.avi",
    model_path="yolov8l-seg.pt",
    conf_threshold=0.8,
):
    # Load the segmentation model
    model = YOLO(model_path)
    names = model.model.names

    # Open the video capture
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the video writer for the output
    out = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame.")
            break

        # Perform prediction on the frame
        results = model.predict(frame, conf=conf_threshold)

        # Annotator for visualizing results
        annotator = Annotator(frame, line_width=2)

        # Process segmentation masks if available
        if results[0].masks is not None:
            classes = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy

            for mask, cls in zip(masks, classes):
                annotator.seg_bbox(
                    mask=mask, mask_color=colors(int(cls), True), label=names[int(cls)]
                )

        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame with segmentation
        cv2.imshow("Instance Segmentation", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
