import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backend_bases import MouseButton
import cv2

from segmentation import segment_video
from approximation import approximate_bar
from model import SegmentModel
from image_selector import select

IMG_NAME = "image_barre.png"

# Init segmentation model
segment_model = SegmentModel()
segment_model.load_model()
predictor = segment_model.get_predictor()


##################### SEGMENTATION DE LA BARRE ############################


def show_mask_only(mask, random_color=False):
    if random_color:
        color = np.random.random(3)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image * 255


def on_click(event):
    coords = []
    labels = []

    # assign global variable to access outside of function
    global fig, imgplot, img_in, bar_mask

    h, w = img_in.shape[:2]

    ix, iy = event.xdata, event.ydata

    if (
        ix > 1 and ix < w and iy > 1 and iy < h
    ):  # > 1 is a trick to avoid considering click on button

        if event.button is MouseButton.LEFT:
            labels.append(1)
        else:
            labels.append(0)

        coords.append((ix, iy))

        input_point = np.int16(coords)
        input_label = np.int16(labels)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask_image = show_mask_only(masks)
        bar_mask = masks[0]
        out = cv2.addWeighted(
            img_in.astype(np.float32), 0.7, mask_image.astype(np.float32), 0.3, 0.0
        )
        out = out.astype(np.uint8)

        for pt, lab in zip(input_point, input_label):
            x, y = pt
            if lab == 1:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.circle(out, (x, y), 3, color, -1)

        imgplot.set_data(out)
    fig.canvas.draw_idle()


############# selection of the region to process

# Run the main function with the image file name
img = select(IMG_NAME)

# processing of image ##############
predictor.set_image(img)

global imgplot, img_in, fig, bar_mask


############## selction of the bar with mouse #############
img_in = img.copy()

fig, ax = plt.subplots(figsize=(16, 9))

# Call click func
cid = fig.canvas.mpl_connect("button_press_event", on_click)
imgplot = plt.imshow(img_in)

axcolor = "lightgoldenrodyellow"
ax.margins(x=0)
# Create a matplotlib.widgets.Button to go to validate pole_selection.
validax = plt.axes([0.8, 0.025, 0.15, 0.04])
button_valid = Button(validax, "pole validation", color=axcolor, hovercolor="0.975")


def valid_store(event):
    global valid_and_store
    plt.close("all")


button_valid.on_clicked(valid_store)

# Create a matplotlib.widgets.Button to go reject mask_selection.
validax = plt.axes([0.2, 0.025, 0.15, 0.04])
button_reject = Button(validax, "reject", color=axcolor, hovercolor="0.975")


def reject(event):
    plt.close("all")


button_reject.on_clicked(reject)
plt.show()

################ contour of mask

mask_barre = img[..., 0].copy() * 0
mask_barre = np.where(bar_mask, 255, 0).astype(np.uint8)
np.save("mask_barre", mask_barre)

cnt_barre, _ = cv2.findContours(mask_barre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
epsilon = 0.01
cnt_barre = cv2.approxPolyDP(cnt_barre[0], epsilon, True)
cv2.drawContours(img, [cnt_barre], -1, (0, 0, 255), 3)

plt.imshow(img)
plt.show()


approximate_bar(img, mask_barre)

segment_video()
