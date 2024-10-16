import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import Button
from matplotlib.backend_bases import MouseButton
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


model_path = ""
img_name = 'image_barre.png'

##################### SEGMENTATION DE LA BARRE ############################

def show_mask_only(mask, random_color=False):
    if random_color:
        color = np.random.random(3)
    else:
        color = np.array([30/255, 144/255, 255/255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image * 255


def select_roi(eclick, erelease):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    
    global roi
    
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    #print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    #print(f"The buttons you used were: {eclick.button} {erelease.button}")
    roi = np.int16( [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)] )

def on_click(event):
        
    global ix, iy

    # assign global variable to access outside of function
    global coords, labels, predictor, fig, imgplot, img_in, bar_mask
    
    h, w = img_in.shape[:2]
    

    ix, iy = event.xdata, event.ydata
    
    if ix > 1 and ix < w and iy > 1 and iy < h: # > 1 is a trick to avoid considering click on button
        
        if event.button is MouseButton.LEFT: labels.append(1)
        else: labels.append(0)
                
        coords.append((ix, iy))
        
        input_point = np.int16(coords)
        #input_label = np.ones(len(input_point)).astype(np.int16)
        input_label = np.int16(labels)
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
            )
        
        mask_image = show_mask_only(masks)
        bar_mask = masks[0]
        out = cv2.addWeighted(img_in.astype(np.float32), 0.7, mask_image.astype(np.float32), 0.3, 0.0)
        out = out.astype(np.uint8)
        
        for pt, lab in zip(input_point, input_label):
            x, y = pt
            if lab == 1: color = (0, 255, 0)
            else: color = (255, 0, 0)
            cv2.circle(out, (x, y), 3, color, -1)
    
        imgplot.set_data(out)
    fig.canvas.draw_idle()


############# selection of the region to process
roi = None

# Callback function to capture the selected ROI
def select_roi(eclick, erelease):
    global roi
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    roi = (x1, y1, x2, y2)
    print(f"Selected ROI: ({x1}, {y1}) to ({x2}, {y2})")

# Load image using OpenCV
frame = cv2.imread(img_name)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Get the image dimensions
h, w = frame.shape[:2]

# Plot the image
fig_crop, ax = plt.subplots(figsize=(13, 7))
ax.imshow(rgb)

# Initialize the RectangleSelector
selector = RectangleSelector(
    ax, select_roi,
    useblit=True,
    button=[1, 3],  # disable the middle button
    minspanx=5, minspany=5,
    spancoords='pixels',
    interactive=True
)

# Display the plot
plt.show()

# Once the ROI is selected, you can access it from the global 'roi' variable
if roi:
    x1, y1, x2, y2 = roi
    print(f"Selected ROI coordinates: {x1}, {y1}, {x2}, {y2}")
###### definition and intialisation of model ###################
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

##device = "cuda"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=model_path + sam_checkpoint)
sam.to(device=device)

global predictor
predictor = SamPredictor(sam)
################################################################

#frame = cv2.imread(img_name)
#img =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = rgb[y1:y2, x1:x2]

#processing of image ##############
predictor.set_image(img)

global imgplot, img_in, coords, labels, fig, bar_mask


############## selction of the bar with mouse #############
img_in = img.copy()
coords = []
labels = []

fig, ax = plt.subplots(figsize=(16,9))
# Call click func
global cid
cid = fig.canvas.mpl_connect('button_press_event', on_click)
imgplot = plt.imshow(img_in)
#plt.connect('button_press_event', on_click)
axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)
# Create a `matplotlib.widgets.Button` to go to validate pole_selection.
validax = plt.axes([0.8, 0.025, 0.15, 0.04])
button_valid = Button(validax, 'pole validation', color=axcolor, hovercolor='0.975')

def valid_store(event):
    global valid_and_store
    plt.close('all')
    valid_mask = True
button_valid.on_clicked(valid_store)

# Create a `matplotlib.widgets.Button` to go reject mask_selection.
validax = plt.axes([0.2, 0.025, 0.15, 0.04])
button_reject = Button(validax, 'reject', color=axcolor, hovercolor='0.975')

def reject(event):
    plt.close('all')
    #global coords, labels
    #coords = [(0,0)]
    #labels = [0]
button_reject.on_clicked(reject)
plt.show()

################ contour of mask

mask_barre = img[...,0].copy() * 0
mask_barre = np.where(bar_mask, 255, 0).astype(np.uint8)
np.save("mask_barre",mask_barre)
#plt.imshow(mask_barre, cmap='gray')

cnt_barre, _ = cv2.findContours(mask_barre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#epsilon = 0.00001 * cv2.arcLength(cnt_mask[0], True)
epsilon = 0.01
cnt_barre = cv2.approxPolyDP(cnt_barre[0], epsilon, True)
cv2.drawContours(img, [cnt_barre], -1, (0,0,255), 3)

plt.imshow(img)
plt.show()



##################### APPROXIMATION DE LA BARRE ############################


cnt_barre, _ = cv2.findContours(mask_barre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#epsilon = 0.00001 * cv2.arcLength(cnt_mask[0], True)
epsilon = 100
rows,cols = img.shape[:2]
cn=cnt_barre[0]
N=len(cn)
n=N//4
[vx1,vy1,x1,y1] = cv2.fitLine(cn[0:n], 2,0,0.001,0.001)

[vx2,vy2,x2,y2] = cv2.fitLine(cn[3*n:N], 2,0,0.001,0.001)
coef=(vy1/vx1+vy2/vx2)/2
x=(x1+x2)/2
y=(y1+y2)/2

lefty = int((-x*coef) + y)
righty = int(((cols-x)*coef)+y)

cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

plt.imshow(img)
plt.show()



##################### SEGMENTATION DE L'ATHLETE ############################

model = YOLO("yolov8l-seg.pt")  # segmentation model
names = model.model.names
cap = cv2.VideoCapture("video_saut.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('instance-segmentation.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, conf=0.8)
    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(int(cls), True),
                               label=names[int(cls)])

    out.write(im0)
    cv2.imshow("instance-segmentation", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()