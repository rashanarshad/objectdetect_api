import base64
import cv2
import numpy as np
from urllib.parse import quote
from base64 import b64encode
import keras.backend as K
import keras.backend.tensorflow_backend as tb
from object_detect import  decode_netout, correct_yolo_boxes, get_boxes, draw_boxes, do_nms
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form
import uvicorn
K.set_image_data_format('channels_last')

app = FastAPI()


origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('model.h5')





def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

@app.get('/health-check')
def healthcheck():
    return {'healthy': 'true'}

@app.post('/')
async def return_detected(file: str = Form(...)):
    #convert string of image data to uint8
    img64 = file
    img = readb64(img64)

    #keep original image and dimensions to draw and return to request
    img_orig = img
    h_orig, w_orig, _ = img_orig.shape

    w_orig = w_orig /416
    h_orig = h_orig /416
    #resize img for model
    img = cv2.resize(img, (416, 416))
    tb._SYMBOLIC_SCOPE.value = True

    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo
    # load and prepare image
    image, image_w, image_h = img, input_w, input_h
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    # make prediction
    yhat = model.predict(image)
    # summarize the shape of the list of arrays
    print([a.shape for a in yhat])
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the probability threshold for detected objects
    class_threshold = 0.6
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # define the labels
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    # draw what we found
    img = draw_boxes(img_orig, v_boxes, v_labels, v_scores, w_orig, h_orig)

    retval, buffer = cv2.imencode('.jpg', img)
    data = b64encode(buffer)
    data = data.decode('ascii')
    data_url = 'data:image/webp;base64,{}'.format(quote(data))
    return data_url


