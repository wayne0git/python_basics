# How to use streamlit
# 1. Run *.py in CMD by streamlit run *.py
# 2. Open web browser to see UI
# 3. Modify *.py code and press Rerun in web browser

import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import time

from matplotlib import pyplot as plt

# Constant
MIN_CONF = 0.5
NMS_THRESH = 0.3

def detect_people(frame, net, ln, personIdx=0):
    # grab the dimensions of the frame and  initialize the list of
    # results
    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, centroids, and
    # confidences, respectively
    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter detections by (1) ensuring that the object
            # detected was a person and (2) that the minimum
            # confidence is met
            if classID == personIdx and confidence > MIN_CONF:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results

if __name__ == '__main__':
    # Title
    st.title('Streamlit example -- Yolo Detection')
    st.markdown('### [Reference](https://towardsdatascience.com/make-your-machine-learning-models-come-alive-with-streamlit-48e6eb8e3004)')

    # Select Page
    choice = st.sidebar.selectbox('Select Main Page', ['Get Started', 'YOLO Detection'])

    if choice == 'Get Started':
        # Checkbox
        show_txt = st.sidebar.checkbox('Show Text')
        show_plt = st.sidebar.selectbox('Show Plot', ['Plot', 'Map', 'Image'])

        # Streamlit basics
        st.header('Installation')
        st.code('pip install streamlit')

        st.header('How to import')
        st.code('import streamlit as st')

        st.header('How to use')
        st.code('streamlit run *.py')
        st.code('streamlit run URL/*.py')

        # Streamlit Text
        if show_txt:
            st.header('Show text / data')
            with st.echo():
                st.header('Text (Header)')
                st.subheader('Text (Subheader)')
                st.write('Text (Write)')
                'Text (Use st.write implicitly)' 

        # Streamlit Plot
        st.header('Show Plot')
        if show_plt == 'Plot':
            with st.echo():
                st.line_chart(pd.DataFrame(np.random.randn(20, 2), columns=['a', 'b']))
        elif show_plt == 'Map':
            with st.echo():
                st.map(pd.DataFrame(np.random.randn(10, 2) / [50, 50] + [37.76, -122.4], columns=['lat', 'lon']))
        else:
            with st.echo():
                st.image(np.random.randint(0, 255, (50, 50)), 'Show Image')

        # Streamlit Utility
        st.header('Show ProgressBar')
        with st.echo():
            bar = st.progress(0)
            for i in range(100):
                bar.progress(i)
    else:
        # Upload image
        st.header('Upload image data for YOLO detection')
        st.subheader('Support JPG, BMP, PNG')
        image_file = st.file_uploader('Upload image', type=['jpg', 'bmp', 'png'])

        show_prob = st.sidebar.checkbox('Show Probability')

        if image_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            # YOLO Detection
            # Preprocess
            frame = cv2.resize(frame, (416, 416))

            # Get parameter
            MIN_CONF = st.sidebar.slider('Min Conf', 0.0, 1.0, 0.3, 0.05)

            # Inference
            if st.button('Run Inference'):
                # Initialization
                # load the COCO class labels our YOLO model was trained on
                labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
                LABELS = open(labelsPath).read().strip().split("\n")

                # derive the paths to the YOLO weights and model configuration
                weightsPath = os.path.sep.join(['yolo-coco', "yolov3.weights"])
                configPath = os.path.sep.join(['yolo-coco', "yolov3.cfg"])

                # load our YOLO object detector trained on COCO dataset (80 classes)
                net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

                # determine only the *output* layer names that we need from YOLO
                ln = net.getLayerNames()
                ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

                results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

                if show_prob:
                    # Show detection results in dataframe
                    probs = [result[0] for result in results]
                    df = pd.DataFrame(dict(ID=list(range(len(results))), Prob=probs))
                    st.dataframe(df)

                    # Simple plot
                    st.line_chart(df['Prob'])

                # Loop over the results
                for (i, (prob, bbox, centroid)) in enumerate(results):
                    # Extract the bounding box and centroid coordinates
                    (startX, startY, endX, endY) = bbox
                    (cX, cY) = centroid

                    # Overlay
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.circle(frame, (cX, cY), 5, (0, 255, 0), 1)

                # Show result
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width = True)
                st.success("Found {} persons\n".format(len(results)))
