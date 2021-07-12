class YinsYOLO:

    print("-----------------------------------------------------")
    print(
    """
    Yin's YOLO Package (For Quick Deployment)
    Copyright © YINS CAPITAL, 2009 – Present
    For more information, go to www.YinsCapital.com
    """ )
    print("-----------------------------------------------------")
    
    def Personal_AI_Surveillance(whatObject = 'pistol', useAdvancedYOLO = True, confidence=0.1, whichCam = 0, verbose=False):
        
        """
        # READ:
        # Object detection webcam example using tiny yolo
        # Usage: python object_detection_webcam_yolov3_tiny.py
        """

        # Import necessary packages
        import cvlib as cv
        from cvlib.object_detection import draw_bbox
        import cv2

        # Check out laptop cam:
        # 0 is the first camera (on laptop) 
        # 1 is the second camera (ex. I have a usb cam connected to the laptop that is higher resolution)
        # and then you can do 2, 3, ... if you have installed more cameras.
        # webcam = cv2.VideoCapture(1)
        # print(f'Camera resolution is {webcam.get(3)} by {webcam.get(4)}.')

        # Setup *alert()* Function

        import time
        from IPython.core.magics.execution import _format_time
        from IPython.display import display as d
        from IPython.display import Audio
        from IPython.core.display import HTML
        import numpy as np
        import logging as log

        def alert():
            """ makes sound on client using javascript (works with remote server) """      
            framerate = 44100
            duration  = 0.1
            freq      = 300
            t         = np.linspace(0, duration, framerate*duration)
            data      = np.sin(2*np.pi*freq*t)
            d(Audio(data, rate=framerate, autoplay=True))

        # The following code will start a new window with live camera feed from your laptop. 
        # The notebook will print out a long list of results, with objects detected or not. 
        # To shut it down, make sure current window is in the camera feed and press 'q'. 

        # Open Camera
        webcam = cv2.VideoCapture(whichCam)
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()

        # Loop through frames
        while webcam.isOpened():

            # Read frame from webcam 
            status, frame = webcam.read()
            if not status:
                break

            # Apply object detection
            # 80 common objects: https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
            if useAdvancedYOLO:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3') # this is very slow
            else:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')
            
            # Print Comment
            if verbose:
                print(bbox)
                print(label)
                print(conf)
            
            # Output
            # print(bbox, label, conf)
            # Set Alert (if see a knife)
            tmp = label
            for i in tmp:
                if i == whatObject:
                    alert()

            # Draw bounding box over detected objects
            # We take output from *cv.detect_common_objects* to print them out on videos
            # by using *draw_bbox()*
            out = draw_bbox(frame, bbox, label, conf, write_conf=True)

            # Display output
            cv2.imshow("Yin's AI Surveillance", out)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # End of function
    
        
    def Personal_AI_Surveillance_Basic_Grouping(whatObject = ['cell phone', 'person'], useAdvancedYOLO = True, confidence=0.1, whichCam = 0, useAlert=False, verbose=False):
        
        """
        # READ:
        # Object detection webcam example using tiny yolo
        # Usage: python object_detection_webcam_yolov3_tiny.py
        """

        # Import necessary packages
        import cvlib as cv
        from cvlib.object_detection import draw_bbox
        import cv2

        # Check out laptop cam:
        # 0 is the first camera (on laptop) 
        # 1 is the second camera (ex. I have a usb cam connected to the laptop that is higher resolution)
        # and then you can do 2, 3, ... if you have installed more cameras.
        # webcam = cv2.VideoCapture(1)
        # print(f'Camera resolution is {webcam.get(3)} by {webcam.get(4)}.')

        # SUPPORT FUNCTIONS:
        # Setup *alert()* Function
        import time
        from IPython.core.magics.execution import _format_time
        from IPython.display import display as d
        from IPython.display import Audio
        from IPython.core.display import HTML
        import numpy as np
        import logging as log

        def alert():
            """ makes sound on client using javascript (works with remote server) """      
            framerate = 44100
            duration  = 1
            freq      = 300
            t         = np.linspace(0, duration, framerate*duration)
            data      = np.sin(2*np.pi*freq*t)
            d(Audio(data, rate=framerate, autoplay=True))

        # The following code will start a new window with live camera feed from your laptop. 
        # The notebook will print out a long list of results, with objects detected or not. 
        # To shut it down, make sure current window is in the camera feed and press 'q'. 
        
        def drawBox(img, bbox, labels, confidence, colors=None, write_conf=False, whatObject=whatObject):
            classes = None
            COLORS = np.random.uniform(0, 255, size=(80, 3))

            if classes is None:
                classes = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

            placeholder = []
            for eachItem in whatObject:
                if eachItem in labels:
                    placeholder.append(eachItem)
                    
            
            for i, label in enumerate(placeholder):
                if colors is None:
                    color = COLORS[classes.index(label)]
                else:
                    color = colors[classes.index(label)]
                if write_conf:
                    label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'
                cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)
                if (len(placeholder) > 1) & (i < len(placeholder) - 1):
                    start_point = (round((bbox[i][0]+bbox[i][1])/2+bbox[i][0]), round((bbox[i][2]+bbox[i][3])/2+bbox[i][2]))
                    end_point   = (round((bbox[i+1][0]+bbox[i+1][1])/2+bbox[i+1][0]), round((bbox[i+1][2]+bbox[i+1][3])/2+bbox[i+1][2]))
                else:
                    start_point = (0,0)
                    end_point = (0,0)
                cv2.line(img, start_point, end_point, (0,255,0), 1)
                cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return img

        # Open Camera
        webcam = cv2.VideoCapture(whichCam)
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()

        # Loop through frames
        while webcam.isOpened():

            # Read frame from webcam 
            status, frame = webcam.read()
            if not status:
                break

            # Apply object detection
            # 80 common objects: https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
            if useAdvancedYOLO:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3') # this is very slow
            else:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')
            
            # Print Comment
            if verbose:
                print(bbox)
                print(label)
                print(conf)
            
            # Output
            # print(bbox, label, conf)
            # Set Alert (if see a knife)
            tmp = label
            for i in tmp:
                if i == whatObject:
                    if useAlert:
                        alert()

            # Draw bounding box over detected objects
            # We take output from *cv.detect_common_objects* to print them out on videos
            # by using *draw_bbox()*
            # sample: drawBox(img, bbox, labels, confidence, colors=None, write_conf=False)
            out = drawBox(img=frame, bbox=bbox, labels=label, confidence=conf, write_conf=True, whatObject=whatObject)

            # Display output
            cv2.imshow("Yin's AI Surveillance", out)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # End of function
        
    def Combat_Surveillance(
        windowName = "Yin's AI Surveillance",
        whatObject = ['cell phone', 'person'], 
        useAdvancedYOLO = True, confidence=0.1, 
        whichCam = 0, useAlert=False, verbose=False):
        
        """
        # READ:
        # Object detection webcam example using tiny yolo
        # Usage: python object_detection_webcam_yolov3_tiny.py
        """

        # Import necessary packages
        import cvlib as cv
        from cvlib.object_detection import draw_bbox
        import cv2

        # Check out laptop cam:
        # 0 is the first camera (on laptop) 
        # 1 is the second camera (ex. I have a usb cam connected to the laptop that is higher resolution)
        # and then you can do 2, 3, ... if you have installed more cameras.
        # webcam = cv2.VideoCapture(1)
        # print(f'Camera resolution is {webcam.get(3)} by {webcam.get(4)}.')

        # SUPPORT FUNCTIONS:
        # Setup *alert()* Function
        import time
        from IPython.core.magics.execution import _format_time
        from IPython.display import display as d
        from IPython.display import Audio
        from IPython.core.display import HTML
        import numpy as np
        import logging as log

        def alert():
            """ makes sound on client using javascript (works with remote server) """      
            framerate = 44100
            duration  = 1
            freq      = 300
            t         = np.linspace(0, duration, framerate*duration)
            data      = np.sin(2*np.pi*freq*t)
            d(Audio(data, rate=framerate, autoplay=True))

        # The following code will start a new window with live camera feed from your laptop. 
        # The notebook will print out a long list of results, with objects detected or not. 
        # To shut it down, make sure current window is in the camera feed and press 'q'. 
        
        def drawBox(img, bbox, labels, confidence, colors=None, write_conf=False, whatObject=whatObject):
            classes = None
            COLORS = np.random.uniform(0, 255, size=(80, 3))

            if classes is None:
                classes = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

            placeholder = []
            for eachItem in whatObject:
                if eachItem in labels:
                    placeholder.append(eachItem)
                    
            
            for i, label in enumerate(placeholder):
                if colors is None:
                    color = COLORS[classes.index(label)]
                else:
                    color = colors[classes.index(label)]
                if write_conf:
                    label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'
                cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 1)
                cv2.rectangle(img, 
                              (round((bbox[i][3]-bbox[i][1])*0.2+bbox[i][1]), bbox[i][0]), 
                              (round((bbox[i][3]-bbox[i][1])*0.8+bbox[i][1]), round((bbox[i][2]-bbox[i][0])*0.4+bbox[i][0])), color, 2)
                start_point_1 = (round((bbox[i][3]-bbox[i][1])*0.5+bbox[i][1]), round((bbox[i][2]-bbox[i][0])*0.3+bbox[i][0]))
                end_point_1   = (round((bbox[i][3]-bbox[i][1])*0.1+bbox[i][1]), round((bbox[i][2]-bbox[i][0])*0.9+bbox[i][0]))
                cv2.line(img, start_point_1, end_point_1, (0,255,0), 1)
                start_point_2 = (round((bbox[i][3]-bbox[i][1])*0.5+bbox[i][1]), round((bbox[i][2]-bbox[i][0])*0.3+bbox[i][0]))
                end_point_2   = (round((bbox[i][3]-bbox[i][1])*0.9+bbox[i][1]), round((bbox[i][2]-bbox[i][0])*0.9+bbox[i][0]))
                cv2.line(img, start_point_2, end_point_2, (0,255,0), 1)
                cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(img, "center", start_point_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(img, start_point_1, 30, color, 1)
                start_point_ht = (round((bbox[i][3]-bbox[i][1])*0.5+bbox[i][1]), round((bbox[i][2]-bbox[i][0])*0.5+bbox[i][0]))
                cv2.putText(img, "heart", start_point_ht, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(img, start_point_1, 30, color, 1)
                cv2.putText(img, "elbow", end_point_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(img, end_point_1, 30, color, 1)
                cv2.putText(img, "elbow", end_point_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(img, end_point_2, 30, color, 1)

            return img

        # Open Camera
        webcam = cv2.VideoCapture(whichCam)
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()

        # Loop through frames
        while webcam.isOpened():

            # Read frame from webcam 
            status, frame = webcam.read()
            if not status:
                break

            # Apply object detection
            # 80 common objects: https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
            if useAdvancedYOLO:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3') # this is very slow
            else:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')
            
            # Print Comment
            if verbose:
                print(bbox)
                print(label)
                print(conf)
            
            # Output
            # print(bbox, label, conf)
            # Set Alert (if see a knife)
            tmp = label
            for i in tmp:
                if i == whatObject:
                    if useAlert:
                        alert()

            # Draw bounding box over detected objects
            # We take output from *cv.detect_common_objects* to print them out on videos
            # by using *draw_bbox()*
            # sample: drawBox(img, bbox, labels, confidence, colors=None, write_conf=False)
            out = drawBox(img=frame, bbox=bbox, labels=label, confidence=conf, write_conf=True, whatObject=whatObject)

            # Display output
            cv2.imshow(windowName, out)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # End of function
        
    def Combat_Surveillance_Phase_II(
        windowName = "Yin's AI Surveillance",
        whatObject = ['cell phone', 'person'], 
        useAdvancedYOLO = True, confidence=0.1, 
        whichCam = 0, useAlert=False, verbose=False):
        
        """
        # READ:
        # Object detection webcam example using tiny yolo
        # Usage: python object_detection_webcam_yolov3_tiny.py
        """

        # Import necessary packages
        import cvlib as cv
        from cvlib.object_detection import draw_bbox
        import cv2
        import random

        # Check out laptop cam:
        # 0 is the first camera (on laptop) 
        # 1 is the second camera (ex. I have a usb cam connected to the laptop that is higher resolution)
        # and then you can do 2, 3, ... if you have installed more cameras.
        # webcam = cv2.VideoCapture(1)
        # print(f'Camera resolution is {webcam.get(3)} by {webcam.get(4)}.')

        # SUPPORT FUNCTIONS:
        # Setup *alert()* Function
        import time
        from IPython.core.magics.execution import _format_time
        from IPython.display import display as d
        from IPython.display import Audio
        from IPython.core.display import HTML
        import numpy as np
        import logging as log

        def alert():
            """ makes sound on client using javascript (works with remote server) """      
            framerate = 44100
            duration  = 1
            freq      = 300
            t         = np.linspace(0, duration, framerate*duration)
            data      = np.sin(2*np.pi*freq*t)
            d(Audio(data, rate=framerate, autoplay=True))

        # The following code will start a new window with live camera feed from your laptop. 
        # The notebook will print out a long list of results, with objects detected or not. 
        # To shut it down, make sure current window is in the camera feed and press 'q'. 
        
        def drawBox(img, bbox, labels, confidence, colors=None, write_conf=False, whatObject=whatObject):
            classes = None
            COLORS = np.random.uniform(0, 255, size=(80, 3))

            if classes is None:
                classes = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

            placeholder = []
            for eachItem in whatObject:
                if eachItem in labels:
                    placeholder.append(eachItem)
                    
            
            for i, label in enumerate(placeholder):
                if colors is None:
                    color = COLORS[classes.index(label)]
                else:
                    color = colors[classes.index(label)]
                if write_conf:
                    label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'
                
                # scanning
                cv2.putText(img, "scanning ...", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                # cv2.putText(img, windowName, (300,300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                
                # major detection component
                # cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 1)
                
                # sub component: box surrounding head
                cv2.rectangle(img, 
                              (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.35), round(bbox[i][1]-(bbox[i][3]-bbox[i][1])*0.1)), 
                              (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.65), round(bbox[i][1]+(bbox[i][3]-bbox[i][1])*0.5)), 
                              color, 2)
                
                # sub component: elbow lines
                start_point_1 = (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.5), round(bbox[i][1]+(bbox[i][3]-bbox[i][1])*0.15))
                end_point_1   = (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.1), round(bbox[i][1]+(bbox[i][3]-bbox[i][1])*0.95))
                cv2.line(img, start_point_1, end_point_1, (0,255,0), 1)
                start_point_2 = start_point_1
                end_point_2   = (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.9), round(bbox[i][1]+(bbox[i][3]-bbox[i][1])*0.95))
                cv2.line(img, start_point_2, end_point_2, (0,255,0), 1)
                cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(img, start_point_1, 30, color, 1)
                cv2.putText(img, "elbow", end_point_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(img, end_point_1, 30, color, 1)
                cv2.putText(img, "elbow", end_point_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(img, end_point_2, 30, color, 1)
                
                # sub copmonent: head label
                cv2.putText(img, "head", start_point_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(img, start_point_1, 30, color, 1)
                
                # sub component: nose label
                start_point_chin = (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.5), round(bbox[i][1]+(bbox[i][3]-bbox[i][1])*0.3))
                cv2.putText(img, "nose", start_point_chin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # sub component: heart label
                start_point_heart = (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.5), round(bbox[i][1]+(bbox[i][3]-bbox[i][1])*0.7))
                cv2.putText(img, "heart", start_point_heart, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # sub component: center label
                start_point_heart = (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.5), round(bbox[i][1]+(bbox[i][3]-bbox[i][1])*0.9))
                cv2.putText(img, "center", start_point_heart, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # sub component: fists
                if round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.35) < 200:
                    fist_pos_R = (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.6), round(bbox[i][1]+(bbox[i][3]-bbox[i][1])*0.6))
                    cv2.putText(img, "Alert: jab ...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.putText(img, "Counter-measure: dodge ...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.circle(img, fist_pos_R, random.randint(70, 120), color, 1)
                elif round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.35) > 270:
                    fist_pos_L = (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.4), round(bbox[i][1]+(bbox[i][3]-bbox[i][1])*0.6))
                    cv2.putText(img, "Alert: cross ...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.putText(img, "Counter-measure: weave ...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.circle(img, fist_pos_L, random.randint(70, 120), color, 1)
                else:
                    fist_pos_M = (round(bbox[i][0]+(bbox[i][2]-bbox[i][0])*0.5), round(bbox[i][1]+(bbox[i][3]-bbox[i][1])*0.6))
                    cv2.putText(img, "Alert: no attacks ...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.putText(img, "Counter-measure: attack ...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.circle(img, fist_pos_M, random.randint(70, 120), color, 1)
                
            return img

        # Open Camera
        webcam = cv2.VideoCapture(whichCam)
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()

        # Loop through frames
        while webcam.isOpened():

            # Read frame from webcam 
            status, frame = webcam.read()
            frameCopy = frame
            if not status:
                break

            # Apply object detection
            # 80 common objects: https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
            if useAdvancedYOLO:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3') # this is very slow
            else:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')
            
            # Print Comment
            if verbose:
                print(bbox)
                print(label)
                print(conf)
            
            # Output
            # print(bbox, label, conf)
            # Set Alert (if see a knife)v 
            tmp = label
            for i in tmp:
                if i == whatObject:
                    if useAlert:
                        alert()

            # Draw bounding box over detected objects
            # We take output from *cv.detect_common_objects* to print them out on videos
            # by using *draw_bbox()*
            # sample: drawBox(img, bbox, labels, confidence, colors=None, write_conf=False)
            out = drawBox(img=frame, bbox=bbox, labels=label, confidence=conf, write_conf=True, whatObject=whatObject)

            # Display output
            cv2.imshow(windowName, out)
            cv2.imshow(windowName, frameCopy)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # End of function
        
    # Begin function
    def MaskDetector():
        # Credits: I sourced the code from the following:
        # author: Arun Ponnusamy
        # website: https://www.arunponnusamy.com
        # github: https://github.com/arunponnusamy/cvlib/blob/a21fc330cddb5e06dd82e0acf77e934ae413adee/cvlib/face_detection.py

        # face detection webcam example
        # usage: python face_detection_webcam.py 

        # import necessary packages
        import numpy as np
        import cvlib as cv
        import cv2

        # open webcam
        webcam = cv2.VideoCapture(0)

        if not webcam.isOpened():
            print("Could not open webcam")
            exit()


        # loop through frames
        while webcam.isOpened():

            # read frame from webcam 
            status, frame = webcam.read()

            if not status:
                print("Could not read frame")
                exit()

            # apply face detection
            face, confidence = cv.detect_face(frame)
            #frameCopy = frame
            #frameCopy = np.array(frame)[face[0][1]:face[0][0]][face[0][3]:face[0][2]]
            cropped = frame
            imMEAN = np.mean(cropped)
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            grMEAN = np.mean(gray)
            # thresholds: 
            # np.mean(imMEAN), np.mean(grMEAN), np.std(imMEAN), np.std(grMEAN)
            # (132.47409459264787, 133.2298545294259, 41.95549425306896, 42.26304526011298)
            THRESHOLD = 97
            if imMEAN > THRESHOLD:
                maskLabel = "Mask On"
                print('imMEAN=', imMEAN,'; label:', maskLabel)
            else:
                maskLabel = "No Mask"
                print('imMEAN=', imMEAN, '; label:', maskLabel)
                
            print(face)
            print(confidence)

            # loop through detected faces
            for idx, f in enumerate(face):

                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                #cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                text = "Face detected: " + "{:.2f}%".format(confidence[idx] * 100)
                if maskLabel == 'Mask On':
                    textM_On = "YOU ARE SAFE!"
                    cv2.putText(frame, textM_On, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 3)
                else:
                    textM_No = 'PLEASE WEAR A MASK!'
                    cv2.putText(frame, textM_No, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 3)
                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write confidence percentage on top of face rectangle
                #cv2.putText(frame, text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 3)
                #cv2.putText(frame, str(imMEAN), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 3)
                #cv2.putText(frame, textM, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 3)

            # display output
            cv2.imshow("Real-time face detection", frame)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # release resources
        webcam.release()
        cv2.destroyAllWindows()
        
    # Begin function
    def StarterMode(whichCam=2):
        # Credits: I sourced the code from the following:
        # author: Arun Ponnusamy
        # website: https://www.arunponnusamy.com
        # github: https://github.com/arunponnusamy/cvlib/blob/a21fc330cddb5e06dd82e0acf77e934ae413adee/cvlib/face_detection.py

        # face detection webcam example
        # usage: python face_detection_webcam.py 

        # import necessary packages
        import numpy as np
        import random
        import cvlib as cv
        import cv2

        # open webcam
        webcam = cv2.VideoCapture(whichCam)

        if not webcam.isOpened():
            print("Could not open webcam")
            exit()

        # loop through frames
        while webcam.isOpened():

            # read frame from webcam 
            status, frame = webcam.read()

            if not status:
                print("Could not read frame")
                exit()

            # apply face detection
            # upgrade 1: 
            # there is a build-in CNN
            # that looks for human face inside of the frame
            # frame is coming from camera live feed
            # low-level AI here
            face, confidence = cv.detect_face(frame)
                
#             print(face)
#             print(confidence)

            # loop through detected faces
            for idx, f in enumerate(face):

                # location
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]
                startXleft = startX - 120
                endXright = endX + 120

                # draw rectangle over face
                cv2.rectangle(frame, (startXleft, startY), (endXright, endY), (0,255,0), 2)
                text = "Starter Mode Initiated:"
                textConf = "Confidence: " + "{:.2f}%".format(confidence[idx] * 100)
                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write output
                cv2.putText(frame, text, (startXleft,Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(frame, textConf, (startXleft,Y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.circle(frame, (int(round(startX+0.75*(endX-startX),0)), int(round(startY+0.35*(endY-startY),0))), random.randint(40, 50), (0,255,0), 1)
                
                # write date
                from datetime import datetime
                now = datetime.now()
                dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
                textDate = "Date & Time: " + str(dt_string)
                cv2.putText(frame, textDate, (startXleft,endY+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # write market: SPY
                import yfinance as yf # yahoo finance | same with quantmod(), i.e. in quantmod() the default API is yahoo finance
                import pandas as pd
                from datetime import date
                today = date.today()
                start_date = pd.to_datetime('2013-01-01')
                end_date = pd.to_datetime(str(today))
                SPY = yf.download("SPY", start_date, end_date) # live data
                DIA = yf.download("DIA", start_date, end_date) # live data
                textSPY = "S&P 500 Last Day Close: " + str(round(SPY.iloc[len(SPY)-1][3], 2)) + "; Volume: " + str(round(SPY.iloc[len(SPY)-1][5], 2))
                textDIA = "Dow Jones Index Last Day Close: " + str(round(DIA.iloc[len(DIA)-1][3], 2)) + "; Volume: " + str(round(DIA.iloc[len(DIA)-1][5], 2))
                # backend
                # do more (in backend) in machine learning
                # do RNN
                # do prediction
                # I wake up today, my google glass automatically tell me today's closing price within 7 bucks error margin
                cv2.putText(frame, textSPY, (startXleft,endY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(frame, textDIA, (startXleft,endY+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # display output
            # this is front-end
            cv2.imshow("Starter Mode Initiation", frame)
            
            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # release resources
        webcam.release()
        cv2.destroyAllWindows()
        
    # Define function
    def StealthMode(whichCam=2):
        # library: pip install in command line or !pip install in notebook
        import cv2
        import numpy as np
        import time

        def recordBackground():
            print("""
                      Recording Background .....................
                """)
            cap = cv2.VideoCapture(whichCam) # starts with 0, 1, 2, ... always check to make sure it's the right camera
            time.sleep(1)
            background=0
            for i in range(30):
                ret,background = cap.read()
            cap.release()
            cv2.destroyAllWindows()
            background = np.flip(background,axis=1)
            print("""
                     Background recorded !!
                """)
            return background

        def getInvisible():
            background=recordBackground()

            print("""
                     Get ready to become invisible .....................
                """)
            cap = cv2.VideoCapture(whichCam)
            while(cap.isOpened()):
                ret, img = cap.read()
                frame = img
                img = np.flip(img,axis=1)

                # Converting image to HSV color space. # this is to ensure we have a consistent color pallete
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                value = (35, 35)

                blurred = cv2.GaussianBlur(hsv, value,0) # this is to make the graph smoother but this does not reduce resolution (in case you can tell with human eyes)

                # Create masks:
                # note: color is hard-coded; and it does not take location into consideration
                #       anywhere in the picture frame, as long as the color matches the pre-defined
                #       color threshold, it will get rendered and masked
                # remark: this is an area where AI can step in and fix it
                lower_red = np.array([0,120,70])
                upper_red = np.array([10,255,255])
                mask1 = cv2.inRange(hsv,lower_red,upper_red)

                lower_red = np.array([170,120,70])
                upper_red = np.array([180,255,255])
                mask2 = cv2.inRange(hsv,lower_red,upper_red)

                # Addition of the two masks to generate the final mask.
                mask = mask1+mask2 # we want to cover as many different types of "red" as we possibly can
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)) # this is to remove noise but we maintain resolution level

                # Replacing pixels corresponding to cloak with the background pixels.
                # np.where() equivalent as which() in R
                img[np.where(mask==255)] = background[np.where(mask==255)] # masking effect comes in here, and it will replace the particular location detected with values in the pixesl from background (which is pre-recorded in the previous function)
                cv2.imshow('Stealth Mode Initiated',img)
                cv2.imshow("Source Video", frame)
                
                # Kill screen
                k = cv2.waitKey(1)
                if k == 27:
                    cap.release()
                    break

                # press "Q" to stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()

        # Run function
        getInvisible()
        
    # Begin function
    def FaceEdgeDetector():
        # Credits: I sourced the code from the following:
        # OpenCV: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html#additional-resources

        # import necessary packages
        import numpy as np
        import cvlib as cv
        import cv2
        from matplotlib import pyplot as plt

        # open webcam
        webcam = cv2.VideoCapture(0)

        if not webcam.isOpened():
            print("Could not open webcam")
            exit()


        # loop through frames
        while webcam.isOpened():

            # read frame from webcam 
            status, frame = webcam.read()

            if not status:
                print("Could not read frame")
                exit()

            # apply face detection
            face, confidence = cv.detect_face(frame)
            edges = cv2.Canny(frame, 100, 200)
            

            print(face)
            print(confidence)
            print("----")
            print(edges)
            print(edges.shape)
            print("----")
            print("----")
            print("----")
            print("----")
            print("Edge image...")
            img = frame
            plt.subplot(121),plt.imshow(img,cmap = 'gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(edges,cmap = 'gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            plt.show()

            # loop through detected faces
            for idx, f in enumerate(face):

                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                text = "Face detected: " + "{:.2f}%".format(confidence[idx] * 100)
                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write confidence percentage on top of face rectangle
                cv2.putText(frame, text, (startX,Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
            # loop through detected faces
            for idx, f in enumerate(face):

                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(edges, (startX,startY), (endX,endY), (255,255,255), 2)
                text = "Face detected: " + "{:.2f}%".format(confidence[idx] * 100) + "; Edges marked!"
                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write confidence percentage on top of face rectangle
                cv2.putText(edges, text, (startX,Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            
            def checkList(confidence):
                if not confidence:
                    return True
                else:
                    return False
                
            # display output
            if checkList(confidence):
                cv2.imshow("Real-time Edge Detection", frame)
            else:
                cv2.imshow("Real-time Edge Detection", edges)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # release resources
        webcam.release()
        cv2.destroyAllWindows()
        
    # Begin function
    def FaceCornerDetector(whichCam = 0,):
        # Credits: I sourced the code from the following:
        # OpenCV: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

        # import necessary packages
        import numpy as np
        import cvlib as cv
        import cv2
        from matplotlib import pyplot as plt

        # open webcam
        webcam = cv2.VideoCapture(whichCam)

        if not webcam.isOpened():
            print("Could not open webcam")
            exit()


        # loop through frames
        while webcam.isOpened():

            # read frame from webcam 
            status, frame = webcam.read()

            if not status:
                print("Could not read frame")
                exit()

            # apply face detection
            face, confidence = cv.detect_face(frame)
            img = frame
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray,2,3,0.04)

            # result is dilated for marking the corners, not important
            # dst = cv2.dilate(dst,None)

            # Threshold for an optimal value, it may vary depending on the image.
            img[dst>0.01*dst.max()]=[0,0,255]

            print(face)
            print(confidence)
            print("----")


            # loop through detected faces
            for idx, f in enumerate(face):

                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                #cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                #text = "Face detected: " + "{:.2f}%".format(confidence[idx] * 100)
                #Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write confidence percentage on top of face rectangle
                #cv2.putText(frame, text, (startX,Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
            # loop through detected faces
            for idx, f in enumerate(face):

                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
                text = "Face detected: " + "{:.2f}%".format(confidence[idx] * 100) + "; Edges marked!"
                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write confidence percentage on top of face rectangle
                cv2.putText(img, text, (startX,Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            
            def checkList(confidence):
                if not confidence:
                    return True
                else:
                    return False
                
            # display output
            if checkList(confidence):
                cv2.imshow("Real-time Edge Detection", frame)
            else:
                cv2.imshow("Real-time Edge Detection", img)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # release resources
        webcam.release()
        cv2.destroyAllWindows()
        
        
        
    # Begin function
    def HarrisFeatureFaceDetector():
        # Credits: I sourced the code from the following:
        # OpenCV: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html#additional-resources

        # import necessary packages
        import numpy as np
        import cvlib as cv
        import cv2
        from matplotlib import pyplot as plt

        # open webcam
        webcam = cv2.VideoCapture(0)

        if not webcam.isOpened():
            print("Could not open webcam")
            exit()

        # loop through frames
        while webcam.isOpened():

            # read frame from webcam 
            status, frame = webcam.read()

            if not status:
                print("Could not read frame")
                exit()

            # apply face detection
            face, confidence = cv.detect_face(frame)
            edges = cv2.Canny(frame, 100, 200)
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

            print(face)
            print(confidence)
            print("----")
            print(edges)
            print(edges.shape)
            print("----")
            print("----")
            print("----")
            print("----")
            print("Edge image...")
            img = frame
            plt.subplot(121),plt.imshow(img,cmap = 'gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(edges,cmap = 'gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            plt.show()

            # loop through detected faces
            for idx, f in enumerate(face):

                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                #cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                text = "Face detected: " + "{:.2f}%".format(confidence[idx] * 100)
                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write confidence percentage on top of face rectangle
                cv2.putText(frame, text, (startX,Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
            # loop through detected faces
            for idx, f in enumerate(face):

                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]
                # [[427, 102, 616, 375]]

                # draw rectangle over face
                # cv2.rectangle(edges, (startX,startY), (endX,endY), (255,255,255), 2)
                text = "Face detected: " + "{:.2f}%".format(confidence[idx] * 100) + "; Edges marked!"
                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write confidence percentage on top of face rectangle
                cv2.putText(edges, text, (startX,Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                # Harris Corner Detector
                dst = cv2.cornerHarris(edges,2,3,0.001)
                print("dst here:")
                print(dst)
                print(dst.shape)
                print(type(dst))

                # result is dilated for marking the corners, not important
                dst = cv2.dilate(dst,None)

                # limit output of Harris Corner Detector to be within human face rectangle
                dst[:startY, :]=0
                dst[endY::, :]=0
                dst[:, :startX]=0
                dst[:, endX::]=0
                
                # Threshold for an optimal value, it may vary depending on the image.
                img[dst>0.01*dst.max()]=[0,255,0]

            
            def checkList(confidence):
                if not confidence:
                    return True
                else:
                    return False
                
            # display output
            if checkList(confidence):
                cv2.imshow("Real-time Live Cam", frame)
                cv2.imshow("Real-time Edge Detection", img)
            else:
                cv2.imshow("Real-time Live Cam", frame)
                cv2.imshow("Real-time Edge Detection", img)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # release resources
        webcam.release()
        cv2.destroyAllWindows()
