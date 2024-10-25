import cv2
import numpy as np
import time

## find cameras
def find_available_cameras(max_cameras=10):
    available_cameras = []
    
    # Loop through possible camera indices and test if the camera is available
    for camera_id in range(max_cameras):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            available_cameras.append(camera_id)
            cap.release()  # Release the camera once it's confirmed to be working
    
    return available_cameras

available_cameras = find_available_cameras()

if available_cameras:
    print(f"Available camera IDs: {available_cameras}")
else:
    print("No cameras found.")

## set stream
id = input("input id to use for stream: ")
cap = cv2.VideoCapture(int(id))

## ArUco detection
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

time.sleep(2)


width, height = 800, 600
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if np.ravel(ids).shape[0] == 4:
        print("Corners", corners)
        pts1 = np.mean(np.stack([corners[0], corners[1], corners[2], corners[3]], axis=0), axis = 2).squeeze()
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped_frame = cv2.warpPerspective(frame, matrix, (width, height))
        cv2.imshow('transform', warped_frame)

        cv2.aruco.drawDetectedMarkers(gray, corners, ids)
        
    cv2.imshow("frame", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()