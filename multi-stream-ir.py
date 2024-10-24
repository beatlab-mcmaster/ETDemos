import cv2
from concurrent.futures import ProcessPoolExecutor
from pupil_labs.realtime_api.simple import discover_devices, Device
import seaborn as sns
import numpy as np

## ArUco detection
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)


def device_properties(device: Device):
    return f"""Device {device.phone_name} 
    ip           : {device.phone_ip} 
    id           : {device.phone_id} 
    battery      : {device.battery_state}({device.battery_level_percent})
    memory state : {device.memory_state}({device.memory_num_free_bytes} free bytes)
    glasses ver. : {device.version_glasses}
    glasses sno. : {device.serial_number_glasses}
    scenecam sno.: {device.serial_number_scene_cam}"""


def stream_from_device(args):
    device = Device(args["device_ip"], 8080)
    print("starting stream for device: ", device_properties(device))
    device_name = device.phone_name
    # print("starting stream for device ", device_name)
    radius = args["radius"]
    color = args["color"]
    thickness = args["thickness"]

    while True:
        try:
            scene_sample, gaze_sample = device.receive_matched_scene_video_frame_and_gaze()
            worldview_frame = scene_sample.bgr_pixels
            
            #Find arUco markers in scene
            gray = cv2.cvtColor(worldview_frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray)
            print("Detected markers:", ids)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(worldview_frame, corners, ids)


            # dt_gaze = datetime.fromtimestamp(gaze_sample.timestamp_unix_seconds)
            # dt_scene = datetime.fromtimestamp(scene_sample.timestamp_unix_seconds)
            center_coordinates = (round(gaze_sample.x), round(gaze_sample.y))
            img_with_gaze = cv2.circle(worldview_frame, center_coordinates, radius, color, thickness)
            img_with_gaze = cv2.circle(img_with_gaze, center_coordinates, int(radius*0.75), (255, 255, 255), int(thickness*0.5))
            img_with_gaze = cv2.circle(img_with_gaze, center_coordinates, int(radius*0.50), (0, 0, 0), int(thickness*0.5))
            img_with_gaze = cv2.circle(img_with_gaze, center_coordinates, int(radius*0.25), (255, 255, 255), int(thickness*0.5))
            img_with_gaze = cv2.circle(scene_sample.bgr_pixels, center_coordinates, int(radius*0.15), color, int(thickness*0.25))

            resized = cv2.resize(img_with_gaze, args["window_resolution"])
            cv2.imshow(device_name, resized)
            if cv2.waitKey(1) == ord('q'):
                device.close()
                break
        except Exception as e:
            print(device_name, 'close... error:', e)
            device.close()
        except KeyboardInterrupt:
            device.close()

if __name__ == '__main__':

    default_network_id = "192.168.2"
    network_id = input(f"Input network id or press Enter to use default ({default_network_id}) \n")
    network_id = default_network_id if not network_id.strip() else network_id

    device_ids = input("Input device ids/numbers (as list of ints)")
    assert type(eval(device_ids))==list, "Incorrect input provided"

    palette = sns.color_palette("colorblind", len(device_ids))
    args = []
    for i,id_ in enumerate(eval(device_ids)):
        args.append({"device_ip": f"{network_id}.{id_+100}",
            "color": palette[i],
            # Scene video presentation size
            "window_resolution": [640,480],
            # Gaze circle parameters
            "radius": 30,
            "thickness": 20
        })

    # Start streaming from each device on a separate process
    with ProcessPoolExecutor() as executor:
        futures = executor.map(stream_from_device, args)