import cv2
from concurrent.futures import ProcessPoolExecutor
from pupil_labs.realtime_api.simple import discover_devices, Device
import seaborn as sns
import numpy as np

def device_properties(device: Device):
    return f"""Device {device.phone_name} 
    ip           : {device.phone_ip} 
    id           : {device.phone_id} 
    battery      : {device.battery_state}({device.battery_level_percent})
    memory state : {device.memory_state}({device.memory_num_free_bytes} free bytes)
    glasses ver. : {device.version_glasses}
    glasses sno. : {device.serial_number_glasses}
    scenecam sno.: {device.serial_number_scene_cam}"""

## Pupil Labs' DNS search is not always reliable 
# def find_devices(device_names, timeout=5):
#   print(f"Searching for devices {device_names} on the network")
#   while True:
#       devices_found = discover_devices(timeout)
#       devices_found_names, devices_found_ips = [], []
#       for device in devices_found:
#           print(device_properties(device))
#           devices_found_names.append(device.phone_name) 
#           if device.phone_name in device_names:
#               devices_found_ips.append(device.phone_ip)
#           device.close()
#       if (set(device_names) != set(devices_found_names)):
#           print(f"Missing devices: {set(device_names) - set(devices_found_names)}")
#           res = input("Search again? (y/n)")
#           if res in ["N", "n"]:
#               break
#       else:
#           break
#   return devices_found_ips


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
            scene_sample, eyes_sample, gaze_sample = device.receive_matched_scene_and_eyes_video_frames_and_gaze()
            eye_size = [args["window_resolution"][0] / 3, 0]
            eyes = cv2.resize(eyes_sample.bgr_pixels,
                              [args["window_resolution"][0],
                               args["window_resolution"][1] // 2])

            # dt_gaze = datetime.fromtimestamp(gaze_sample.timestamp_unix_seconds)
            # dt_scene = datetime.fromtimestamp(scene_sample.timestamp_unix_seconds)
            center_coordinates = (round(gaze_sample.x), round(gaze_sample.y))
            img_with_gaze = cv2.circle(scene_sample.bgr_pixels, center_coordinates, radius, color, thickness)
            img_with_gaze = cv2.circle(img_with_gaze, center_coordinates, int(radius*0.75), (255, 255, 255), int(thickness*0.5))
            img_with_gaze = cv2.circle(img_with_gaze, center_coordinates, int(radius*0.50), (0, 0, 0), int(thickness*0.5))
            img_with_gaze = cv2.circle(img_with_gaze, center_coordinates, int(radius*0.25), (255, 255, 255), int(thickness*0.5))
            img_with_gaze = cv2.circle(scene_sample.bgr_pixels, center_coordinates, int(radius*0.15), color, int(thickness*0.25))

            resized = cv2.resize(img_with_gaze, args["window_resolution"])

            img_with_gaze_and_eyes = cv2.vconcat([resized, eyes]) 
            cv2.imshow(device_name,  img_with_gaze_and_eyes)

            if cv2.waitKey(1) == ord('q'):
                device.close()
                cv2.destroyWindow(device_name)
                break
        except Exception as e:
            print(device_name, 'close... error:', e)
            device.close()
            cv2.destroyWindow(device_name)
        except KeyboardInterrupt:
            device.close()
            cv2.destroyWindow(device_name)

if __name__ == '__main__':
    
    device_ids = input("Input device ids/numbers (as list of ints)")
    assert type(eval(device_ids))==list, "Incorrect input provided"

    palette = sns.color_palette("colorblind", len(device_ids))
    args = []
    for i,id_ in enumerate(eval(device_ids)):
        args.append({"device_ip": "192.168.25.{:d}".format(id_+100),
            "color": palette[i],
            # Scene video presentation size
            "window_resolution": [640,480],
            # Gaze circle parameters
            "radius": 30,
            "thickness": 20
        })

    # Start streaming from each device on a separate process
    try:
        with ProcessPoolExecutor() as executor:
            futures = executor.map(stream_from_device, args)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
   
    cv2.destroyAllWindows()
