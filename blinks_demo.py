import cv2
from concurrent.futures import ProcessPoolExecutor
from pupil_labs.realtime_api.simple import discover_devices, Device
from pl_blinks.blink_detector.blink_detector import blink_detection_pipeline
from pl_blinks.blink_detector.helper import (
    stream_images_and_timestamps,
    update_array,
    compute_blink_rate,
    plot_blink_rate,
)
import seaborn as sns
import numpy as np
import winsound

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
# 	print(f"Searching for devices {device_names} on the network")
# 	while True:
# 		devices_found = discover_devices(timeout)
# 		devices_found_names, devices_found_ips = [], []
# 		for device in devices_found:
# 			print(device_properties(device))
# 			devices_found_names.append(device.phone_name) 
# 			if device.phone_name in device_names:
# 				devices_found_ips.append(device.phone_ip)
# 			device.close()
# 		if (set(device_names) != set(devices_found_names)):
# 			print(f"Missing devices: {set(device_names) - set(devices_found_names)}")
# 			res = input("Search again? (y/n)")
# 			if res in ["N", "n"]:
# 				break
# 		else:
# 			break
# 	return devices_found_ips


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
			# dt_gaze = datetime.fromtimestamp(gaze_sample.timestamp_unix_seconds)
			# dt_scene = datetime.fromtimestamp(scene_sample.timestamp_unix_seconds)
			center_coordinates = (round(gaze_sample.x), round(gaze_sample.y))
			img_with_gaze = cv2.circle(scene_sample.bgr_pixels, center_coordinates, radius, color, thickness)
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


def detect_blinks(device_ip):
	device = Device(device_ip, 8080)
	device_name = device.phone_name
	print("starting stream for device: ", device_properties(device))
	left_images, right_images, timestamps = stream_images_and_timestamps(device)

	# let's keep track of the last 100 blinks
	# blink_times = np.zeros(100)
	# avg_blink_rate = np.zeros(100)
	# blink_rate_last_30s = np.zeros(100)

	blink_counter = 0
	# starting_time = time.time()

	while True:
		try:
		    blink_event = next(blink_detection_pipeline(left_images, right_images, timestamps))

		    blink_counter += 1
		    # elapsed_time = blink_event.start_time / 1e9 - starting_time
		    # blink_times = update_array(blink_times, elapsed_time)
		    # avg_blink_rate = update_array(
		    #     avg_blink_rate, compute_blink_rate(blink_counter, elapsed_time)
		    # )
		    # blink_counter_last_30s = np.sum(blink_times > max(blink_times[0] - 30, 0))
		    # blink_rate_last_30s = update_array(
		    #     blink_rate_last_30s, blink_counter_last_30s / min(30, blink_times[0])
		    # )
		    # plot_blink_rate(blink_times, avg_blink_rate, blink_rate_last_30s)
		    # print("blink detected for device ", device.phone_name)
		    winsound.PlaySound("SystemExit", winsound.SND_ALIAS) #'SystemExit','SystemExclamation', 'SystemAsterisk', 'SystemHand', 'SystemQuestion'
		    winsound.Beep(17000, 300)
		except Exception as e:
			print(device_name, 'error:', e)
			device.close()
			return
		except KeyboardInterrupt:
			device.close()
			return


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

	try:
		# Start streaming from each device on a separate process
		with ProcessPoolExecutor() as executor:
			futures = executor.map(detect_blinks, [arg["device_ip"] for arg in args])

	except KeyboardInterrupt:
		print("KeyboardInterrupt: Exiting gracefully...")

