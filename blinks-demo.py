import cv2
import os, sys
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
import platform

def device_properties(device: Device):
    return f"""Device {device.phone_name} 
    ip           : {device.phone_ip} 
    id           : {device.phone_id} 
    battery      : {device.battery_state}({device.battery_level_percent})
    memory state : {device.memory_state}({device.memory_num_free_bytes} free bytes)
    glasses ver. : {device.version_glasses}
    glasses sno. : {device.serial_number_glasses}
    scenecam sno.: {device.serial_number_scene_cam}"""


def detect_blinks(args):
	device = Device(args["device_ip"], 8080)
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
		    print("blink detected for device ", device.phone_name)
		    if platform.system() == "Windows":
		        import winsound
		        freq = 1000*int(args["device_ip"].strip(".")[-1][-1])
		        # winsound.PlaySound("SystemExit", winsound.SND_ALIAS) #'SystemExit','SystemExclamation', 'SystemAsterisk', 'SystemHand', 'SystemQuestion'
		        winsound.Beep(freq, 1000)
		        print("Sound frequency:", freq)
		    elif platform.system() == "Darwin":
		        # sys.stdout.write('\a')
		        os.system(f"say \"{args['say']}\"")
		    else:
		        print("Unsupported platform.")	    	
		    	
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
	if len(eval(device_ids)) > 2:
		raise Exception("More than two devices provided. Exiting...")
	else:
		print("Using subnet 192.168.25.1xx.")

	palette = ["ey", "o"]
	args = []
	for i,id_ in enumerate(eval(device_ids)):
		args.append({"device_ip": "192.168.25.{:d}".format(id_+100),
			"say": palette[i]
		})

	try:
		# Start streaming from each device on a separate process
		with ProcessPoolExecutor() as executor:
			futures = executor.map(detect_blinks, args)

	except KeyboardInterrupt:
		print("KeyboardInterrupt: Exiting gracefully...")
		sys.exit()

