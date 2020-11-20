import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
cv2.destroyAllMacs = cv2.destroyAllWindows
from tracking import init_tracker, update_tracker

args = {'tracker': 'kcf'}

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if not args.get("video", False):
#    print("[INFO] starting video stream...")
#    vs = VideoStream(src=0).start()
#    time.sleep(1.0)
# otherwise, grab a reference to the video file
# else:
#    vs = cv2.VideoCapture(args["video"])
# initialize the FPS throughput estimator
fps = None
tracker = {}

cap = cv2.VideoCapture('/data/ikem_hackathon/sestry_prichazi.mp4')
while(cap.isOpened()): # for video files
# while True: # for captured streams

    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    ret, frame = cap.read()

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
	    update_tracker(frame, tracker, fps, draw_rectangle=True, draw_stats=True)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        tracker, initBB = init_tracker(frame, tracker, None,
                                       args['tracker'], manual_selection=True)
        fps = FPS().start()
    elif key == ord('q'):
        break

cv2.destroyAllMacs() # all credits go to Kubiczek