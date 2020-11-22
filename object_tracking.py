import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
cv2.destroyAllMacs = cv2.destroyAllWindows
from tracking import init_tracker, update_tracker
from pose_estimation import get_pose
from utils import non_max_suppression_fast as non_max_suppression
from utils import blur_or_blacken
import numpy as np
from collections import defaultdict

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
frame_counter = 0

# cap = cv2.VideoCapture('/data/ikem_hackathon/KOAK Box 5.avi')
# cap = cv2.VideoCapture('/data/ikem_hackathon/sestry_prichazi.mp4')
# cap = cv2.VideoCapture('/data/ikem_hackathon/nurse_and_night_ligth_transition.mp4')

cap = cv2.VideoCapture('/data/ikem_hackathon/cuts/frames_detections/sestry_prichazy/outt.mp4')

visualize = False #True # False
original_fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total)
original_fps_rounded = int(np.round(original_fps))
print(f"original fps of the video: {original_fps}")
number_of_persons_stats = defaultdict(int)
segments_with_multiple_people = []
slack_for_people_detection = 0
treshold_slack_for_people_detection = 4 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

img_cnt = 0
more_people = 0
while(cap.isOpened()): # for video files
# while True: # for captured streams

    
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    ret, frame = cap.read()
    if frame is None:       
        break        
    frame_counter += 1
    
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=500)    
    if img_cnt == 0:
        height, width = frame.shape[:2]
        out = cv2.VideoWriter('/data/ikem_hackathon/processed/1.mp4', fourcc, 24.0, (width, height))
    img_cnt += 1

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
	    update_tracker(frame, tracker, fps, draw_rectangle=True, draw_stats=True)

    if frame_counter:# % original_fps_rounded == 0:        
        personsKeypoints = get_pose(frame, draw_pose=True)
        out.write(frame)
        no_of_persons = len(personsKeypoints)
        if no_of_persons > 1 and more_people == 0:
            #new segment with more people
            segments_with_multiple_people.append({"start":frame_counter/original_fps_rounded})
            more_people += 1
        if no_of_persons == 1 and more_people > 0:
            slack_for_people_detection += 1 
            if slack_for_people_detection > treshold_slack_for_people_detection:
                segments_with_multiple_people[-1].update({"end":frame_counter/original_fps_rounded})
                slack_for_people_detection = 0
                more_people = 0

        number_of_persons_stats[no_of_persons] += 1

    if visualize:
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
print(number_of_persons_stats)
print(segments_with_multiple_people)