import cv2

def init_tracker(frame, tracker, initBB, tracker_name='kcf', manual_selection=False):
    """frame is openCV image, ie. cv.Mat
    initBB is a openCV rectangle or n-tuple of values (x, y, w, h)
    tracker_name has to be known opencv tracker - NB. opencv-contrib has to be install in
    order to get the trackers working
    manual_selection allows to select the ROI by hand
    """

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    if tracker_name in OPENCV_OBJECT_TRACKERS:
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()

        if manual_selection:
            initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        
        tracker.init(frame, initBB)
        return tracker, initBB            


def update_tracker(frame, tracker, fps, draw_rectangle=True, draw_stats=False):
    """Method updating the tracker by taking the previous 
    frame and previous bounding box
    """
    
    (success, box) = tracker.update(frame)
    if success and draw_rectangle:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    fps.update()
    fps.stop()
    if draw_stats:
        (H, W) = frame.shape[:2]
        info = [
                ("Tracker", str(tracker)),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
        for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)