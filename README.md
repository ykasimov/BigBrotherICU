# BigBrotherICU


## Tracking 
Currently the selection of ROI is done manually, I realised that while tracking nurses (more generally the people in blue on a blue floor), it's good to select just upper part of the body with head and arms (i supposed because of the skin color) instead the whole body - the tracking works way better.

It seems to me as a good compromise between speed and precision use the KCF tracker - can achieve 30+ frames per seconds while drawing the bounding boxes and the textual information about tracking