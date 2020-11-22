# BigBrotherICU


## Image segment selection
```bash
ffmpeg -i /data/ikem_hackathon/KOAK\ Box\ 5.avi -ss 00:28:16 -t 00:02:00 -c copy sestry_prichazi.mp4
```

## Tracking 
Currently the selection of ROI is done manually, I realised that while tracking nurses (more generally the people in blue on a blue floor), it's good to select just upper part of the body with head and arms (i supposed because of the skin color) instead the whole body - the tracking works way better.

It seems to me as a good compromise between speed and precision use the KCF tracker - can achieve 30+ frames per seconds while drawing the bounding boxes and the textual information about tracking


## Privacy
For privacy reasons, the patient should be obfuscated
```python
bb = (231, 91, 87, 106)
blur_or_blacken(frame, bb, blur=False)
cv2.imshow("Frame", frame)
key = cv2.waitKey(0)
```
## Visualisation
There is simplistic tool for visualising the interesing events
simply run it by
```bash
python -m http.server 8080
```

### Description
A monitoring system for ICU in hospitals which is capable of detecting patietns and medical personnel, distinguishing between doctors and nurse, detecting actions of people such as lying, sitting, walking, eating etc. Detection of such actions allows to get a better overview of what happens in ICU at all times, for example a frequency of patients walking during a night. 


