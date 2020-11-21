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