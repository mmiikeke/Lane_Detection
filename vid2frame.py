import cv2
from pathlib import Path
 
SAVEPATH="/Data/demo_vid_frame"
# Opens the Video file
cap= cv2.VideoCapture('path/to/video')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frame_path=Path(SAVEPATH).joinpath(str(i)+'jpg')
    cv2.imwrite(str(frame_path),frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()