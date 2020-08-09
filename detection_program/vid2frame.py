import cv2
from pathlib import Path
 
SAVEPATH="data/video/frame"
# Opens the Video file
cap= cv2.VideoCapture('data/video/vid.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frame_path=Path(SAVEPATH).joinpath(str(i).zfill(5)+'.jpg')
    cv2.imwrite(str(frame_path),frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()