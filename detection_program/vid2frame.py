import cv2
from pathlib import Path
 
SAVEPATH="D:/mike/github/Lane_Detection/data/video/frame"
# Opens the Video file
cap= cv2.VideoCapture('D:/mike/github/Lane_Detection/data/video/video_1.avi')
i=0
while(cap.isOpened()):
    print(i)
    ret, frame = cap.read()
    if ret == False:
        break
    frame_path=Path(SAVEPATH).joinpath(str(i).zfill(5)+'.jpg')
    cv2.imwrite(str(frame_path),frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()