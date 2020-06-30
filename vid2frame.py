import cv2
from pathlib import Path
 
SAVEPATH="./Data/demo_vid_clip"
# Opens the Video file
cap= cv2.VideoCapture('./Data/test_video.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frame_path=Path(SAVEPATH).joinpath(str(i)+'.jpg')
    print(frame_path)
    cv2.imwrite(str(frame_path),frame)
    i+=1
    if i==2000:
        break
 
cap.release()
cv2.destroyAllWindows()