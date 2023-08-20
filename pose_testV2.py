from ultralytics import YOLO 
import math
import cv2

model = YOLO("yolov8n-pose.pt") 
             
results = model.predict(source='D:/Image processing/YOLO pose -20230815/data/data_pose/12.jpg', save=True, conf=0.5)
print(results[0].boxes.data.tolist())

def calculateAngle(keypoint1, keypoint2, keypoint3):
    angle = math.degrees(math.atan2(int(keypoint3[1]) - int(keypoint2[1]),int(keypoint3[0]) - int(keypoint2[0]) ) - math.atan2(int(keypoint1[1]) - int(keypoint2[1]),int(keypoint1[0]) - int(keypoint2[0])))
    return angle

def distance_btw_wrist(i):
    right_wrist = results[0].keypoints.data[i][10]
    left_wrist = results[0].keypoints.data[i][9]
    difference = (math.sqrt((right_wrist[0] - left_wrist[0])**2 + (right_wrist[1] - left_wrist[1])**2))/10
    return difference

def looking_down(i):
    left_eye = int(results[0].keypoints.data[i][1][1])
    right_eye = int(results[0].keypoints.data[i][2][1])
    left_ear = int(results[0].keypoints.data[i][3][1])
    right_ear = int(results[0].keypoints.data[i][4][1])
    if(left_eye>left_ear or right_eye>right_ear):
        msg = "looking down"
    else:
        msg = "looking up"
    return msg

for i in range(0,len(results[0].boxes.data)):
    right_elbow = abs(calculateAngle(results[0].keypoints.data[i][6],results[0].keypoints.data[i][8],results[0].keypoints.data[i][10]))
    left_elbow = abs(calculateAngle(results[0].keypoints.data[i][9],results[0].keypoints.data[i][7],results[0].keypoints.data[i][5]))
    eye_mid = (((results[0].keypoints.data[i][1][0]) + (results[0].keypoints.data[i][2][0])) / 2, ((results[0].keypoints.data[i][1][1]) + (results[0].keypoints.data[i][2][1])) / 2)
    shoulder_mid = (((results[0].keypoints.data[i][5][0]) + (results[0].keypoints.data[i][6][0])) / 2, ((results[0].keypoints.data[i][5][1]) + (results[0].keypoints.data[i][6][1])) / 2)
    neck_bend_angle2 =abs(calculateAngle(shoulder_mid,results[0].keypoints.data[i][0],eye_mid))
    if (neck_bend_angle2>180):
        neck_angle = 360 - neck_bend_angle2
    else:
        neck_angle = neck_bend_angle2

    def phone_looking(i):
        if (((looking_down(i)== "looking down")and (distance_btw_wrist(i)<10)) or ((110 < neck_angle < 120) and (0.5 < distance_btw_wrist(i) < 0.9))):
            label = "Looking at phone"
        else:
           label = "Not looking at phone"
        return label
    
    print(f'The calculated left albow angle of {i} person is {left_elbow}')
    print(f'The calculated right albow angle of {i} person is {right_elbow}')
    print(f'The differnce {i} person is {distance_btw_wrist(i)}')
    print(f'The head bend {i} person is {neck_angle}')
    print(f'The person {i} is {looking_down(i)}')

im= cv2.imread("D:/Image processing/YOLO pose -20230815/data/data_pose/12.jpg")

while(1):
    for i in range(0,len(results[0].boxes.data)):
        x1,y1,x2,y2,score,class_id = results[0].boxes.data[i]
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)
        cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(im, phone_looking(i), (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        #cv2_imshow(im)
    cv2.imshow("Annotated Image",im)
    if(cv2.waitKey(1)& 0xFF == ord('q')):
       break
p1=results[0].boxes.data[0] 
cv2.rectangle(im,(int(p1[0]),int(p1[1])),(int(p1[2]),int(p1[3])),(0,255,0),5)
cv2.destroyAllWindows()