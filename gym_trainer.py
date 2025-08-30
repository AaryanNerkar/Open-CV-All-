import cv2
import mediapipe as mp
import numpy as np
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose
#calculating angle
def calculate_angle(a,b,c):
        a=np.array(a)
        b=np.array(b)
        c=np.array(c)
        
        radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
        angle=np.abs(radians*180.0/np.pi)
        if angle>180.0:
                angle=360-angle
        return angle
        

cap=cv2.VideoCapture(0)
#curl counter variables
counter=0
stage=None
#setup mediapipe pose
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
   while cap.isOpened():
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    #make detection
    results=pose.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
    #extract landmarks
    try:
        landmarks=results.pose_landmarks.landmark
        
        #Get coordinates
        shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        #calculate angle
        angle=calculate_angle(shoulder,elbow,wrist)
        #visualize angle
        cv2.putText(image,str(angle),
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,255),2,cv2.LINE_AA)
                    
        print(landmarks)
        #Curl counter logic
        if angle>160:
                stage="down"
        if angle<30 and stage=="down":
                stage="up"
                counter=+1
                print(counter)
        
    except:
            pass
    cv2.rectangle(image,(0,0),(225,73),(245,117,16),-1)
    #rep data
    cv2.putText(image,'REPS',(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(image,str(counter),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
    
    cv2.putText(image,'STAGE',(65,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(image,stage,(60,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
    
    #render detection
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
    cv2.imshow("media pipe feed",image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()