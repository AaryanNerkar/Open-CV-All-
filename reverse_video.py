import cv2
cap=cv2.VideoCapture("mixkit-man-and-woman-jogging-together-on-the-street-40881-hd-ready.mp4")
l=[]
c=1
while True:
    ret,frame=cap.read()

    if ret==True:
        file_name=r"C:\Users\Hp\OneDrive\Desktop\PY_DSA\img\demo"+str(c)+".jpg"
        l.append(file_name)
        cv2.imwrite(file_name,frame)
        c=c+1
        
        cv2.imshow("frame",frame)
        if cv2.waitKey(8)&0xff==ord('q'):
         break   
    else:
        break
l.reverse()
for i in 1:
    img_new=cv2.imread(i)
    
    cv2.imshow("new_frame",img_new)
    if cv2.waitkey(18)&0xff==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()