import cv2
import numpy as np
import csv
from PIL import Image
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
cap = cv2.VideoCapture(0)

data = pd.read_csv("D:/Studia/MGR/III semestr/Praca Dyplomowa/CD/Program_glowny/train_spr.csv").as_matrix()

clf = DecisionTreeClassifier()

xtrain = data[0:31,1:22501]
train_label=data[0:31,0]

clf.fit(xtrain, train_label)
how_letter = '0'
delay = 0
     
while(1):
        
    try:
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
    #define region of interest
        roi=frame[100:250, 100:250]
         
        cv2.rectangle(frame,(100,100),(250,250),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
         
    # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
     
    #Print text in frame    
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Exit = ESC',(0,450), font, 2, (0,255,0), 3, cv2.LINE_AA)
        
        if how_letter[0] == '1':
            cv2.putText(frame,'You showed the letter: A',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '2':
            cv2.putText(frame,'You showed the letter: B',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '3':
            cv2.putText(frame,'You showed the letter: C',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '4':
            cv2.putText(frame,'You showed the letter: D',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '5':
            cv2.putText(frame,'You showed the letter: E',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '6':
            cv2.putText(frame,'You showed the letter: F',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '7':
            cv2.putText(frame,'You showed the letter: G',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '8':
            cv2.putText(frame,'You showed the letter: H',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '9':
            cv2.putText(frame,'You showed the letter: I',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '10':
            cv2.putText(frame,'You showed the letter: J',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '11':
            cv2.putText(frame,'You showed the letter: K',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '12':
            cv2.putText(frame,'You showed the letter: L',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '13':
            cv2.putText(frame,'You showed the letter: M',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '14':
            cv2.putText(frame,'You showed the letter: N',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '15':
            cv2.putText(frame,'You showed the letter: O',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '16':
            cv2.putText(frame,'You showed the letter: P',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)        
        elif how_letter[0] == '17':
            cv2.putText(frame,'You showed the letter: Q',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '18':
            cv2.putText(frame,'You showed the letter: R',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '19':
            cv2.putText(frame,'You showed the letter: S',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '20':
            cv2.putText(frame,'You showed the letter: T',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '21':
            cv2.putText(frame,'You showed the letter: U',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '22':
           cv2.putText(frame,'You showed the letter: V',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '23':
            cv2.putText(frame,'You showed the letter: W',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '24':
           cv2.putText(frame,'You showed the letter: X',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '25':
            cv2.putText(frame,'You showed the letter: Y',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '26':
            cv2.putText(frame,'You showed the letter: Z',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        elif how_letter[0] == '27':
            cv2.putText(frame,'You showed the letter: NO HAND',(0,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
        #show the windows
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
        
    except:
        pass
    
    delay = delay + 1
    
############################################ Translator
    if delay == 25:
        
        cv2.imwrite('test.jpg', mask)
        img=Image.open("D:/Studia/MGR/III semestr/Praca Dyplomowa/CD/Program_glowny/test.jpg")
        imgarray=np.array(img)
        
        plik=open("plik_tekstowy.txt",'w')
        for i in range(0, 150):
            for j in range (0, 150):
                plik.write(str(imgarray[i][j])+",")
                #s +=str(imgarray[i][j])+','
                #tablica.append(str(imgarray[i][j])+",")
        plik.write("\n")
        plik.flush()
        plik.close 
        
        buffor = open("plik_tekstowy.txt", "r").readline()
        
        with open('dana_wejsciowa.csv', 'w', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)  
            csvwriter.writerow([buffor])
            csvwriter.writerow([buffor])
            csvwriter.writerow([buffor])

            
        plik.close
    
        data_testing = pd.read_csv("D:/Studia/MGR/III semestr/Praca Dyplomowa/CD/Program_glowny/dana_wejsciowa.csv").as_matrix()      
        xtest = data_testing[0,0:22500]
        how_letter = clf.predict([xtest])
        print(how_letter)
        delay = 0
        
############################################    
    k = cv2.waitKey(5) & 0xFF    
    
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    
    