import cv2
import mediapipe as mp
import mouse as ms
from screeninfo import get_monitors
import mouse as ms

# Get information about all monitors
monitors = get_monitors()

# Assuming you want information about the primary monitor
primary_monitor = monitors[0]

# Get screen width and height
sw = int(primary_monitor.width)
sh = int(primary_monitor.height)

Left_point = [0,sh/2]
Right_point = [sw,sh/2]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(sw,sh))
        image = cv2.flip(image, 1)

        results = hands.process(image)

      
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
        fingerCount = 0

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label
  
                handLandmarks = []

                for landmarks in hand_landmarks.landmark:
                    handLandmarks.append([landmarks.x, landmarks.y])

                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCount += 1
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    fingerCount += 1
                    

              
                if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                    fingerCount += 1    
                if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle finger
                    fingerCount += 1
                if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring finger
                    fingerCount += 1
                if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                    fingerCount += 1

                 
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                    )

        height, width, _ = image.shape
        cv2.putText(image, str(fingerCount), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
        if fingerCount > 7:
          
          ms.move(int(handLandmarks[8][0]*sw),int(handLandmarks[8][1]*sh),duration=0.01)
          
          cv2.circle(image,(int(handLandmarks[12][0]*sw),int(handLandmarks[12][1]*sh)),17,(0,0,255),3)



      
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
