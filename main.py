import cv2
import mediapipe as mp
import time
import math


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

circleCount = 0
pTime = 0
cTime = 0


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lmList = []

    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for ids, lm in enumerate(handLandmarks.landmark):

                # print(id, lm)

                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append((id, cx, cy))
                # print(id , cx, cy)
                # print(id)
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

        if len(lmList) >= 9:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            distance = math.hypot(x2 - x1, y2 - y1)

            if distance < 30:
                cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)
                circleCount += 1


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}, Circle Count: {int(circleCount)}', (10,70), cv2.FILE_NODE_INT, 2, (255,255,255), 2)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break



cap.release()
cv2.destroyAllWindows()

print("Total Count: {}".format(circleCount))