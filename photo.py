import cv2 as cv
import time, shutil, os
import threading as th
import mediapipe as mp
import math, sys
import mouse as ms
from PIL import Image, ImageOps


# ------------------------------------- funcam calss ---------------------------------

class funcam:
    def __init__(self):
        self.camera = cv.VideoCapture(0)

    def testacsess(self):
        if self.camera.isOpened():
            r, p = self.camera.read()
        return p

    def exitcam(self):
        cv.destroyAllWindows()

    def livecam(self, stopkey="q"):
        while True:
            x = self.testacsess()
            cv.imshow('cam', x)
            if cv.waitKey(1) & 0xFF == ord(stopkey):
                break

    def hiddencam(self, stopkey='q', snaptime=3):
        while True:
            x = self.testacsess()
            cv.imwrite("test.jpg", x)
            time.sleep(snaptime)
            if cv.waitKey(1) & 0xFF == ord(stopkey):
                break
        self.exitcam()

    def graycam(self, stopkey="q"):
        while True:
            cv.waitKey(3)
            x = self.testacsess()
            x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
            cv.imshow('cam', x)
            if cv.waitKey(3) & 0xFF == ord(stopkey):
                break

    def recorder(self, outputname="output.avi", stopkey="q"):
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(outputname, fourcc, 20.0, (640, 480))
        while self.camera.isOpened():
            r, p = self.camera.read()
            if r:
                # p = cv.flip(p,0)
                out.write(p)
                cv.imshow('cam', p)
                if cv.waitKey(1) & 0xFF == ord(stopkey):
                    break
        # cam.release()
        out.release()
        cv.destroyAllWindows()

    def filter_record(self, filter=None, stopkey="q", outputname="output.avi"):
        f = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(outputname, f, 20.0, (640, 480))
        filters = {None: cv.COLOR_BGRA2BGR,
                   "gray": cv.COLOR_BGR2GRAY,
                   "hill": cv.COLOR_BGR2HLS,
                   "hothill": cv.COLOR_BGR2HSV_FULL,
                   "yellowlight": cv.COLOR_BGR2LAB,
                   "rgb": cv.COLOR_BGR2RGB}
        while self.camera.isOpened():
            r, p = self.camera.read()
            if r:
                p = cv.cvtColor(p, filters[filter])
                out.write(p)
                cv.imshow('cam', p)
                if cv.waitKey(1) & 0xFF == ord(stopkey):
                    break


# ------------------------------------- funcam calss ---------------------------------


# ------------------------------------- photograph calss ---------------------------------

class photograph:
    filters = {None: cv.COLOR_BGRA2BGR,
               "gray": cv.COLOR_BGR2GRAY,
               "hill": cv.COLOR_BGR2HLS,
               "hothill": cv.COLOR_BGR2HSV_FULL,
               "yellowlight": cv.COLOR_BGR2LAB,
               "rgb": cv.COLOR_BGR2RGB}

    def __init__(self):
        self.cam = cv.VideoCapture(0)
        self.output = "output"
        self.filter = None
        self.input = "inp"

    def camera(self):
        if self.cam.isOpened():
            r, p = self.cam.read()
        return p

    def seeface(self, p):
        face_f2 = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
        # gray = cv.cvtColor(p,cv.COLOR_BGR2GRAY)
        faces2 = face_f2.detectMultiScale(p, 1.3, 3)
        for (x, y, w, h) in faces2:
            cv.rectangle(p, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return p

    def see_eye(self, p):
        eye_f = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
        eyes = eye_f.detectMultiScale(p, 1.3, 3)
        for (x, y, w, h) in eyes:
            cv.rectangle(p, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return p

    def show(self, photo, windowname="wincam", f=None):
        try:
            data = cv.imread(photo)
            data = cv.cvtColor(data, self.filters[f])
        except:
            data = photo
            data = cv.cvtColor(data, self.filters[f])
        cv.imshow(windowname, data)
        k = cv.waitKey(1) & 0x_ff
        return k

    # def show(self):
    #     cv.imshow("test",self.inp)

    def cvdata(self, data, p):
        eye_f = cv.CascadeClassifier(cv.data.haarcascades + data)
        eyes = eye_f.detectMultiScale(p, 1.3, 3)
        for (x, y, w, h) in eyes:
            cv.rectangle(p, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return p

    def test(self):
        while True:
            x = self.camera()
            o = self.see_eye(x)
            z = self.show(o)
            if z == ord('q'):
                break


# ------------------------------------- photograph calss ---------------------------------


# ------------------------------------- handDetector calss ---------------------------------

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=False, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):  # Finds all hands in a frame
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):  # Fetches the position of hands
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                             (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):  # Checks which fingers are up
        fingers = []  # [0,0,0,0,0]
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):  # Finds distance between two fingers
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


# ------------------------------------- handDetector calss ---------------------------------


# ------------------------------------- Facetracker calss ---------------------------------

class Facetracker:
    def __init__(self, draw=True, pen="circle"):
        self.draw = draw
        self.pen = pen

    def Facefinder(self, img):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.draw:
            res = face_mesh.process(rgb_img)
            height, width, _ = rgb_img.shape
            for facial_landmarks in res.multi_face_landmarks:
                for i in range(0, 468):
                    # if i == 159 or i == 33 or i == 145 or i == 133 or  i == 263 or i == 362 or  i == 374 or i == 263 :
                    #     pt1 = facial_landmarks.landmark[i]
                    #     x = int(pt1.x * width)
                    #     y = int(pt1.y * height)
                    #     cv2.circle(image, (x, y), 1, (86,255,0), -1)
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    cv.circle(img, (x, y), 1, (86, 255, 0), -1)
            return img

    def mouthTracker(self, img):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        mouth = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
            375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
            324, 318, 402, 317, 14, 87, 178, 88, 95, 78
        ]
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.draw:
            res = face_mesh.process(rgb_img)
            height, width, _ = rgb_img.shape
            for facial_landmarks in res.multi_face_landmarks:
                for i in range(0, 468):
                    if mouth.count(i) > 0:
                        pt1 = facial_landmarks.landmark[i]
                        print(pt1.x)
                        x = int(pt1.x * width)
                        y = int(pt1.y * height)
                        cv.circle(img, (x, y), 1, (86, 255, 0), -1)
                    # pt1 = facial_landmarks.landmark[i]
                    # x = int(pt1.x * width)
                    # y = int(pt1.y * height)
                    # cv.circle(img, (x, y), 1, (86,255,0), -1)
        return img

    def eyeTracker(self, img):
        eyes = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374,
                373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        rgb_imge = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.draw:
            res = face_mesh.process(rgb_imge)
            h, w, _ = rgb_imge.shape
            for facial_landmarks in res.multi_face_landmarks:
                for i in range(0, 468):
                    if eyes.count(i) > 0:
                        pt1 = facial_landmarks.landmark[i]
                        x = int(pt1.x * w)
                        y = int(pt1.y * h)
                        cv.circle(img, (x, y), 1, (86, 255, 0), -1)
        return img

    def mousevove(self, img):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        rgb_imge = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = face_mesh.process(rgb_imge)
        h, w, _ = rgb_imge.shape
        for facial_landmarks in res.multi_face_landmarks:
            for i in range(0, 468):
                if i == 7:
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * w)
                    y = int(pt1.y * h)
                    cv.circle(img, (x, y), 1, (86, 255, 0), -1)
                    break
            ms.move(x, y)
            print(x, y)
            return img

    def faceRejaction(self, img):
        try:
            data = cv.imread(img)
        except:
            data = img
        graydata = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
        facecase = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
        fces = facecase.detectMultiScale(
            graydata,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        for (x, y, w, h) in fces:
            outdata = data[y:y + h, x:x + w]
            # cv.rectangle(data, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
            # cv.imwrite(str(w) + str(h)+"_test.jpg" , outdata)
            # break
        return outdata

    def findsame(self, img, imgt):
        print(f"inputs : {img} and {imgt}")
        try:
            s = cv.imread(img)
            print("s readed")
        except:
            s = img
        try:
            st = cv.imread(imgt)
            print("st readed")
        except:
            st = img
        imgray = cv.cvtColor(s, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(st, contours, -1, (0, 255, 0), 1)
        print(len(contours))
        return len(contours)

    def samep(self, img1, img2):
        sameself = self.findsame(img1, img1)
        sameother = self.findsame(img2, img1)
        print(f"same points {sameother}")
        return int((sameother * 100) / sameself)


# ------------------------------------- Facetracker calss ---------------------------------


# ------------------------------------- photoshop calss ---------------------------------


class photoshop:
    def __init__(self, input):
        self.filename = input
        self.input = Image.open(input)
        self.output = "outputfiel.png"

    def writeimg(self, path=None, magedata=None):
        if path == None and magedata == None:
            self.input.save(self.output)
        elif path == None and not magedata == None:
            magedata.save(self.output)
        elif not path == None and magedata == None:
            self.input.save(path)
        else:
            magedata.save(path)

    def RGB(self, r, g, b, save=False, path=None):
        data = self.input.getdata()
        rgb = [(d[0] * r, d[1] * g, d[2] * b) for d in data]
        self.input.putdata(rgb)
        if save:
            self.writeimg(path)
        else:
            self.input.show()

    def cut(self, left, top, right, bottom, save=False, path=None):
        data = self.input.crop((left, top, right, bottom))
        if save:
            self.writeimg(path, data)
        else:
            data.show()

    def routate(self, deg, fit=True, save=False, path=None):
        data = self.input.rotate(deg, expand=fit)
        if save:
            self.writeimg(path, data)
        else:
            data.show()


# x = photoshop("4animals_modified.png")
# x.routate(30)

# ----------------------------------------
#         lablatoar


# f = Facetracker()
# s = photograph()

# data = f.faceRejaction("13970320274_01308600-3931256446.jpg")
# # while True:
# #     s.show(data)

# base = "baseappel.jpg" , "baseorange.png"
# l = ["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg","6.png","7.jpg","8.jpg","9.jpg"]
# count = 0
# smaefound = False
# x = f.samep("9.jpg","5.jpg")
# print(f"output: {x}")
# for i in l:
#     smaefound = False
#     count += 1
#     os.mkdir(f"data{count}")

#     for j in l:
#         try:
#             s = f.findsame(j,i)
#         except:
#             pass
#         print(f"\n\n{(s*100)/700}\n\n")
#         if (s*100)/100 > 40:
#             try:
#                 shutil.move(j,f"./data{count}/{j}")
#                 # l.remove(j)
#                 print(j)
#                 smaefound = True
#             except Exception as e:
#                 print(e)
#     if smaefound :
#         try:
#             shutil.move(i,f"./data{count}/{i}")
#         except:
#             pass
# l.remove(i)

# for i


# cam = photograph()
# facese = Facetracker()
# # while True:
# camera = cam.camera()
# try:
#     x = facese.faceRejaction(camera)
# except:
#     x = camera

# exitkey = cam.show(x)


# im = cv.imread("test.jpg")
# while True:
#     t = x.show("test.jpg")
#     if t == ord("q"):
#         break


# برای ربات


# x = Facetracker()
# y = photograph()
# while True


# import cv2

# # Load images as grayscale
# image1 = cv2.imread("1.jpg", 0)
# image2 = cv2.imread("2.jpg", 0)

# # Calculate the per-element absolute difference between
# # two arrays or between an array and a scalar
# diff = 255 - cv2.absdiff(image1, image2)

# cv2.imshow('diff', diff)
# cv2.waitKey()


