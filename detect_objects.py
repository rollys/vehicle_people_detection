# -*- coding: utf-8 -*-
import cv2
import argparse

class ObjectsDetection:
    def __init__(self, video_path, cascade_vehicle_path, cascade_person_path, enabled_vehicle, enabled_people,
                 enabled_hog_people):
        self.cascade_vehicle_src = cascade_vehicle_path
        self.cascade_people_src = cascade_person_path
        self.enabled_vehicle = enabled_vehicle
        self.enabled_people = enabled_people
        self.enabled_hog_people = enabled_hog_people
        self.video_src = video_path
        self.cars_num_frame_before = 0
        self.cars_num_frame = 0
        self.cars_num_total = 0
        self.people_num_frame_before = 0
        self.people_num_frame = 0
        self.people_num_total = 0

    def video_capture(self):
        vc = cv2.VideoCapture(self.video_src)
        car_cascade = cv2.CascadeClassifier(self.cascade_vehicle_src)
        person_cascade = cv2.CascadeClassifier(self.cascade_people_src)

        # inicializando el HOG descriptor/person detector
        if int(self.enabled_hog_people) == 1:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        while True:

            ret, img = vc.read()
            if (type(img) == type(None)):
                break
            #Convierte el frame a tonos grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if int(self.enabled_vehicle) == 1:
                cars = car_cascade.detectMultiScale(gray, 1.1, 1)
                self.cars_elements(cars, img)
                self.cars_count(img)

            if int(self.enabled_people) == 1 and int(self.enabled_hog_people) != 1:
                people = person_cascade.detectMultiScale(gray, 1.1, 1)
                self.people_elements(people, img)
                self.people_count(img)

            if int(self.enabled_hog_people) == 1:
                (people, weights) = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
                self.people_elements(people, img, (255, 0, 0))
                self.people_count(img)

            # self.haar_iterate(cars, img, (0, 0, 255))
            # self.haar_iterate(people, img, (0, 255, 0))


            cv2.imshow('Captura de autos y personas', img)

            edges = cv2.Canny(gray, 50, 200)
            cv2.imshow('Canny', edges)

            if cv2.waitKey(33) == 27:
                break

        cv2.destroyAllWindows()

    def haar_iterate(self, pattern, capture, color):
        for (x, y, w, h) in pattern:
            cv2.rectangle(capture, (x, y), (x + w, y + h), color, 2)

    def cars_elements(self, pattern, capture):
        self.cars_num_frame_before = self.cars_num_frame
        self.cars_num_frame = 0
        for (x, y, w, h) in pattern:
            cv2.rectangle(capture, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.cars_num_frame +=1

        if self.cars_num_frame > self.cars_num_frame_before:
            self.cars_num_total += self.cars_num_frame - self.cars_num_frame_before

    def people_elements(self, pattern, capture, color=(0, 255, 0)):
        self.people_num_frame_before = self.people_num_frame
        self.people_num_frame = 0
        for (x, y, w, h) in pattern:
            cv2.rectangle(capture, (x, y), (x + w, y + h), color, 2)
            self.people_num_frame +=1

        if self.people_num_frame > self.people_num_frame_before:
            self.people_num_total += self.people_num_frame - self.people_num_frame_before

    def cars_count(self, capture):
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(capture, 'Autos', (10, 200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(capture, (10, 205), (60, 205), (255, 255, 255), 1)
        cv2.putText(capture, 'Frame: ' + str(self.cars_num_frame), (10, 220), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(capture, 'Total: ' + str(self.cars_num_total), (10, 235), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    def people_count(self, capture):
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(capture, 'Personas', (10, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(capture, (10, 155), (90, 155), (255, 255, 255), 1)
        cv2.putText(capture, 'Frame: ' + str(self.people_num_frame), (10, 170), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(capture, 'Total: ' + str(self.people_num_total), (10, 185), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="./videos/video5.avi", help="Ruta del video a analizar")
ap.add_argument("-cv", "--cascade-vehicle", default="./haarcascade/cars.xml", help="Ruta del clasificador de autos")
ap.add_argument("-cp", "--cascade-people", default="./haarcascade/people2.xml", help="Ruta del clasificador de personas")
ap.add_argument("-ev", "--enabled-vehicle", default=1, help="Habilitar la detección de autos")
ap.add_argument("-ep", "--enabled-people", default=1, help="Habilitar la detección de personas")
ap.add_argument("-ehp", "--enabled-hog-people", default=0, help="Habilitar la detección de personas con HOG")
args = vars(ap.parse_args())

od = ObjectsDetection(args['video'], args['cascade_vehicle'], args['cascade_people'], args['enabled_vehicle'], args['enabled_people'], args['enabled_hog_people'])
od.video_capture()
