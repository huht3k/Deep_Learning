# coding: utf-8
#!/usr/bin/env python

"""
demo interface for object detection of video
two tasks, two threads
    read video and detetion
    show the results on video, do the recording etc.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
import sys
import datetime

from threading import Thread
import time
import copy

import configparser
from queue import Queue


class FileVideoStream:
    def __init__(self, path, transform=None, args=[], queue_size=128):
        self.count = 0
        self.stream = cv2.VideoCapture(path)

        # self.stream.set(cv2.CAP_PROP_FPS, 60)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.cnt_for_video = 0
        self.cnt_for_image = 0

        self.getConfig()
        self.small_width = int(self.size[0] / self.RED)
        self.small_hight = int(self.size[1] / self.RED)
        self.small = np.zeros((self.ROW*self.COL, self.small_hight, self.small_width, 3), dtype='uint8')
        self.small_num = 0
        self.small_full = False

        self.stopped = False
        # the unified interface of detection such as faster rcnn, yolo, ssd etc
        self.transform = transform
        self.args = args

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def getConfig(self):
        conf = configparser.ConfigParser()
        conf.read("./video_objects_detect_interface.ini")
        self.base_name = str(conf.get("Path", 'base_name'))
        interval_video = conf.get("Name", 'interval_video')
        self.interval_video = int(interval_video)
        interval_image = conf.get("Name", 'interval_image')
        self.interval_image = int(interval_image)
        self.ROW = int(conf.get("Size", 'ROW'))
        self.COL = int(conf.get("Size", 'COL'))
        self.RED = int(conf.get("Size", 'RED'))
        self.X0 = int(conf.get("Size", 'X0'))
        self.Y0 = int(conf.get("Size", 'Y0'))


    def update(self):
        # keep looping infinitely
        while True:
            if self.stopped:
                break
            if self.cnt_for_video % self.interval_video == 0:
                self.cnt_for_video = 0
                now = datetime.datetime.now()
                dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
                # video_path = "recorders/" + dt_string + ".avi"
                video_path = self.base_name + dt_string + ".avi"
                videoWriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.size)

            if self.cnt_for_image % self.interval_image == 0:
                self.cnt_for_image = 0
                now = datetime.datetime.now()
                dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
                image_path_base = self.base_name + dt_string

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stopped = True
                    break

                self.cnt_for_video += 1
                videoWriter.write(frame)

                self.count += 1
                b_detected = False
                if self.transform and (self.count % 4 == 0):
                    self.count = 0
                    original_frame = copy.copy(frame)
                    frame, b_detected = self.transform(*self.args, frame)

                if b_detected:
                    self.cnt_for_image += 1
                    image_path = image_path_base + "_" + str(self.cnt_for_image) + ".jpg"
                    cv2.imwrite(image_path, original_frame)
                    self.small[self.small_num] = cv2.resize(frame, (self.small_width, self.small_hight))
                    self.small_num = (self.small_num + 1) % (self.ROW * self.COL)
                    if self.small_num == 0:
                        self.small_full = True

                # add the frame to the queue
                self.Q.put(frame)

            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        img = self.Q.get()

        # speed up using cycle buffer
        cur = self.small_num
        if self.small_full:
            for i in range(self.ROW-1, -1, -1):
                for j in range(self.COL-1, -1, -1):
                    cur = (cur - 1 + self.ROW * self.COL) % (self.ROW * self.COL)
                    img[(self.X0 + i*self.small_hight): (self.X0 + (i+1)*self.small_hight),
                    (self.Y0 + j*self.small_width): (self.Y0 + (j+1)*self.small_width)] = self.small[cur]
        else:
            cnt = 0
            for i in range(self.ROW):
                for j in range(self.COL):
                    if (i * self.ROW + j) > cur - 1:
                        break
                    img[(self.X0 + i * self.small_hight): (self.X0 + (i + 1) * self.small_hight),
                    (self.Y0 + j * self.small_width): (self.Y0 + (j + 1) * self.small_width)] = self.small[cnt]
                    cnt += 1
        return img

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()

    def getParameter(self):
        print("frame per second: {}".format(self.fps))
        print("size of the video: {}".format(self.size))
        print("base_name: {}".format(self.base_name))
        print("interval_video: {}".format(self.interval_video))
        print("interval_image: {}".format(self.interval_image))
        print("small image width: {}".format(self.small_width))
        print("small image height: {}".format(self.small_hight))


def dummy_detect(im):
    rst = npr.randint(0, 2, 1)
    if rst[0] == 1:
        bDetected = True
    else:
        bDetected = False

    return im, bDetected


if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read("video_objects_detect_interface.ini")
    video_path = conf.get("Input", 'video_path')
    if video_path == '0' or video_path == '1' or video_path == '2':
        video_path = int(video_path)

    fvs = FileVideoStream(video_path, transform=dummy_detect, args=[], queue_size=10)

    fvs.start()
    fvs.getParameter()

    while True:
        if fvs.stopped:
            break
        frame = fvs.read()
        if frame is not None:
            cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('video frame', frame)

            if cv2.waitKey(10) == 27:  # exit if Escape is hit
                break

    # do a bit of cleanup
    cv2.destroyAllWindows()


