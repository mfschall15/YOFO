from sort.sort import *

class tracker:
    def __init__(self):
        self.yolo_confidence = 50
        self.min_hits = 2
        self.max_age = 5

    def get_tracker(self):
        mot_tracker = Sort(max_age = self.max_age, min_hits = self.min_hits)
        return mot_tracker

class video_frame:
    def __init__(self):
        self.img = None
        self.objects = {}
    

tracker = tracker()
frame_data = video_frame()