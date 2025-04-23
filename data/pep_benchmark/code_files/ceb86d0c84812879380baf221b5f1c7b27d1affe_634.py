import consts
import dlib
import cv2
import util
from filter import Filter2D
from hand_detector import HandDetector


class HandTracker:
    def __init__(self, options):
        self.options = options
        self.max_score = 0
        self.tracker = dlib.correlation_tracker()
        self.hand_detector = HandDetector(options)
        self.empty_frames = 0
        self.wrong_frames = 0

    def get_hand_rect(self, frame):
        frame_scaled = cv2.resize(frame, (
            self.options[consts.tracking_image_width],
            self.options[consts.tracking_image_height]))
        score, det_rel = self.hand_detector.detect_hand(frame)
        if self.max_score == 0 and score > 0:
            position = util.from_relative(det_rel, frame_scaled.shape)
            position = util.fit_rect(position, frame_scaled.shape)
            self.tracker.start_track(frame_scaled, util.to_dlib(position))
            self.max_score = score
        if self.max_score > 0:
            self.tracker.update(frame_scaled)
            position = util.fit_rect(util.from_dlib(self.tracker.get_position()), frame_scaled.shape)
            pos_rel = util.to_relative(position, frame_scaled.shape)
            if score <= 0:
                self.empty_frames += 1
                if self.empty_frames >= self.options[consts.empty_frames]:
                    self.max_score = 0
                    self.empty_frames = 0
            else:
                self.empty_frames = 0
            if util.are_different_locations(pos_rel, det_rel):
                self.wrong_frames += 1
                if self.wrong_frames == 5:
                    self.wrong_frames = 0
                    self.wrong_frames = 0
                    position = util.from_relative(det_rel, frame_scaled.shape)
                    position = util.fit_rect(position, frame_scaled.shape)
                    self.tracker.start_track(frame_scaled, util.to_dlib(position))
                    self.max_score = score
            else:
                self.wrong_frames = 0
            rect = util.from_relative(pos_rel, frame.shape)
            hand_rect = util.to_square(rect, True)
            return hand_rect
        else:
            return None