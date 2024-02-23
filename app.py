#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import math

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

import random
# Load the emoji image
emoji_path = 'E:\\WPI_Courses\\Sem3\\CV\\week14\\thumbsup.png'
emoji = cv.imread(emoji_path, cv.IMREAD_UNCHANGED)
desired_size = (100, 100) 
emoji = cv.resize(emoji, desired_size)
emoji_alpha = emoji[:, :, 3]
emoji = emoji[:, :, :3]

emoji_path_td = 'E:\\WPI_Courses\\Sem3\\CV\\week14\\thumbsdown.png'
emoji_td = cv.imread(emoji_path_td, cv.IMREAD_UNCHANGED)
desired_size_td = (100, 100) 
emoji_td = cv.resize(emoji_td, desired_size_td)
emoji_alpha_td = emoji_td[:, :, 3]
emoji_td = emoji_td[:, :, :3]

snowflake_path = 'E:\\WPI_Courses\\Sem3\\CV\\week14\\snowflake.png' 
snowflake_img = cv.imread(snowflake_path, cv.IMREAD_UNCHANGED)

def overlay_snowflake(background, snowflake, position, scale=1.0, angle=0):
    x, y = position
    snowflake = cv.resize(snowflake, (0, 0), fx=scale, fy=scale)

    # Rotate the snowflake
    center = (snowflake.shape[1] // 2, snowflake.shape[0] // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    snowflake = cv.warpAffine(snowflake, rotation_matrix, (snowflake.shape[1], snowflake.shape[0]))

    h, w = snowflake.shape[:2]

    # Overlay boundaries
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)

    # Check if the overlay area is valid
    if y1 >= y2 or x1 >= x2:
        return background

    # Crop the overlay image and alpha mask to fit the background
    snowflake_cropped = snowflake[:y2-y1, :x2-x1]
    alpha_snowflake = snowflake_cropped[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_snowflake

    # Overlay snowflake
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = alpha_snowflake * snowflake_cropped[:, :, c] + alpha_background * background[y1:y2, x1:x2, c]

    return background

# Function to draw fog effect
def apply_fog_effect(image, intensity=0.2):
    fog = np.full(image.shape, 255, dtype=image.dtype)
    cv.addWeighted(image, 1 - intensity, fog, intensity, 0, image)
    return image

# Initialize snowflakes data
num_snowflakes = 50  # Adjust the number of snowflakes
snowflakes = [{
    'position': (random.randint(0, 640), random.randint(-100, -10)),
    'scale': random.uniform(0.015, 0.08),
    'speed': random.randint(1, 4),
    'angle': random.randint(0, 360) 
} for _ in range(num_snowflakes)]

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    if y1 >= y2 or x1 >= x2:
        # If the calculated positions result in no overlap, return the image as is
        return img

    # Adjust the overlay and the alpha mask to match the ROI shape
    overlay_cropped = img_overlay[0:y2-y1, 0:x2-x1]
    alpha_cropped = alpha_mask[0:y2-y1, 0:x2-x1][..., np.newaxis] / 255.0

    if alpha_cropped is not None:
        overlay = alpha_cropped * overlay_cropped
        background = (1.0 - alpha_cropped) * img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = overlay + background
    else:
        img[y1:y2, x1:x2] = overlay_cropped

    return img

def overlaytd_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    if alpha_mask is not None:
        # Reshape the alpha mask to match the color image shape
        alpha = alpha_mask[..., np.newaxis] / 255.0
        overlay = alpha * img_overlay[0:y2-y1, 0:x2-x1]
        background = (1.0 - alpha) * img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = overlay + background
    else:
        img[y1:y2, x1:x2] = img_overlay[0:y2-y1, 0:x2-x1]

    return img

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def draw_heart(image, center, size, color):
    # Calculate points for the heart shape
    x, y = center
    w, h = size
    top_left = (x - w // 2, y)
    bottom_point = (x, y + h // 2)

    # Points for the left and right arcs of the heart
    left_arc = [top_left, (x - w // 2, y - h // 3), (x, y - h // 2)]
    right_arc = [(x, y - h // 2), (x + w // 2, y - h // 3), (x + w // 2, y)]

    # Draw two halves of the heart
    cv.fillPoly(image, [np.array([top_left, *left_arc, bottom_point])], color)
    cv.fillPoly(image, [np.array([top_left, *right_arc, bottom_point])], color)

    return image

def draw_hearts_effect(image, hand_center, num_hearts=5, max_offset=50):
    for _ in range(num_hearts):
        # Randomly offset the heart's position
        offset_x = random.randint(-max_offset, max_offset)
        offset_y = random.randint(-max_offset, max_offset)
        heart_center = (hand_center[0] + offset_x, hand_center[1] + offset_y)

        # Random size and color for each heart
        size = (random.randint(20, 40), random.randint(20, 40))  # width, height
        color = (0, 0, 255)  # Red color

        image = draw_heart(image, heart_center, size, color)
    return image


def draw_balloons_effect(image):
    # Logic to draw balloons effect
    # Drawing simple colored circles as balloons
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(3):
        cv.circle(image, (100 + i * 100, 300), 50, colors[i], -1)
    return image

def draw_thumbs_up_effect(image):
    # Logic to draw thumbs up effect
    cv.rectangle(image, (150, 250), (170, 350), (0, 255, 0), -1)
    
    # Top part of the thumb
    cv.rectangle(image, (170, 150), (250, 170), (0, 255, 0), -1)
    cv.circle(image, (250, 170), 20, (0, 255, 0), -1)

    # Bottom part of the hand
    cv.rectangle(image, (170, 250), (230, 350), (0, 255, 0), -1)

    return image

def draw_confetti_effect(image):
    h, w = image.shape[:2]  # Get the height and width of the image
    num_confetti = 50  # Increase the number of confetti pieces for full coverage

    for i in range(num_confetti):
        x, y = random.randint(0, w), random.randint(0, h)  # Random position within the whole screen
        size = random.randint(5, 10)  # Random size of the confetti
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
        cv.rectangle(image, (x, y), (x + size, y + size), color, -1)  # Draw the confetti

    return image

def generate_laser_beam(img, color, thickness=2):
    start_point = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
    end_point = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
    cv.line(img, start_point, end_point, color, thickness)

def draw_laser_effect(image, position, color=(0, 255, 0), thickness=2):
    """
    Draws two laser beams originating from the bottom corners of the screen,
    connecting at a given position.
    
    Parameters:
        image (numpy.ndarray): The image on which to draw the laser beams.
        position (tuple): The current position (x, y) for the connecting point of the laser beams.
        color (tuple): The color of the laser beam in BGR format (blue, green, red).
        thickness (int): The thickness of the laser beam.
    """
    height, width = image.shape[:2]
    cv.line(image, (0, height), position, color, thickness)
    cv.line(image, (width, height), position, color, thickness)

def overlay_image_alpha_cylindrical(img, img_overlay, pos, alpha_mask):
    x, y = pos
    h, w = img.shape[:2]
    overlay_h, overlay_w = img_overlay.shape[:2]

    # Calculate positions
    y1, y2 = max(0, y), min(h, y + overlay_h)
    x1, x2 = x, x + overlay_w

    # Check if the emoji goes beyond the screen on the right
    if x2 > w:
        x2_left = w
        x2_right = x2 - w
    else:
        x2_left = x2
        x2_right = 0

    # Left part (visible on screen)
    if alpha_mask is not None:
        alpha_left = alpha_mask[..., np.newaxis][0:y2-y1, 0:x2_left-x1] / 255.0
        overlay_left = alpha_left * img_overlay[0:y2-y1, 0:x2_left-x1]
        background_left = (1.0 - alpha_left) * img[y1:y2, x1:x2_left]
        img[y1:y2, x1:x2_left] = overlay_left + background_left
    else:
        img[y1:y2, x1:x2_left] = img_overlay[0:y2-y1, 0:x2_left-x1]

    # Right part (wrapped to the left side)
    if x2_right > 0:
        if alpha_mask is not None:
            alpha_right = alpha_mask[..., np.newaxis][0:y2-y1, -x2_right:] / 255.0
            overlay_right = alpha_right * img_overlay[0:y2-y1, -x2_right:]
            background_right = (1.0 - alpha_right) * img[y1:y2, 0:x2_right]
            img[y1:y2, 0:x2_right] = overlay_right + background_right
        else:
            img[y1:y2, 0:x2_right] = img_overlay[0:y2-y1, -x2_right:]

    return img


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    # Example variables to control the laser animation
    laser_x = 0
    laser_y = cap_height // 2  # Start from the middle of the screen vertically
    laser_dx = 5  # Change in position per frame
    
    angle = 0  # Ensure this is outside your main loop
    angle1 = 60
    angle2 = 120
    angle3 = 180
    angle4 = 240  # Ensure this is outside your main loop
    anglea = 0
    angle1a = -60
    angle2a = -120
    angle3a = -180
    angle4a = -240 
    angle_increment = math.radians(2)  # Ensure this is a meaningful value

    # Define a list of colors to cycle through (BGR format)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    color_index = 0  # Start with the first color

    # Define how many frames to show each color
    frames_per_color = 100  # Adjust this value as needed

    # Initialize a frame counter
    frame_counter = 0    

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        thumbs_up_count = 0
        thumbs_up_detected =  False
        thumbs_down_detected =  False
        snowfall_active = False
        laser_active = False
        screen_center_x = cap_width // 2

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                thumb_tip_x, thumb_tip_y = landmark_list[4][0], landmark_list[4][1]
                pinky_tip_x, pinky_tip_y = landmark_list[20][0], landmark_list[20][1]  # Pinky tip

                # Calculate mean position
                mean_x, mean_y = (thumb_tip_x + pinky_tip_x) // 2, (thumb_tip_y + pinky_tip_y) // 2
                mean_pos = (mean_x, mean_y)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Logic for triggering effects based on gesture
                if keypoint_classifier_labels[hand_sign_id] == 'Hearts':
                    debug_image = draw_hearts_effect(debug_image, mean_pos)
                elif keypoint_classifier_labels[hand_sign_id] == 'Peace Gesture':
                    debug_image = draw_balloons_effect(debug_image)
                elif keypoint_classifier_labels[hand_sign_id] in('ThumbsUp' or 'Fireworks'):
                    thumbs_up_count +=1
                    thumbs_up_detected = True
                elif keypoint_classifier_labels[hand_sign_id] == 'Confetti':
                    debug_image = draw_confetti_effect(debug_image)
                elif keypoint_classifier_labels[hand_sign_id] == 'ThumbsDown':
                    thumbs_down_detected = True
                elif keypoint_classifier_labels[hand_sign_id] == 'OK':
                    snowfall_active  = True
                elif keypoint_classifier_labels[hand_sign_id] == 'Lasers':
                    laser_active  = True
                    #debug_image = draw_laser_effect(debug_image)

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                #debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                #debug_image = draw_landmarks(debug_image, landmark_list)
                # debug_image = draw_info_text(
                #     debug_image,
                #     brect,
                #     handedness,
                #     keypoint_classifier_labels[hand_sign_id],
                #     point_history_classifier_labels[most_common_fg_id[0][0]],
                # )
        else:
            point_history.append([0, 0])

        if thumbs_up_detected:  
            if thumbs_up_count == 2:
                # Trigger fireworks effect
                pass
            elif thumbs_up_count == 1:
                emoji_x = thumb_tip_x + 300
                if thumb_tip_x > screen_center_x -75:
                    # If the thumb is on the right side, display emoji on the left
                    emoji_x = thumb_tip_x - 350 - emoji.shape[1]
                emoji_x = max(0, min(emoji_x, cap_width - emoji.shape[1]))  # Ensure within bounds
                emoji_y = max(0, min(thumb_tip_y - 150, cap_height - emoji.shape[0]))  # Ensure within bounds
                debug_image = overlay_image_alpha(debug_image, emoji, (emoji_x, emoji_y), emoji_alpha)

        elif thumbs_down_detected:
                emoji_x_td = thumb_tip_x + 300
                if thumb_tip_x > screen_center_x - 75:
                    # If the thumb is on the right side, display emoji on the left
                    emoji_x_td = thumb_tip_x - 350 - emoji_td.shape[1]
                emoji_x_td = max(0, min(emoji_x_td, cap_width - emoji_td.shape[1]))  # Ensure within bounds
                emoji_y_td = max(0, min(thumb_tip_y - 250, cap_height - emoji_td.shape[0]))  # Ensure within bounds
                debug_image = overlay_image_alpha(debug_image, emoji_td, (emoji_x_td, emoji_y_td), emoji_alpha_td)

        elif snowfall_active:
            for snowflake in snowflakes:
                debug_image = overlay_snowflake(debug_image, snowflake_img, snowflake['position'], snowflake['scale'])
                snowflake['position'] = (snowflake['position'][0], snowflake['position'][1] + snowflake['speed'])
                snowflake['angle'] = (snowflake['angle'] + 5) % 360  # Update angle
        
                if snowflake['position'][1] > debug_image.shape[0]:
                    snowflake['position'] = (random.randint(0, 640), random.randint(-100, -10))
                    snowflake['scale'] = random.uniform(0.015, 0.1)
                    snowflake['speed'] = random.randint(1, 4)
                    snowflake['angle'] = random.randint(0, 360)

            debug_image = apply_fog_effect(debug_image, intensity=0.175)  # Adjust intensity for desired fog effect

        elif laser_active:
            # laser_x += laser_dx
            # laser_y += math.sqrt((cap_width+cap_height)**2 - laser_x**2)
            # print(laser_y)

            # if laser_x >= cap_width or laser_x <= 0:
            #     laser_dx *= -1  # Change direction when hitting screen bounds
            # if laser_y >= cap_height or laser_y <= 0:
            #     laser_y -= 1  # Change direction when hitting screen bounds
            # Example values for the center and radius of the arch
            center_x, center_y = cap_width/2 , cap_height
            radius = cap_height   # Example radius
            print(center_x, center_y,cap_width,cap_height)
            # Increment angle for movement
            angle += angle_increment
            laser_x = center_x + radius * math.cos(angle)
            laser_y = center_y + radius * math.sin(angle) 
            if laser_x >= cap_width or laser_x <= 0:
                laser_dx *= -1  
            angle1 += angle_increment
            laser_x1 = center_x + radius * math.cos(angle1)
            laser_y1 = center_y + radius * math.sin(angle1) 

            angle2 += angle_increment
            laser_x2 = center_x + radius * math.cos(angle2)
            laser_y2 = center_y + radius * math.sin(angle2) 

            angle3 += angle_increment
            laser_x3 = center_x + radius * math.cos(angle3)
            laser_y3 = center_y + radius * math.sin(angle3) 

            angle4 += angle_increment
            laser_x4 = center_x + radius * math.cos(angle4)
            laser_y4 = center_y + radius * math.sin(angle4) 

            anglea -= angle_increment
            laser_xa = center_x + radius * math.cos(anglea)
            laser_ya = center_y + radius * math.sin(anglea) 
            if laser_x >= cap_width or laser_x <= 0:
                laser_dx *= -1  

            angle1a -= angle_increment
            laser_x1a = center_x + radius * math.cos(angle1a)
            laser_y1a = center_y + radius * math.sin(angle1a) 

            angle2a -= angle_increment
            laser_x2a = center_x + radius * math.cos(angle2a)
            laser_y2a = center_y + radius * math.sin(angle2a) 

            angle3a -= angle_increment
            laser_x3a = center_x + radius * math.cos(angle3a)
            laser_y3a = center_y + radius * math.sin(angle3a) 

            angle4a -= angle_increment
            laser_x4a = center_x + radius * math.cos(angle4a)
            laser_y4a = center_y + radius * math.sin(angle4a)   

            # Increment frame counter
            frame_counter += 1

            # Check if it's time to switch to the next color
            if frame_counter >= frames_per_color:
                color_index = (color_index + 1) % len(colors)  # Move to the next color, loop back to the start if necessary
                frame_counter = 0  # Reset the frame counter

            # # Use updated position to draw lasers
            draw_laser_effect(debug_image, (int(laser_x), int(laser_y)),colors[color_index])
            draw_laser_effect(debug_image, (int(laser_x1), int(laser_y1)),colors[color_index])
            #draw_laser_effect(debug_image, (int(laser_x2), int(laser_y2)),colors[color_index])
            draw_laser_effect(debug_image, (int(laser_x3), int(laser_y3)),colors[color_index])
            #draw_laser_effect(debug_image, (int(laser_x4), int(laser_y4)),(255, 0, 0))
            draw_laser_effect(debug_image, (int(laser_xa), int(laser_ya)),colors[color_index])
            #draw_laser_effect(debug_image, (int(laser_x1a), int(laser_y1a)),colors[color_index])
            draw_laser_effect(debug_image, (int(laser_x2a), int(laser_y2a)),colors[color_index])
            draw_laser_effect(debug_image, (int(laser_x3a), int(laser_y3a)),colors[color_index])
            #draw_laser_effect(debug_image, (int(laser_x4a), int(laser_y4a)),(255, 0, 0))

        debug_image = draw_point_history(debug_image, point_history)
        #debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
