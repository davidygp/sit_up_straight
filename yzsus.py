# import the necessary packages
import cv2
import dlib
from imutils import face_utils
import json
import math
import numpy as np
import playsound
from scipy import ndimage
import time

# Press V for verbose => show the facial landmarks
# Press Q for quiet => hide the facial landmarks
# Press S to small size the screen => only show the traffic light
# Press E to enlarge the screen => show the whole thing as per normal

# Press ESC to exit the webcam
# Press ESC to exit the last image shown (it will also disappear in 30s)

# Configuration Parameters
with open("config.json") as json_file:
    config = json.load(json_file)

processed_Xth_frame = config["processed_Xth_frame"]
frameWidth = config["frameWidth"]
frameHeight = config["frameHeight"]

eye_threshold_in_pixels = config["eye_threshold_in_pixels"]
tilt_frontback_threshold_in_pixels = config[
    "tilt_frontback_threshold_in_pixels"
]
tilt_leftright_threshold_in_degrees = config[
    "tilt_leftright_threshold_in_degrees"
]
tilt_shoulder_threshold_in_pixels = config["tilt_shoulder_threshold_in_pixels"]
overall_okay_threshold = config["overall_okay_threshold"]

wrong_posture_duration_in_seconds = config["wrong_posture_duration_in_seconds"]
seated_duration_in_seconds = config["seated_duration_in_seconds"]

soundclip_filepath = config["soundclip_filepath"]
face_landmarks_dat_filepath = config["face_landmarks_dat_filepath"]
human_pose_prototxt_filepath = config["human_pose_prototxt_filepath"]
human_pose_caffemodel_filepath = config["human_pose_caffemodel_filepath"]

"""
# Due to compute we don't process every frame but every Xth frame
processed_Xth_frame = 3

# Specify the input image dimensions
frameWidth = 960
frameHeight = 720

# Thresholds for the rule-based model
eye_threshold_in_pixels = 5
tilt_frontback_threshold_in_pixels = 20
tilt_leftright_threshold_in_degrees = 15
tilt_shoulder_threshold_in_pixels = 20
overall_okay_threshold = 0.5

# The duration in seconds before...
# The alarm is sounded for wrong posture.
wrong_posture_duration_in_seconds = 3
# The text asks you to get up and move.
seated_duration_in_seconds = 10


soundclip_filepath = "./bleepsound_cut.mp3"
face_landmarks_dat_filepath = "./shape_predictor_68_face_landmarks.dat"
human_pose_prototxt_filepath = "./pose_deploy_linevec.prototxt"
human_pose_caffemodel_filepath = "./pose_iter_440000.caffemodel"
"""


# FOR HUMAN POSE ESTIMATION #
def searchPts(prMap, prThres=0.1):
    blur = cv2.GaussianBlur(prMap, (3, 3), 0, 0)
    mask = np.uint8(blur > prThres)
    pts = []

    if cv2.__version__ == "3.4.2":
        (_, ctrs, _) = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        (ctrs, _) = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

    for ctr in ctrs:
        blobs = np.zeros(mask.shape)
        blobs = cv2.fillConvexPoly(blobs, ctr, 1)
        blob = blur * blobs

        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(blob)
        pts.append(maxLoc + (prMap[maxLoc[1], maxLoc[0]],))

    return (pts, mask)


def getAllPoints(cfMaps, imgWidth, imgHeight, cfThres=0.1, numOfKeyPts=18):
    ptGrp = []
    ptList = np.zeros((0, 3))
    idx = 0

    for keyPts in range(numOfKeyPts):
        prMap = cfMaps[0, keyPts, :, :]
        prMap = cv2.resize(prMap, (imgWidth, imgHeight))

        (pts, _) = searchPts(prMap, prThres=cfThres)

        ptWithId = []
        for pt in range(len(pts)):
            ptWithId.append(pts[pt] + (idx,))
            plList = np.vstack([ptList, pts[pt]]) # noqa : F841
            idx = idx + 1

        ptGrp.append(ptWithId)
    return (ptGrp, ptList)


# For iris extraction
def extract_iris_cm_and_centre_of_eye(
    facial_landmark3743,
    facial_landmark3844,
    facial_landmark3945,
    facial_landmark4046,
    facial_landmark4147,
    facial_landmark4248,
    frameHeight,
    frameWidth,
    gray,
):
    # Extract information about the left/right eye
    eye_region = np.array(
        [
            facial_landmark3743,
            facial_landmark3844,
            facial_landmark3945,
            facial_landmark4046,
            facial_landmark4147,
            facial_landmark4248,
        ],
        np.int32,
    )
    eye_mask = np.zeros((frameHeight, frameWidth), np.uint8)
    cv2.polylines(eye_mask, [eye_region], True, 255, 2)
    cv2.fillPoly(eye_mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=eye_mask)

    eye_min_x = np.min(eye_region[:, 0])
    eye_max_x = np.max(eye_region[:, 0])
    eye_min_y = np.min(eye_region[:, 1])
    eye_max_y = np.max(eye_region[:, 1])

    eye_grey = eye[eye_min_y:eye_max_y, eye_min_x:eye_max_x]
    eye_grey = cv2.GaussianBlur(eye_grey, (1, 1), 0)
    # TODO: TBH this threshold sucks, find a better one.
    _, eye_threshold1 = cv2.threshold(
        eye_grey, 3, 1, cv2.THRESH_BINARY_INV
    )  # Get the edges
    _, eye_threshold2 = cv2.threshold(
        eye_grey, 20, 1, cv2.THRESH_BINARY_INV
    )  # Get both the iris + the edges
    eye_threshold = eye_threshold2 - eye_threshold1

    eye_centre_block = (
        eye_min_x + int((eye_max_x - eye_min_x) / 2),
        eye_min_y + int((eye_max_y - eye_min_y) / 2),
    )
    try:
        eye_cm_iris = ndimage.measurements.center_of_mass(eye_threshold.T)
        eye_cm_iris = (
            eye_min_x + int(eye_cm_iris[0]),
            eye_min_y + int(eye_cm_iris[1]),
        )

        output = [eye_centre_block, eye_cm_iris]
    except Exception:
        output = [eye_centre_block, eye_centre_block]
    return output


def angle_point9_point_28(point9, point28):
    """
    1) Get length of line between the 2 points
    2) Find the point of point28 projected onto a vertical line from point9
    3) Find
    """

    length1 = math.sqrt(
        (point28[0] - point9[0]) ** 2 + (point28[1] - point9[1]) ** 2
    )

    projected28 = (point9[0], point28[1])
    lengthX = math.sqrt(
        (projected28[0] - point9[0]) ** 2 + (projected28[1] - point9[1]) ** 2
    )

    return math.acos(lengthX / length1) * 180 / math.pi


def main():
    # FOR FACIAL LANDMARK ESTIMATION #
    # initialize dlib's face detector (HOG-based) and then create the facial
    # landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmarks_dat_filepath)

    # FOR HUMAN POSE ESTIMATION #
    print("[INFO] loading human pose predictor...")
    # Read the network into Memory
    net = cv2.dnn.readNetFromCaffe(
        human_pose_prototxt_filepath, human_pose_caffemodel_filepath
    )

    # For counting of duration in general
    total_count = 0
    shown_count = 0

    eyes_not_front_count = 0
    eyes_okay_count = 0

    shoulder_slanted_left_count = 0
    shoulder_slanted_right_count = 0
    shoulder_okay_count = 0

    head_too_left_count = 0
    head_too_right_count = 0
    head_too_back_count = 0
    head_too_front_count = 0
    head_okay_count = 0

    # For counting of duration for alarm purposes
    tmp_shoulder_slanted_left_count = 0
    tmp_shoulder_slanted_right_count = 0

    tmp_head_too_left_count = 0
    tmp_head_too_right_count = 0
    tmp_head_too_back_count = 0
    tmp_head_too_front_count = 0

    # For the display options
    verbose = False
    small_sized = False
    debug = False

    t_start = time.time()
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()

        if total_count % processed_Xth_frame == 0:
            shown_count += 1

            # If the image is of a different size, resize it
            if frame.shape != (frameWidth, frameHeight):
                frame = cv2.resize(
                    frame,
                    (frameWidth, frameHeight),
                    interpolation=cv2.INTER_AREA,
                )

            # flip the image left-right
            frame = cv2.flip(frame, 1)
            frame_save = frame

            # Convert to greyscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply the face detector
            rects = detector(gray, 0)

            if len(rects) > 0:
                rect = rects[0]
                rect_saved = rect
            elif ("rect" not in locals()) and ("rect" not in globals()):
                continue
            else:
                rect = rect_saved

            # Apply the facial landmarks detector
            facial_landmarks = predictor(gray, rect)
            facial_landmarks = face_utils.shape_to_np(facial_landmarks)

            # the chin
            facial_landmark9 = (facial_landmarks[8][0], facial_landmarks[8][1])
            # the t-zone
            facial_landmark28 = (
                facial_landmarks[27][0],
                facial_landmarks[27][1],
            )
            # the left edge of face
            facial_landmark1 = (facial_landmarks[0][0], facial_landmarks[0][1])
            # the right edge of face
            facial_landmark17 = (
                facial_landmarks[16][0],
                facial_landmarks[16][1],
            )
            # the left eye
            facial_landmark37 = (
                facial_landmarks[36][0],
                facial_landmarks[36][1],
            )
            facial_landmark38 = (
                facial_landmarks[37][0],
                facial_landmarks[37][1],
            )
            facial_landmark39 = (
                facial_landmarks[38][0],
                facial_landmarks[38][1],
            )
            facial_landmark40 = (
                facial_landmarks[39][0],
                facial_landmarks[39][1],
            )
            facial_landmark41 = (
                facial_landmarks[40][0],
                facial_landmarks[40][1],
            )
            facial_landmark42 = (
                facial_landmarks[41][0],
                facial_landmarks[41][1],
            )
            # the rights eye
            facial_landmark43 = (
                facial_landmarks[42][0],
                facial_landmarks[42][1],
            )
            facial_landmark44 = (
                facial_landmarks[43][0],
                facial_landmarks[43][1],
            )
            facial_landmark45 = (
                facial_landmarks[44][0],
                facial_landmarks[44][1],
            )
            facial_landmark46 = (
                facial_landmarks[45][0],
                facial_landmarks[45][1],
            )
            facial_landmark47 = (
                facial_landmarks[46][0],
                facial_landmarks[46][1],
            )
            facial_landmark48 = (
                facial_landmarks[47][0],
                facial_landmarks[47][1],
            )

            #############################
            # Draw the base traffic light
            #############################
            if small_sized:
                trafficlightWidth = 120
                trafficlightHeight = 100
                traffic_light = (
                    np.ones(
                        (trafficlightHeight, trafficlightWidth), dtype=np.int8
                    )
                    * 100
                )
                traffic_light = np.repeat(
                    traffic_light[:, :, np.newaxis], 3, axis=2
                )
                # for head
                cv2.circle(
                    traffic_light,
                    (
                        int(trafficlightWidth / 2) - 20,
                        int(trafficlightHeight / 2),
                    ),
                    10,
                    (0, 0, 0),
                    -1,
                )  # left
                cv2.circle(
                    traffic_light,
                    (
                        int(trafficlightWidth / 2) + 20,
                        int(trafficlightHeight / 2),
                    ),
                    10,
                    (0, 0, 0),
                    -1,
                )  # right
                cv2.circle(
                    traffic_light,
                    (
                        int(trafficlightWidth / 2),
                        int(trafficlightHeight / 2) - 20,
                    ),
                    10,
                    (0, 0, 0),
                    -1,
                )  # top
                cv2.circle(
                    traffic_light,
                    (int(trafficlightWidth / 2), int(trafficlightHeight / 2)),
                    10,
                    (0, 0, 0),
                    -1,
                )  # middle
                cv2.circle(
                    traffic_light,
                    (
                        int(trafficlightWidth / 2),
                        int(trafficlightHeight / 2) + 20,
                    ),
                    10,
                    (0, 0, 0),
                    -1,
                )  # bottom
                # for shoulder
                cv2.circle(
                    traffic_light,
                    (
                        int(trafficlightWidth / 2) - 20,
                        int(trafficlightHeight / 2) + 40,
                    ),
                    10,
                    (0, 0, 0),
                    -1,
                )  # left
                cv2.circle(
                    traffic_light,
                    (
                        int(trafficlightWidth / 2) + 20,
                        int(trafficlightHeight / 2) + 40,
                    ),
                    10,
                    (0, 0, 0),
                    -1,
                )  # right
                cv2.circle(
                    traffic_light,
                    (
                        int(trafficlightWidth / 2),
                        int(trafficlightHeight / 2) + 40,
                    ),
                    10,
                    (0, 0, 0),
                    -1,
                )  # middle

            # If not small_sized
            # for head
            cv2.circle(
                frame, (int(frameWidth / 2) - 20, 40), 10, (0, 0, 0), -1
            )  # left
            cv2.circle(
                frame, (int(frameWidth / 2) + 20, 40), 10, (0, 0, 0), -1
            )  # right
            cv2.circle(
                frame, (int(frameWidth / 2), 20), 10, (0, 0, 0), -1
            )  # top
            cv2.circle(
                frame, (int(frameWidth / 2), 40), 10, (0, 0, 0), -1
            )  # middle
            cv2.circle(
                frame, (int(frameWidth / 2), 60), 10, (0, 0, 0), -1
            )  # bottom
            # for shoulder
            cv2.circle(
                frame, (int(frameWidth / 2) - 20, 80), 10, (0, 0, 0), -1
            )  # left
            cv2.circle(
                frame, (int(frameWidth / 2) + 20, 80), 10, (0, 0, 0), -1
            )  # right
            cv2.circle(
                frame, (int(frameWidth / 2), 80), 10, (0, 0, 0), -1
            )  # middle

            #########################################################
            # => head TILT LEFT/RIGHT, leaned FORWARD/ BACKWARDS <= #
            #########################################################
            # from the facial landmarks 9 and 28, plot the angle of the face
            # to the verticle
            angle_to_vertical = angle_point9_point_28(
                facial_landmark9, facial_landmark28
            )
            if facial_landmark9[0] < facial_landmark28[0]:
                angle_to_vertical_str = (
                    "Angle to vertical: "
                    + "+"
                    + str(round(angle_to_vertical, 2))
                )
            else:
                angle_to_vertical_str = (
                    "Angle to vertical: "
                    + "-"
                    + str(round(angle_to_vertical, 2))
                )

            if debug:
                cv2.putText(
                    frame,
                    angle_to_vertical_str,
                    (10, frameHeight - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.line(
                    frame,
                    (facial_landmark9),
                    (facial_landmark28),
                    (255, 255, 255),
                    1,
                )
                cv2.line(
                    frame,
                    (facial_landmark9),
                    (facial_landmark9[0], facial_landmark28[1]),
                    (255, 255, 255),
                    1,
                )

            if (
                angle_to_vertical > tilt_leftright_threshold_in_degrees
                and facial_landmarks[8][0] < facial_landmarks[27][0]
            ):
                head_message = "(Head) Too much to the RIGHT."
                head_too_right_count += 1
                tmp_head_too_right_count += 1
                (
                    tmp_head_too_left_count,
                    tmp_head_too_back_count,
                    tmp_head_too_front_count,
                ) = (0, 0, 0)
                cv2.circle(
                    frame, (int(frameWidth / 2) + 20, 40), 10, (0, 0, 255), -1
                )  # right
                if small_sized:
                    cv2.circle(
                        traffic_light,
                        (
                            int(trafficlightWidth / 2) + 20,
                            int(trafficlightHeight / 2),
                        ),
                        10,
                        (0, 0, 255),
                        -1,
                    )  # right
            elif (
                angle_to_vertical > tilt_leftright_threshold_in_degrees
                and facial_landmark9[0] > facial_landmarks[27][0]
            ):
                head_message = "(Head) Too much to the LEFT."
                head_too_left_count += 1
                tmp_head_too_left_count += 1
                (
                    tmp_head_too_right_count,
                    tmp_head_too_back_count,
                    tmp_head_too_front_count,
                ) = (0, 0, 0)
                cv2.circle(
                    frame, (int(frameWidth / 2) - 20, 40), 10, (0, 0, 255), -1
                )  # left
                if small_sized:
                    cv2.circle(
                        traffic_light,
                        (
                            int(trafficlightWidth / 2) - 20,
                            int(trafficlightHeight / 2),
                        ),
                        10,
                        (0, 0, 255),
                        -1,
                    )  # left
            elif (
                facial_landmark28[1]
                - (facial_landmark1[1] + facial_landmark17[1]) / 2
                > tilt_frontback_threshold_in_pixels
            ):
                head_message = "(Head) Too much to the FRONT."
                head_too_front_count += 1
                tmp_head_too_front_count += 1
                (
                    tmp_head_too_left_count,
                    tmp_head_too_right_count,
                    tmp_head_too_back_count,
                ) = (0, 0, 0)
                cv2.circle(
                    frame, (int(frameWidth / 2), 60), 10, (0, 0, 255), -1
                )  # bottom
                if small_sized:
                    cv2.circle(
                        traffic_light,
                        (
                            int(trafficlightWidth / 2),
                            int(trafficlightHeight / 2) + 20,
                        ),
                        10,
                        (0, 0, 255),
                        -1,
                    )  # bottom
            elif (
                facial_landmark1[1] + facial_landmark17[1]
            ) / 2 - facial_landmark28[1] > tilt_frontback_threshold_in_pixels:
                head_message = "(Head) Too much to the BACK."
                head_too_back_count += 1
                tmp_head_too_back_count += 1
                (
                    tmp_head_too_left_count,
                    tmp_head_too_right_count,
                    tmp_head_too_front_count,
                ) = (0, 0, 0)
                cv2.circle(
                    frame, (int(frameWidth / 2), 20), 10, (0, 0, 255), -1
                )  # top
                if small_sized:
                    cv2.circle(
                        traffic_light,
                        (
                            int(trafficlightWidth / 2),
                            int(trafficlightHeight / 2) - 20,
                        ),
                        10,
                        (0, 0, 255),
                        -1,
                    )  # top
            else:
                head_message = "(Head) You are OKAY!"
                head_okay_count += 1
                (
                    tmp_head_too_left_count,
                    tmp_head_too_right_count,
                    tmp_head_too_back_count,
                    tmp_head_too_front_count,
                ) = (0, 0, 0, 0)
                cv2.circle(
                    frame, (int(frameWidth / 2), 40), 10, (0, 255, 0), -1
                )  # middle
                if small_sized:
                    cv2.circle(
                        traffic_light,
                        (
                            int(trafficlightWidth / 2),
                            int(trafficlightHeight / 2),
                        ),
                        10,
                        (0, 255, 0),
                        -1,
                    )  # middle
            cv2.putText(
                frame,
                head_message,
                (10, frameHeight - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            ###################################
            # => EYES NOT LOOKING STRAIGHT <= #
            ###################################
            (
                left_eye_centre_block,
                left_eye_cm_iris,
            ) = extract_iris_cm_and_centre_of_eye(
                facial_landmark37,
                facial_landmark38,
                facial_landmark39,
                facial_landmark40,
                facial_landmark41,
                facial_landmark42,
                frameHeight,
                frameWidth,
                gray,
            )
            left_eye_x_diff = (
                left_eye_cm_iris[0] - left_eye_centre_block[0]
            )  # If positive, means its right of centre
            left_eye_y_diff = (
                left_eye_cm_iris[1] - left_eye_centre_block[1]
            )  # If positive, means its below of centre

            (
                right_eye_centre_block,
                right_eye_cm_iris,
            ) = extract_iris_cm_and_centre_of_eye(
                facial_landmark43,
                facial_landmark44,
                facial_landmark45,
                facial_landmark46,
                facial_landmark47,
                facial_landmark48,
                frameHeight,
                frameWidth,
                gray,
            )
            right_eye_x_diff = (
                right_eye_cm_iris[0] - right_eye_centre_block[0]
            )  # If positive, means its right of centre
            right_eye_y_diff = (
                right_eye_cm_iris[1] - right_eye_centre_block[1]
            )  # If positive, means its below of centre

            if debug:
                left_iris_str = "LEFT: %s %s" % (
                    left_eye_centre_block,
                    left_eye_cm_iris,
                )
                right_iris_str = "RIGHT: %s %s" % (
                    right_eye_centre_block,
                    right_eye_cm_iris,
                )
                cv2.putText(
                    frame,
                    left_iris_str,
                    (10, frameHeight - 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    right_iris_str,
                    (10, frameHeight - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            if (
                abs(left_eye_x_diff) > eye_threshold_in_pixels
                or abs(left_eye_y_diff) > eye_threshold_in_pixels
                or abs(right_eye_x_diff) > eye_threshold_in_pixels
                or abs(right_eye_y_diff) > eye_threshold_in_pixels
            ):
                if verbose:
                    frame = cv2.line(
                        frame,
                        left_eye_cm_iris,
                        (
                            int(left_eye_cm_iris[0] + left_eye_x_diff * 1.25),
                            int(left_eye_cm_iris[1] + left_eye_y_diff * 1.25),
                        ),
                        (255, 255, 255),
                        1,
                    )
                    frame = cv2.line(
                        frame,
                        right_eye_cm_iris,
                        (
                            int(
                                right_eye_cm_iris[0] + right_eye_x_diff * 1.25
                            ),
                            int(
                                right_eye_cm_iris[1] + right_eye_y_diff * 1.25
                            ),
                        ),
                        (255, 255, 255),
                        1,
                    )
                eyes_not_front_count += 1
                cv2.circle(
                    frame, (int(frameWidth / 2), 40), 8, (0, 0, 255), -1
                )  # middle
                if small_sized:
                    cv2.circle(
                        traffic_light,
                        (
                            int(trafficlightWidth / 2),
                            int(trafficlightHeight / 2),
                        ),
                        8,
                        (0, 0, 255),
                        -1,
                    )  # middle
            else:
                eyes_okay_count += 1
                cv2.circle(
                    frame, (int(frameWidth / 2), 40), 8, (0, 255, 0), -1
                )  # middle
                if small_sized:
                    cv2.circle(
                        traffic_light,
                        (
                            int(trafficlightWidth / 2),
                            int(trafficlightHeight / 2),
                        ),
                        8,
                        (0, 255, 0),
                        -1,
                    )  # middle

            #################################
            # => SHOULDERS NOT  STRAIGHT <= #
            #################################
            # Apply shoulder detector
            iptH = 80  # How much to resize when creating the blob image
            iptW = int((iptH / frameHeight) * frameWidth)
            blob = cv2.dnn.blobFromImage(
                image=frame,
                scalefactor=1.0 / 255,
                size=(iptW, iptH),
                mean=(0, 0, 0),
                swapRB=False,
                crop=False,
            )
            net.setInput(blob)
            output = net.forward()

            # get point masks
            onePrMap = output[0, 2, :, :]
            onePrMap = cv2.resize(onePrMap, (frameWidth, frameHeight))
            (pts, mask) = searchPts(onePrMap)

            # get points groups
            (ptGrp, ptList) = getAllPoints(
                cfMaps=output, imgWidth=frameWidth, imgHeight=frameHeight
            )

            if len(ptGrp[2]) > 0 and len(ptGrp[5]) > 0:
                # the left shoulder
                shoulder_landmark2 = (ptGrp[2][0][0], ptGrp[2][0][1])
                # the right shoulder
                shoulder_landmark5 = (ptGrp[5][0][0], ptGrp[5][0][1])
                if (
                    shoulder_landmark2[1] - shoulder_landmark5[1]
                    > tilt_shoulder_threshold_in_pixels
                ):
                    shoulder_message = "(Shoulder) Too slanted to the RIGHT."
                    shoulder_slanted_right_count += 1
                    shoulder_slanted_left_count += 1
                    shoulder_slanted_right_count = 0
                    cv2.circle(
                        frame,
                        (int(frameWidth / 2) + 20, 80),
                        10,
                        (0, 0, 255),
                        -1,
                    )  # right
                    if small_sized:
                        cv2.circle(
                            traffic_light,
                            (
                                int(trafficlightWidth / 2 + 20),
                                int(trafficlightHeight / 2 + 40),
                            ),
                            10,
                            (0, 0, 255),
                            -1,
                        )  # middle
                elif (
                    shoulder_landmark5[1] - shoulder_landmark2[1]
                    > tilt_shoulder_threshold_in_pixels
                ):
                    shoulder_message = "(Shoulder) Too slanted to the LEFT."
                    shoulder_slanted_left_count += 1
                    shoulder_slanted_right_count += 1
                    shoulder_slanted_left_count = 0
                    cv2.circle(
                        frame,
                        (int(frameWidth / 2) - 20, 80),
                        10,
                        (0, 0, 255),
                        -1,
                    )  # left
                    if small_sized:
                        cv2.circle(
                            traffic_light,
                            (
                                int(trafficlightWidth / 2 - 20),
                                int(trafficlightHeight / 2 + 40),
                            ),
                            10,
                            (0, 0, 255),
                            -1,
                        )  # middle
                else:
                    shoulder_message = "(Shoulder) You are OKAY!"
                    shoulder_okay_count += 1
                    (
                        shoulder_slanted_left_count,
                        shoulder_slanted_right_count,
                    ) = (
                        0,
                        0,
                    )
                    cv2.circle(
                        frame, (int(frameWidth / 2), 80), 10, (0, 255, 0), -1
                    )  # middle
                    if small_sized:
                        cv2.circle(
                            traffic_light,
                            (
                                int(trafficlightWidth / 2),
                                int(trafficlightHeight / 2 + 40),
                            ),
                            10,
                            (0, 255, 0),
                            -1,
                        )  # middle
                cv2.putText(
                    frame,
                    shoulder_message,
                    (10, frameHeight - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            ########################################
            # => Plot facial/shoulder landmarks <= #
            ########################################
            if verbose:
                # Plot the facial landmarks
                for i in [
                    0,
                    1,
                    2,
                    14,
                    15,
                    16,
                    8,
                    27,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                ]:
                    (x, y) = facial_landmarks[i]
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1, cv2.LINE_AA)
                # Plot the shoulder landmarks
                for i in [1, 2, 5]:
                    # for j in range(len(ptGrp[i])):
                    if len(ptGrp[i]) > 0:
                        cv2.circle(
                            frame,
                            ptGrp[i][0][0:2],
                            3,
                            (255, 0, 0),
                            -1,
                            cv2.LINE_AA,
                        )

            ########################
            # => Plot timestamp <= #
            ########################
            # Get the current timestamp
            t = time.time()
            t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
            # Plot the current timestamp
            cv2.putText(
                frame,
                t_str,
                (10, frameHeight - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # For sounding of the alarm based on time
            tmp_too_count = sum(
                [
                    tmp_head_too_left_count,
                    tmp_head_too_right_count,
                    tmp_head_too_back_count,
                    tmp_head_too_front_count,
                    shoulder_slanted_right_count,
                    shoulder_slanted_left_count,
                ]
            )
            tmp_duration = t - t_start
            if (
                tmp_too_count / shown_count * tmp_duration
                > wrong_posture_duration_in_seconds
            ):
                playsound.playsound(soundclip_filepath)

            # For sounding of the alarm based on duration
            tmp_duration = t - t_start
            if tmp_duration > seated_duration_in_seconds:
                cv2.putText(
                    frame,
                    "Please stand up and move!",
                    (int(frameWidth / 2) - 140, int(frameHeight / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    1,
                )

            # Put the options on the image
            cv2.putText(
                frame,
                "Options: V, Q; S, E; ESC",
                (frameWidth - 200, frameHeight - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Show the image
            if small_sized is False:
                cv2.imshow("Output", frame)
            elif small_sized is True:
                cv2.imshow("Output", traffic_light)

        total_count += 1
        # turn off if total_count once reaches 10m (if idle?)
        if total_count == 10000000:
            break

        c = cv2.waitKey(1)
        if c == 27:  # ESC key
            break
        elif c == 118:  # V key
            verbose = True
        elif c == 113:  # Q key
            verbose = False
        elif c == 61:  # + key
            debug = True
            verbose = True
        elif c == 45:  # - key
            debug = False
            verbose = False
        elif c == 115:  # S key
            small_sized = True
        elif c == 101:  # E key
            small_sized = False

    cap.release()
    cv2.destroyAllWindows()

    # Visualize the last frame taken with some output metrics
    t_end = time.time()
    duration = t_end - t_start
    time_spent_msg = "Total time shown: %1.2fs" % (duration)
    eyes_okay_time_msg = "(Eyes) Okay: %1.2fs" % (
        eyes_okay_count / shown_count * duration
    )
    head_too_time_msg1 = "(Head) Too (front,back): %1.2fs, %1.2fs" % (
        head_too_front_count / shown_count * duration,
        head_too_back_count / shown_count * duration,
    )
    head_too_time_msg2 = "(Head) Too (left,right): %1.2fs, %1.2fs" % (
        head_too_left_count / shown_count * duration,
        head_too_right_count / shown_count * duration,
    )
    head_okay_time_msg = "(Head) Okay: %1.2fs" % (
        head_okay_count / shown_count * duration
    )
    shoulder_too_time_msg = (
        "(Shoulder) Slanted (left,right): %1.2fs, %1.2fs"
        % (
            shoulder_slanted_left_count / shown_count * duration,
            shoulder_slanted_right_count / shown_count * duration,
        )
    )
    shoulder_okay_time_msg = "(Shoulder) Okay: %1.2fs" % (
        shoulder_okay_count / shown_count * duration
    )

    cv2.putText(
        frame_save,
        time_spent_msg,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame_save,
        eyes_okay_time_msg,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame_save,
        head_too_time_msg1,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame_save,
        head_too_time_msg2,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame_save,
        head_okay_time_msg,
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame_save,
        shoulder_too_time_msg,
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame_save,
        shoulder_okay_time_msg,
        (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    overall_okay = (
        (eyes_okay_count / shown_count)
        * (head_okay_count / shown_count)
        * (shoulder_okay_count / shown_count)
    )
    if overall_okay > overall_okay_threshold:
        colour = (0, 255, 0)
        overall_okay_time_msg = "Overall: %1.2fs/%1.2fs correct, GOOD!" % (
            overall_okay * duration,
            duration,
        )
    else:
        colour = (0, 0, 255)
        overall_okay_time_msg = "Overall: %1.2fs/%1.2fs only, IMPROVE!" % (
            overall_okay * duration,
            duration,
        )
    cv2.putText(
        frame_save,
        overall_okay_time_msg,
        (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colour,
        1,
    )

    cv2.imshow("Input", frame_save)
    cv2.waitKey(30000)  # Hold the final frame for 30s

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
