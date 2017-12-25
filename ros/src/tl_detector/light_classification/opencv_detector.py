# Author: Jacques Roth (https://github.com/JacquesRoth/CarND-Capstone)

import cv2
import numpy as np
import math

from styx_msgs.msg import TrafficLight

global best_light_x, best_light_y, delta_light
best_light_x = 100000.0
best_light_y = 100000.0
delta_light = 100000.0
# Integrators
global rint, yint, gint, uint
rint = 0
yint = 0
gint = 0
uint = 5
colortxt = ['RED', 'YELLOW', 'GREEN', 'BROKEN', 'UNKNOWN']
global frameno
frameno = 0
#
# Carla Camera Parameters, calculated by parameter optimization program
#
CameraHeight = 0.5  # Camera height on car
CameraCos = 0.9959  # Uptilt angle cos and sin
CameraSin = -0.09
CameraPan = 0.06
CameraF = 1049.0  # Camera focal length & pixel adjustments
PanError = 0.16
PanPixels = int(CameraF * PanError)

global rexpected
rexpected = 7


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def render_region_of_interest(img, vertices):
    for i in range(0, 4):
        x1, y1 = vertices[i]
        next_vertices = i + 1 if i < 3 else 0
        x2, y2 = vertices[next_vertices]
        cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)


def render_recognized_circles(image, circles, border_color=(0, 0, 0), center_color=(0, 0, 255)):
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(image, (i[0], i[1]), i[2], border_color, 2)
        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, center_color, 3)


def recognize_light(image, lower_range, upper_range):
    ranged_image = cv2.inRange(image, lower_range, upper_range)
    return ranged_image


def get_hough_circles(weighted_image, hsv_image, isCarla):
    global rexpected
    bestfraction = 0.0
    posColor = TrafficLight.UNKNOWN
    if isCarla:
        blur_img = cv2.GaussianBlur(weighted_image, (15, 15), 0)
        blur_img = get_canny_edge(blur_img)
        minR = int(rexpected * 0.75 - 0.5)
        maxR = int(rexpected * 1.25 + 0.5)
        if minR < 6:  minR = 6
        if maxR > 25: maxR = 25
        circles = cv2.HoughCircles(blur_img, cv2.HOUGH_GRADIENT, 0.5, 41, param1=30, param2=7, \
                                   minRadius=minR, maxRadius=maxR)
    else:
        canny_img = get_canny_edge(weighted_image)
        return (cv2.HoughCircles(canny_img, cv2.HOUGH_GRADIENT, 3, 20, param1=20, param2=35, \
                                 minRadius=5, maxRadius=25), 0.0, TrafficLight.UNKNOWN)
    #
    # For Carla examine the filled in area around the suspected traffic lights and find the most
    # filled.
    #
    if circles == None:
        return (None, bestfraction, posColor)

    bestcircle = None
    frameH, frameW = blur_img.shape
    #
    # Look at circles returned and pick out the best filled in
    # circle.  Throw away circles with very poor filling.
    #
    i = 0
    for circle in circles[0, :]:
        x = int(circle[0] + 0.5)
        y = int(circle[1] + 0.5)
        r = int(circle[2] + 0.5)
        if (y < r) or y > (frameH - 2 * r):  # 6*r):
            i += 1
            continue
        if (x < r) or x > (frameW - r):
            i += 1
            continue
        square_img = weighted_image[y - r:y + r, x - r:x + r]
        count = np.count_nonzero(square_img)
        fraction = float(count) / (4.0 * r * r)
        if (fraction > 0.2) and (count > 10) and (fraction > bestfraction):
            bestcircle = circle
            bestfraction = fraction
        i += 1

    if bestcircle == None:
        return (None, bestfraction, posColor)
    #
    # Form a rectangular image around the suspected traffic light and look for 3 lights total
    # If we do not find at least 2, consider it not a traffic light.
    # The rectangular image is color ranged to emphasize the yellowish color of the traffic
    # light box.
    #
    x = bestcircle[0]
    y = bestcircle[1]
    r = bestcircle[2]
    top = int(y - 9.0 * r)
    bot = int(y + 9.0 * r)
    left = int(x - 2.0 * r)
    right = int(x + 2.0 * r)
    deltaH = float(bot - top)
    toporig = top
    if top < 0: top = 0
    if left < 0: left = 0
    if bot >= frameH:
        bot = frameH - 1
    if right >= frameW: right = frameW - 1
    if (top < bot) and (left < right):
        tl_image = hsv_image[top:bot, left:right]
        lower_frame = np.array([10, 7, 30])
        upper_frame = np.array([24, 200, 240])
        maskf = cv2.inRange(tl_image, lower_frame,
                            upper_frame)
        minr = int(rexpected * 0.3)  # Set a minimum and maximum radius for the circle search
        maxr = int(rexpected * 2.5)
        if minr < 6:  minr = 6
        if maxr > 26: maxr = 26
        mindist = int(rexpected * 6.0)
        circlesf = cv2.HoughCircles(maskf, cv2.HOUGH_GRADIENT, \
                                    0.5, 12, param1=20, param2=15, minRadius=minr,
                                    maxRadius=maxr)
        #
        # Look at the circles found, ideally 3 but often many more.  Find the average y parameter of
        # all the circles relative to the top of the box.
        # If the average is near the top of the box, assume a green light (green is the bottom light).
        # If near the bottom of the box, assume a red light (red is the bottom light).
        # Otherwise, assume a yellow (center) light.
        #
        ncirclesf = 0
        if circlesf != None:
            sumy = 0.0
            for circlef in circlesf[0, :]:
                yf = circlef[1]
                yfadj = yf - toporig + top
                sumy += yfadj
                ncirclesf += 1
                if ncirclesf >= 40: break
            if ncirclesf != 0:
                avgy = sumy / float(ncirclesf)
                avgy /= deltaH
                if avgy >= 0.6:
                    posColor = TrafficLight.RED
                elif avgy <= 0.4:
                    posColor = TrafficLight.GREEN
                else:
                    posColor = TrafficLight.YELLOW
            else:
                posColor = TrafficLight.UNKNOWN
    #
    # If 2 or more lights detected, return the light most believed to be lit,
    # the fill factor, and the positional Color
    #
    if ncirclesf >= 2:
        if ncirclesf <= 2: posColor = TrafficLight.UNKNOWN
        global best_light_x, best_light_y, delta_light
        delta_light = abs(best_light_x - x) + \
                      abs(best_light_y - y)
        best_light_x = x
        best_light_y = y
        result = [[[bestcircle[0], bestcircle[1],
                    bestcircle[2]]]]
    else:
        return (None, bestfraction, posColor)
    result = np.array(result)
    return (result, bestfraction, posColor)


def recognize_red_light(hsv_image, isCarla):
    if not isCarla:
        lower_red = np.array([0, 60, 130])
        upper_red = np.array([10, 255, 255])
        red1 = recognize_light(hsv_image, lower_red, upper_red)

        lower_red = np.array([160, 70, 100])
        upper_red = np.array([180, 255, 255])
        red2 = recognize_light(hsv_image, lower_red, upper_red)

        weighted_img = cv2.addWeighted(red1, 1.0, red2, 1.0, 0.0)
        return get_hough_circles(weighted_img, None, False)


#
# Carla does not use a red light color recognition routine as the red and yellow
# lights both appear yellow.
#

#
# It is not really necessary to recognize green lights, except for display
# purposes.
#
def recognize_green_light(hsv_image, isCarla):
    if not isCarla:
        lower_green = np.array([50, 100, 120])
        upper_green = np.array([80, 255, 255])
        green1 = recognize_light(hsv_image, lower_green, upper_green)

        lower_green = np.array([60, 150, 190])
        upper_green = np.array([85, 255, 255])
        green2 = recognize_light(hsv_image, lower_green, upper_green)

        weighted_img = cv2.addWeighted(green1, 1.0, green2, 1.0, 0.0)
        return get_hough_circles(weighted_img, None, False)
    #
    # Carla has a green light that appears almost white in the center
    # with a green ring around it.
    #
    debug = True
    lower_green = np.array([40, 80, 90])
    upper_green = np.array([140, 255, 255])
    green1 = recognize_light(hsv_image, lower_green, upper_green)
    lower_green = np.array([0, 0, 240])
    upper_green = np.array([5, 5, 255])
    green2 = recognize_light(hsv_image, lower_green, upper_green)
    green1 = cv2.addWeighted(green1, 1.0, green2, 1.0, 0.0)
    result, bestfraction, posColor = get_hough_circles(green1, hsv_image, True)
    return (result, bestfraction, posColor)


def recognize_yellow_light(hsv_image, isCarla):
    if not isCarla:
        lower_yellow = np.array([30, 70, 150])
        upper_yellow = np.array([45, 170, 180])
        yellow1 = recognize_light(hsv_image, lower_yellow, upper_yellow)

        lower_yellow = np.array([30, 70, 170])
        upper_yellow = np.array([45, 255, 255])
        yellow2 = recognize_light(hsv_image, lower_yellow, upper_yellow)

        weighted_img = cv2.addWeighted(yellow1, 1.0, yellow2, 1.0, 0.0)
        return (get_hough_circles(weighted_img, None, False), 0.0, None)
    #
    # For Carla, the red and yellow lights are approximately the same color
    #
    lower_yellow = np.array([15, 90, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow1 = recognize_light(hsv_image, lower_yellow, upper_yellow)
    result, bestfraction, posColor = get_hough_circles(yellow1, hsv_image, True)
    return (result, bestfraction, posColor)


def get_canny_edge(weighted_img):
    kernel = np.ones((5, 5), np.uint8)
    kernel_erode = np.ones((3, 3), np.uint8)

    erode = cv2.erode(weighted_img, kernel_erode, iterations=1)
    dilated = cv2.dilate(erode, kernel)
    dilated = cv2.dilate(dilated, kernel)
    dilated = cv2.dilate(dilated, kernel)

    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return cv2.Canny(closed, 50, 70)


def recognize_traffic_lights(image,
                             isCarla,
                             CarX,
                             CarY,
                             CarZ,
                             Oz,
                             Ow,
                             Lx,
                             Ly,
                             Lz):
    global rint, yint, gint, uint
    global frameno
    global rexpected

    if not isCarla:
        # Convert to HSV color space to recognize the traffic light
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_circles = recognize_red_light(hsv_image, isCarla)[0]
        if red_circles is not None:
            return TrafficLight.RED

        green_circles = recognize_green_light(hsv_image, isCarla)[0]
        if green_circles is not None:
            return TrafficLight.GREEN

        yellow_circles = recognize_yellow_light(hsv_image, isCarla)[0]
        if yellow_circles is not None:
            return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
    #
    # For Carla, calculate a rectangular ROI and apply it to the image
    #
    # Carla traffic lights are different in color from the simulator and
    # ROI is dynamically calculated
    #
    FrameH, FrameW, _ = image.shape
    FrameHC = FrameH / 2
    FrameWC = FrameW / 2
    debug = True
    cosO = 2.0 * Oz * Ow
    sinO = 2.0 * Ow * Ow - 1.0
    CarO = math.atan2(cosO, sinO)

    # Translate traffic light position so camera is at (0, 0, 0)
    Lx -= CarX
    Ly -= CarY
    Lz -= CarZ + CameraHeight

    # Rotate traffic light position by orientation so camera
    # faces along the x axis
    # Lz needs no changes
    #
    cosO = math.cos(CarO + CameraPan)
    sinO = math.sin(CarO + CameraPan)
    Lxt = Lx
    Lx = Lx * cosO + Ly * sinO
    Ly = Ly * cosO - Lxt * sinO
    #
    # Rotate traffic light position by camera tilt so camera faces
    # along the x axis
    # Ly needs no changes
    #
    Lxt = Lx
    Lx = Lx * CameraCos + Lz * CameraSin
    Lz = -Lxt * CameraSin + Lz * CameraCos
    #
    # Calculate image position
    #
    Ix = int(FrameWC - CameraF * Ly / Lx)
    Iy = int(FrameHC - CameraF * Lz / Lx)
    #
    # Calculate expected radius of traffic light
    #
    rexpected = CameraF * 0.2 / Lx
    if Ix < 0:
        Ix = 0
    elif Ix >= FrameW:
        Ix = FrameW - 1
    if Iy < 0:
        Iy = 0
    elif Iy >= FrameH:
        Iy = FrameH - 1

    RecWC = int(CameraF * 1.5 / Lx)
    RecHC = int(CameraF * 1.5 / Lx)

    if RecWC < 10: RecWC = 10
    if RecHC < 10: RecHC = 10

    RecTop = Iy - RecHC
    RecBot = Iy + RecHC
    RecLeft = Ix - RecWC - PanPixels
    RecRight = Ix + RecWC + PanPixels

    if RecTop < 0: RecTop = 0
    if RecBot > FrameH - 1:
        RecBot = FrameH - 1
    if RecLeft < 0: RecLeft = 0
    if RecRight > FrameW - 1: RecRight = FrameW - 1
    #
    # Form cropped image, just a rectangle
    #
    cimg = image[RecTop:RecBot, RecLeft:RecRight]

    # Convert to HSV color space to recognize the traffic light
    hsv_image = cv2.cvtColor(cimg, cv2.COLOR_BGR2HSV)
    # Match yellow/red lights using the same color filter
    yellow_circles, fractiony, posYellow = recognize_yellow_light(hsv_image, isCarla)
    if yellow_circles is not None:
        colorcode = posYellow
    else:
        #
        # It is not really necessary to check for green lights.
        #
        green_circles, fractiong, posGreen = recognize_green_light(hsv_image, isCarla)
        if green_circles is not None:
            colorcode = TrafficLight.GREEN
        else:
            colorcode = TrafficLight.UNKNOWN

    if colorcode == TrafficLight.RED:
        if rint != 5: rint += 1
        if uint != 0: uint -= 1
        if gint != 0: gint -= 1
        if yint != 0: yint -= 1
    elif colorcode == TrafficLight.YELLOW:
        if yint != 5: yint += 1
        if uint != 0: uint -= 1
        if rint != 0: rint -= 1
        if gint != 0: gint -= 1
    elif colorcode == TrafficLight.GREEN:
        if gint != 5: gint += 1
        if uint != 0: uint -= 1
        if rint != 0: rint -= 1
        if yint != 0: yint -= 1
    else:
        if uint != 5: uint += 1
        if rint != 0: rint -= 1
        if gint != 0: gint -= 1
        if yint != 0: yint -= 1
    if uint > max(rint, gint, yint):
        colorcode = TrafficLight.UNKNOWN
    elif rint > max(uint, gint, yint):
        colorcode = TrafficLight.RED
    elif yint > max(uint, gint, rint):
        colorcode = TrafficLight.YELLOW
    else:
        colorcode = TrafficLight.GREEN

    frameno += 1
    return colorcode