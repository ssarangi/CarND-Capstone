#!/usr/bin/env python
import subprocess as sp
import argparse
import os, fnmatch
import cv2
import rospy
import logging

logger = logging.getLogger(__name__)

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def find_files(directory, pattern):
    print(directory)
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input dir", type=str)
    parser.add_argument("--output", help="Output dir", type=str)

    args = parser.parse_args()
    return args


image_idx = 0
output_dir = None
bridge = CvBridge()


def image_cb(msg):
    logger.info('In Image Color')
    global image_idx, output_dir, bridge
    # fixing convoluted camera encoding...
    if hasattr(msg, 'encoding'):
        if msg.encoding == '8UC3':
            msg.encoding = "rgb8"
    else:
        msg.encoding = 'rgb8'

    camera_image = bridge.imgmsg_to_cv2(msg, "rgb8")
    camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_dir + '/%s.png' % image_idx, camera_image)
    image_idx += 1


def main():
    global output_dir, counter, allpoints

    rospy.init_node("rosbag2images")
    rospy.Subscriber('/image_raw', Image, image_cb)

    args = argument_parser()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for filename in find_files(args.input, '*.bag'):
        print('Playing Bag file: ', filename)
        output_dir = args.output

        # Now play the rosbag file.
        p = sp.Popen(("rosbag play " + filename).split(), stdin=sp.PIPE)
        p.wait()

if __name__ == "__main__":
    main()
