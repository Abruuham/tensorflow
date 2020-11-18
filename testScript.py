import cv2
import sys
import os

''' 
    Helper utils for video files.

    Example usage:
        - Count the number of frames in video 1. 
            `python3 testScript.py count 1.mp4`

        - Rip all the frames in video 1.mp4 to a specified directory
            `python3 testScript.py rip 1.mp4 ~/out` 
'''


def count_frames(vid_path):
    '''
    Brief:
        Counts and prints the number of frames in a video file.

    Parameters:
        vid_path (string) Path to video file.

    Returns:
        None
    '''

    cap = cv2.VideoCapture(vid_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("{} contains {} frames.".format(vid_path, length))


def rip_frames(vid_path, out_path=None):
    '''
    Brief:
        Parses a video file and saves each frame as a png.

    Parameters:
        vid_path (string) Path to video file.
        out_path (string) Path where frame png file will be saved.

    Returns:
        None
    '''

    if out_path == None:
        print("Please specify a directory to save frames")
    else:
        if os.path.exists(out_path):
            vidcap = cv2.VideoCapture(vid_path)
            success, image = vidcap.read()
            count = 0
            print("Ripping frames...")
            while success:
                path = os.path.join(out_path, "frame%d.png" % count)
                cv2.imwrite(path, image)
                success, image = vidcap.read()
                count += 1
            print("Saved {} frames to {}.".format(count, out_path))
        else:
            print("Specified directory does not exist")


if __name__ == '__main__':

    if sys.argv[1] == 'count':
        count_frames(sys.argv[2])
    if sys.argv[1] == 'rip':
        if len(sys.argv) == 4:
            rip_frames(sys.argv[2], sys.argv[3])
        else:
            print("Wrong number of argumemts.`python3 AITAS_utils.py rip <path_to_file> <output_dir>")
