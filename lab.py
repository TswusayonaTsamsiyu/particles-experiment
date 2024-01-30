import cv2 as cv

import image as img
import display as disp
from video import Video, Frame
from utils import Image, Position, exit_for
from fs import get_bg_videos, get_rod_videos
from contours import find_contours, draw_contours

BG_FRAME = 3600
JUMP_FRAMES = 20

BLUR_SIZE = 15
KSIZE = (BLUR_SIZE, BLUR_SIZE)

ESC = 27
CLOSE_BTN = -1
EXIT_CODES = {ESC, CLOSE_BTN}


def prepare(frame: Image) -> Image:
    return img.blur(img.monochrome(frame), KSIZE)


# def make_binary(frame: Image) -> Image:
# return img.adaptive_threshold(frame, adaptive_method=cv.ADAPTIVE_THRESH_GAUSSIAN_C ,block_size=71 ,cut=1)
# return img.threshold(frame, np.std(frame) * 5)


# def has_tracks(frame: Image) -> bool:
#     mean, std = cv.meanStdDev(frame)
#     min_, max_ = cv.minMaxLoc(frame)[:2]
#     print(f"Mean: {round(float(mean[0][0]), 3)}, STD: {round(float(std[0][0]), 3)}, Max: {max_}")
#     return mean > 0.6 and std > 1 and max_ > 25


def has_tracks(threshold: float) -> bool:
    return threshold >= 1


def analyze_frame(frame: Frame, bg: Image) -> None:
    print(f"Processing frame {frame.index}")
    prepared = prepare(frame.pixels)
    subtracted = cv.subtract(prepared, bg)
    thresh, binary = img.threshold_otsu(subtracted)
    if has_tracks(thresh):
        contours = tuple(contour.convex_hull()
                         for contour in find_contours(binary)
                         if contour.area() > 600)
        with disp.window_control(exit_for(EXIT_CODES)):
            shown = disp.fit_to_screen(prepared)
            prepwin = disp.show_window(shown,
                                       title=f"Prepared frame {frame.index}",
                                       position=Position(disp.screen_center().x - shown.shape[1], 0))
            disp.show_window(disp.fit_to_screen(draw_contours(binary, contours)),
                             title=f"Binary frame {frame.index} with contours",
                             position=disp.right_of(prepwin))


def analyze_video(video: Video) -> None:
    frames = video.iter_frames(start=BG_FRAME, jump=JUMP_FRAMES)
    bg = prepare(next(frames).pixels)
    disp.show_single_window(disp.fit_to_screen(bg),
                            title=f"Background frame {BG_FRAME}",
                            position=Position(0, 0))
    for frame in frames:
        analyze_frame(frame, bg)


def main() -> None:
    example_path = list(get_bg_videos())[1]
    print(f"Parsing {example_path.name}...")
    with Video(example_path) as video:
        print(f"Video has {video.frame_num} frames.")
        analyze_video(video)


if __name__ == '__main__':
    main()
