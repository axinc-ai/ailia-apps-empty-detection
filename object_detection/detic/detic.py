import sys
import time
import datetime
import platform

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from dataset_utils import get_lvis_meta_v1, get_in21k_meta_v1
from color_utils import random_color, color_brightness

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_SWINB_LVIS_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis.onnx'
MODEL_SWINB_LVIS_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis.onnx.prototxt'
WEIGHT_SWINB_IN21K_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_in21k.onnx'
MODEL_SWINB_IN21K_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_in21k.onnx.prototxt'
WEIGHT_R50_LVIS_PATH = 'Detic_C2_R50_640_4x_lvis.onnx'
MODEL_R50_LVIS_PATH = 'Detic_C2_R50_640_4x_lvis.onnx.prototxt'
WEIGHT_R50_IN21K_PATH = 'Detic_C2_R50_640_4x_in21k.onnx'
MODEL_R50_IN21K_PATH = 'Detic_C2_R50_640_4x_in21k.onnx.prototxt'
WEIGHT_SWINB_LVIS_OP16_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis_op16.onnx'
MODEL_SWINB_LVIS_OP16_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis_op16.onnx.prototxt'
WEIGHT_SWINB_IN21K_OP16_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_in21k_op16.onnx'
MODEL_SWINB_IN21K_OP16_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_in21k_op16.onnx.prototxt'
WEIGHT_R50_LVIS_OP16_PATH = 'Detic_C2_R50_640_4x_lvis_op16.onnx'
MODEL_R50_LVIS_OP16_PATH = 'Detic_C2_R50_640_4x_lvis_op16.onnx.prototxt'
WEIGHT_R50_IN21K_OP16_PATH = 'Detic_C2_R50_640_4x_in21k_op16.onnx'
MODEL_R50_IN21K_OP16_PATH = 'Detic_C2_R50_640_4x_in21k_op16.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/detic/'

IMAGE_PATH = 'desk.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Detic', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--seed', type=int, default=int(datetime.datetime.now().strftime('%Y%m%d')),
    help='random seed for selection the color of the box'
)
parser.add_argument(
    '-m', '--model_type', default='SwinB_896_4x', choices=('SwinB_896_4x', 'R50_640_4x'),
    help='model type'
)
parser.add_argument(
    '-vc', '--vocabulary', default='lvis', choices=('lvis', 'in21k'),
    help='vocabulary'
)
parser.add_argument(
    '--opset16',
    action='store_true',
    help='Use the opset16 model. In that case, grid_sampler runs inside the model.'
)
parser.add_argument(
    '--area', type=str, default="",
    help='Set area list of (id x1 y1 x2 y2 x3 y3 x4 y4).'
)
parser.add_argument(
    '--csvpath', type=str, default=None,
    help='Set output csv.'
)
parser.add_argument(
    '--imgpath', type=str, default=None,
    help='Set output image.'
)
parser.add_argument(
    '-dw', '--detection_width',
    default=800, type=int,   # tempolary limit to 800px (original : 1333)
    help='The detection width for detic. (default: 800)'
)
parser.add_argument(
    '-at', '--area_threshold',
    default=0.125, type=float,
    help='The area threshold. (default: 800)'
)
parser.add_argument(
    '--accept_label', dest='accept_text_inputs', type=str,
    action='append',
    help='Accept label text. (can be specified multiple times)'
)
parser.add_argument(
    '--deny_label', dest='deny_text_inputs', type=str,
    action='append',
    help='Deny label text. (can be specified multiple times)'
)
parser.add_argument(
    '--multiple_assign',
    action='store_true',
    help='Assign one object to multiple area.'
)
args = update_parser(parser)

if not args.opset16:
    from functional import grid_sample  # noqa

area_list = args.area.split(" ")

# ======================
# Terminate
# ======================

from signal import SIGINT
import signal

terminate_signal = False

def _signal_handler(signal, handler):
    global terminate_signal
    terminate_signal = True

def set_signal_handler():
    signal.signal(signal.SIGINT,  _signal_handler)

# ======================
# Accept and Deny labels
# ======================

if args.accept_text_inputs:
    accept_label = args.accept_text_inputs
else:
    accept_label = ["all"]

if args.deny_text_inputs:
    deny_label = args.deny_text_inputs
else:
    deny_label = ["none"]

def is_accept_label(label):
    global accept_label, deny_label
    accept = False
    for l in accept_label:
        if l == "all" or (l in label):
            accept = True
    for l in deny_label:
        if l != "none" and (l in label):
            accept = False
    return accept

# ======================
# Area detection
# ======================

def prepare_area_mask(frame):
    global area_list
    area_mask = []
    a = 0
    while a < len(area_list):
        area_id = area_list[a]
        mask, target_lines = _create_area_mask(frame, a)
        a = a + 9
        area_mask.append({"id":area_id, "mask":mask,"target_lines":target_lines,"ratio":0.0,"label":""})
    return area_mask

def _create_area_mask(frame, a):
    global area_list
    mask = np.zeros(frame.shape)

    target_lines = [
        [int(area_list[a+1]),int(area_list[a+2])],
        [int(area_list[a+3]),int(area_list[a+4])],
        [int(area_list[a+5]),int(area_list[a+6])],
        [int(area_list[a+7]),int(area_list[a+8])]]

    contours = np.array([target_lines[0], target_lines[1], target_lines[2], target_lines[3], target_lines[0]])
    cv2.fillConvexPoly(mask, points =contours, color=(255, 255, 255))

    return mask, target_lines

def display_area(frame, area_mask):
    for a in range(len(area_mask)):
        area_id = area_mask[a]["id"]
        mask = area_mask[a]["mask"]
        target_lines = area_mask[a]["target_lines"]

        color = (0,0,255)

        #if len(target_lines) >= 2:

        for i in range(len(target_lines) - 1):
            cv2.line(frame, (target_lines[i][0], target_lines[i][1]), (target_lines[i+1][0], target_lines[i+1][1]), color, thickness=1)
        if len(target_lines) >= 4:
            cv2.line(frame, (target_lines[3][0], target_lines[3][1]), (target_lines[0][0], target_lines[0][1]), color, thickness=1)

        if area_mask[a]["ratio"] >= args.area_threshold:
            frame[mask>0] = 255

def display_text(frame, area_mask):
    for a in range(len(area_mask)):
        area_id = area_mask[a]["id"]
        target_lines = area_mask[a]["target_lines"]

        color = (0,0,255)

        label = area_mask[a]["label"]

        cv2.putText(frame, area_id, (target_lines[0][0] + 5,target_lines[0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=1)

        #label = label + "(" + str(int(area_mask[a]["ratio"]*100)/100.0) + ")"
        cv2.putText(frame, area_id + " : "+label, (0, a * 40 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), thickness=3)

def check_area_overwrap(pred_masks, classes, area_mask):
    # get detected label name
    vocabulary = args.vocabulary
    class_names = (
        get_lvis_meta_v1() if vocabulary == 'lvis' else get_in21k_meta_v1()
    )["thing_classes"]
    labels = [class_names[int(i)] for i in classes]  # ailia always returns float tensor so need to add cast

    # reset all ratio
    for a in range(len(area_mask)):
        area_mask[a]["new_ratio"] = 0.0
        area_mask[a]["deny_ratio"] = 0.0

    if args.multiple_assign:
        # one object to multiple area
        for a in range(len(area_mask)):
            # check area of overwrap
            area_mask[a]["ratio"] = 0.0
            m = area_mask[a]["mask"][:,:,0]
            mask_area = m[m > 0] # 0-255
            mask_area_average = np.sum(mask_area)

            for i in range(pred_masks.shape[0]):
                # check area of overwrap
                p = pred_masks[i]
                hit_area = p[m > 0] * 255 # 0-1 -> 0-255
                hit_area_average = np.sum(hit_area)
                ratio = hit_area_average / mask_area_average
                if is_accept_label(labels[i]):
                    if ratio > area_mask[a]["new_ratio"]:
                        area_mask[a]["new_ratio"] = ratio
                else:
                    if ratio > area_mask[a]["deny_ratio"]:
                        area_mask[a]["deny_ratio"] = ratio
    else:
        # one object to one area
        # because if car top is in area_a and car bottom is in area_b,
        # we should assign to only one area
        for i in range(pred_masks.shape[0]):
            # calc maximum overwrap area
            max_ratio = 0.0
            max_a = -1
            for a in range(len(area_mask)):
                # check area of overwrap
                m = area_mask[a]["mask"][:,:,0]
                mask_area = m[m > 0] # 0-255
                mask_area_average = np.sum(mask_area)
                
                # check area of overwrap
                p = pred_masks[i]
                hit_area = p[m > 0] * 255 # 0-1 -> 0-255
                hit_area_average = np.sum(hit_area)
                ratio = hit_area_average / mask_area_average

                # fetch max ratio
                if max_ratio < ratio:
                    max_ratio = ratio
                    max_a = a
            
            # assign prediction to maximum area
            if max_a != -1:
                if is_accept_label(labels[i]):
                    if area_mask[max_a]["new_ratio"] < max_ratio:
                        area_mask[max_a]["new_ratio"] = max_ratio
                else:
                    if area_mask[max_a]["deny_ratio"] < max_ratio:
                        area_mask[max_a]["deny_ratio"] = max_ratio
        
    # set result
    for a in range(len(area_mask)):
        # not worked in pub data because dining tables always high deny ratio
        #if area_mask[a]["new_ratio"] < area_mask[a]["deny_ratio"]:
        #    area_mask[a]["ratio"] = area_mask[a]["ratio"] # use before state, for example, person crossing front of camera
        #else:
        area_mask[a]["ratio"] = area_mask[a]["new_ratio"]

def decide_label(area_mask):
    state_changed = False
    for a in range(len(area_mask)):
        label = "Empty"
        if area_mask[a]["ratio"] >= args.area_threshold:
            label = "Fill"
        if area_mask[a]["label"] != label:
            state_changed = True
        area_mask[a]["label"] = label
    return state_changed

# ======================
# Csv output
# ======================

def open_csv(area_mask):
    csv = open(args.csvpath, mode = 'w')
    csv.write("sec , time")
    for a in range(len(area_mask)):
        csv.write(" , " + area_mask[a]["id"])
    csv.write("\n")
    return csv

def write_csv(csv, fps_time, time_stamp, area_mask):
    csv.write(str(fps_time)+" , "+time_stamp)
    for a in range(len(area_mask)):
        if area_mask[a]["ratio"] >= args.area_threshold:
            label = "1"
        else:
            label = "0"
        csv.write(" , " + label)
    csv.write("\n")
    csv.flush()

def close_csv(csv):
    if csv is not None:
        csv.close()

# ======================
# Secondaty Functions
# ======================

def do_paste_mask(masks, boxes, im_h, im_w):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """

    x0_int, y0_int = 0, 0
    x1_int, y1_int = im_w, im_h
    x0, y0, x1, y1 = np.split(boxes, 4, axis=1)  # each is Nx1

    img_y = np.arange(y0_int, y1_int, dtype=np.float32) + 0.5
    img_x = np.arange(x0_int, x1_int, dtype=np.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1

    gx = np.repeat(img_x[:, None, :], img_y.shape[1], axis=1)
    gy = np.repeat(img_y[:, :, None], img_x.shape[1], axis=2)
    grid = np.stack([gx, gy], axis=3)

    img_masks = grid_sample(masks, grid, align_corners=False)

    return img_masks[:, 0]


def paste_masks_in_image(
        masks, boxes, image_shape, threshold: float = 0.5):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.
    """

    if len(masks) == 0:
        return np.zeros((0,) + image_shape, dtype=np.uint8)

    im_h, im_w = image_shape

    img_masks = do_paste_mask(
        masks[:, None, :, :], boxes, im_h, im_w,
    )
    img_masks = img_masks >= threshold

    return img_masks


def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.

    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False

    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]

    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]

    return res, has_holes


def draw_predictions(img, predictions):
    vocabulary = args.vocabulary

    height, width = img.shape[:2]

    boxes = predictions["pred_boxes"].astype(np.int64)
    scores = predictions["scores"]
    classes = predictions["pred_classes"].tolist()
    masks = predictions["pred_masks"].astype(np.uint8)

    class_names = (
        get_lvis_meta_v1() if vocabulary == 'lvis' else get_in21k_meta_v1()
    )["thing_classes"]
    # labels = [class_names[i] for i in classes] # onnx runtime
    labels = [class_names[int(i)] for i in classes]  # ailia always returns float tensor so need to add cast
    labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

    num_instances = len(boxes)

    np.random.seed(args.seed)
    assigned_colors = [random_color(maximum=255) for _ in range(num_instances)]

    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    if areas is not None:
        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs]
        labels = [labels[k] for k in sorted_idxs]
        masks = [masks[idx] for idx in sorted_idxs]
        assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

    default_font_size = int(max(np.sqrt(height * width) // 90, 10))

    for i in range(num_instances):
        color = assigned_colors[i]
        color = (int(color[0]), int(color[1]), int(color[2]))

        if is_accept_label(labels[i]):
            alpha = 0.5
        else:
            alpha = 0.25
            color = (0, 0, 0)

        img_b = img.copy()

        # draw box
        x0, y0, x1, y1 = boxes[i]
        #cv2.rectangle(
        #    img_b, (x0, y0), (x1, y1),
        #    color=color,
        #    thickness=default_font_size // 4)

        # draw segment
        polygons, _ = mask_to_polygons(masks[i])
        for points in polygons:
            points = np.array(points).reshape((1, -1, 2)).astype(np.int32)
            cv2.fillPoly(img_b, pts=[points], color=color)

        img = cv2.addWeighted(img, 1.0 - alpha, img_b, alpha, 0)

    for i in range(num_instances):
        color = assigned_colors[i]
        color_text = color_brightness(color, brightness_factor=0.7)

        color = (int(color[0]), int(color[1]), int(color[2]))
        color_text = (int(color_text[0]), int(color_text[1]), int(color_text[2]))

        x0, y0, x1, y1 = boxes[i]

        SMALL_OBJECT_AREA_THRESH = 1000
        instance_area = (y1 - y0) * (x1 - x0)

        # for small objects, draw text at the side to avoid occlusion
        text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
        if instance_area < SMALL_OBJECT_AREA_THRESH or y1 - y0 < 40:
            if y1 >= height - 5:
                text_pos = (x1, y0)
            else:
                text_pos = (x0, y1)

        # draw label
        x, y = text_pos
        text = labels[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        height_ratio = (y1 - y0) / np.sqrt(height * width)
        font_scale = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5)
        font_thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, text_pos, (int(x + text_w * 0.6), y + text_h), (0, 0, 0), -1)
        cv2.putText(
            img, text, (x, y + text_h - 5),
            fontFace=font,
            fontScale=font_scale * 0.6,
            color=color_text,
            thickness=font_thickness,
            lineType=cv2.LINE_AA)

    return img


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RGB

    size = args.detection_width
    max_size = args.detection_width
    scale = size / min(im_h, im_w)
    if im_h < im_w:
        oh, ow = size, scale * im_w
    else:
        oh, ow = scale * im_h, size
    if max(oh, ow) > max_size:
        scale = max_size / max(oh, ow)
        oh = oh * scale
        ow = ow * scale
    ow = int(ow + 0.5)
    oh = int(oh + 0.5)

    img = np.asarray(Image.fromarray(img).resize((ow, oh), Image.BILINEAR))

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(
        pred_boxes, scores, pred_classes, pred_masks, im_hw, pred_hw):
    scale_x, scale_y = (
        im_hw[1] / pred_hw[1],
        im_hw[0] / pred_hw[0],
    )

    pred_boxes[:, 0::2] *= scale_x
    pred_boxes[:, 1::2] *= scale_y
    pred_boxes[:, [0, 2]] = np.clip(pred_boxes[:, [0, 2]], 0, im_hw[1])
    pred_boxes[:, [1, 3]] = np.clip(pred_boxes[:, [1, 3]], 0, im_hw[0])

    threshold = 0
    widths = pred_boxes[:, 2] - pred_boxes[:, 0]
    heights = pred_boxes[:, 3] - pred_boxes[:, 1]
    keep = (widths > threshold) & (heights > threshold)

    pred_boxes = pred_boxes[keep]
    scores = scores[keep]
    pred_classes = pred_classes[keep]
    pred_masks = pred_masks[keep]

    mask_threshold = 0.5
    pred_masks = paste_masks_in_image(
        pred_masks[:, 0, :, :], pred_boxes,
        (im_hw[0], im_hw[1]), mask_threshold
    )

    pred = {
        'pred_boxes': pred_boxes,
        'scores': scores,
        'pred_classes': pred_classes,
        'pred_masks': pred_masks,
    }
    return pred


def predict(net, img):
    im_h, im_w = img.shape[:2]
    img = preprocess(img)
    pred_hw = img.shape[-2:]
    im_hw = np.array([im_h, im_w]).astype(np.int64)
    #img[:] = 0 # test for grid sampler

    # feedforward
    if args.opset16:
        output = net.predict([img, im_hw])
    else:
        output = net.predict([img])

    pred_boxes, scores, pred_classes, pred_masks = output

    if not args.opset16:
        pred = post_processing(
            pred_boxes, scores, pred_classes, pred_masks,
            (im_h, im_w), pred_hw
        )
    else:
        pred = {
            'pred_boxes': pred_boxes,
            'scores': scores,
            'pred_classes': pred_classes,
            'pred_masks': pred_masks,
        }

    return pred


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None
    fps = capture.get(cv2.CAP_PROP_FPS)

    frame_shown = False
    area_mask = None
    csv = None
    before_fps_time = -1
    frame_no = 0

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        global terminate_signal
        if terminate_signal:
            break
    
        # timestamp
        time_stamp = str(datetime.datetime.now())

        # inference
        pred = predict(net, frame)

        # prepare mask
        if area_mask == None:
            area_mask = prepare_area_mask(frame)
            if args.csvpath != None:
                csv = open_csv(area_mask)
            else:
                csv = None

        # check area
        check_area_overwrap(pred["pred_masks"].astype(np.uint8), pred["pred_classes"].tolist(), area_mask)

        # decide label
        state_changed = decide_label(area_mask)

        # draw area
        res_img = frame.copy()
        display_area(res_img, area_mask)

        # draw prediction
        res_img = draw_predictions(res_img, pred)

        # draw text
        display_text(res_img, area_mask)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = cv2.resize(res_img, (f_w, f_h))
            writer.write(res_img.astype(np.uint8))
        fps_time = int(frame_no / fps)
        if csv is not None:
            if before_fps_time != fps_time:
                write_csv(csv, fps_time, time_stamp, area_mask)
                before_fps_time = fps_time
        
        # save frame
        if state_changed:
            if args.imgpath:
                path = time_stamp
                path = path.replace(" ","-")
                path = path.replace(".","-")
                path = path.replace(":","-")
                path = args.imgpath+"/"+path+".jpg"
                cv2.imwrite(path, res_img)

        frame_no = frame_no + 1

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    if csv is not None:
        close_csv(csv)

    logger.info('Script finished successfully.')


def main():
    set_signal_handler()

    if args.opset16:
        dic_model = {
            ('SwinB_896_4x', 'lvis'): (WEIGHT_SWINB_LVIS_OP16_PATH, MODEL_SWINB_LVIS_OP16_PATH),
            ('SwinB_896_4x', 'in21k'): (WEIGHT_SWINB_IN21K_OP16_PATH, MODEL_SWINB_IN21K_OP16_PATH),
            ('R50_640_4x', 'lvis'): (WEIGHT_R50_LVIS_OP16_PATH, MODEL_R50_LVIS_OP16_PATH),
            ('R50_640_4x', 'in21k'): (WEIGHT_R50_IN21K_OP16_PATH, MODEL_R50_IN21K_OP16_PATH),
        }
    else:
        dic_model = {
            ('SwinB_896_4x', 'lvis'): (WEIGHT_SWINB_LVIS_PATH, MODEL_SWINB_LVIS_PATH),
            ('SwinB_896_4x', 'in21k'): (WEIGHT_SWINB_IN21K_PATH, MODEL_SWINB_IN21K_PATH),
            ('R50_640_4x', 'lvis'): (WEIGHT_R50_LVIS_PATH, MODEL_R50_LVIS_PATH),
            ('R50_640_4x', 'in21k'): (WEIGHT_R50_IN21K_PATH, MODEL_R50_IN21K_PATH),
        }
    key = (args.model_type, args.vocabulary)
    WEIGHT_PATH, MODEL_PATH = dic_model[key]

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # disable FP16
    if "FP16" in ailia.get_environment(args.env_id).props or platform.system() == 'Darwin':
        logger.warning('This model do not work on FP16. So use CPU mode.')
        args.env_id = 0

    # initialize
    if args.env_id == 0:
        # CPU supporting reuse_interstage
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=False, reuse_interstage=True)
    else:
        # cuDNN only worked with reduce_interstage
        logger.info("GPU only worked with reduce_interstage")
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=True, reuse_interstage=False)
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=memory_mode)

    recognize_from_video(net)


if __name__ == '__main__':
    main()
