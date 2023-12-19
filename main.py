import os
import sys
import torch
import argparse
from PIL import Image
from strhub.data.module import SceneTextDataModule
# import east.detect
# from east.model import EAST
import cv2
import pandas as pd
from strhub.models.utils import load_from_checkpoint
import time
import numpy as np
import torchvision.transforms as transforms
import ahocorasick 
import cProfile, pstats
import Levenshtein
from shapely.geometry import Polygon
from scipy import io
from os import listdir

from multiprocessing import Process, Manager, Lock
from multiprocessing import set_start_method, get_start_method
from joblib import Parallel, delayed
import eval_fast as eval_fast


# from cv2.dnn import TextDetectionModel_DB
# import easyocr


import mmcv
from mmcv import Config
from FAST.models import build_model
from FAST.models.utils import fuse_module, rep_model_convert
from FAST.dataset.utils import scale_aligned_short
from FAST.dataset import build_data_loader
from FAST.utils import ResultFormat, AverageMeter
import json

import string
import sys
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

# set_start_method('spawn')


def convert_to_xywh(bbox):
    """
    Convert a bounding box from (x1, y1, x2, y2, x3, y3, x4, y4) format to (x, y, w, h) format.

    Args:
        bbox (list or tuple): A list or tuple containing the four corner coordinates.

    Returns:
        tuple: A tuple in (x, y, w, h) format.
    """
    # Extract the corner coordinates

    if(len(bbox) == 9):

     x1, y1, x2, y2, x3, y3, x4, y4, angle = bbox

    else:
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox
        angle = 0

    # Compute the top-left and bottom-right coordinates
    x = min(x1, x2, x3, x4)
    y = min(y1, y2, y3, y4)
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)

    # Calculate the width and height
    w = x_max - x
    h = y_max - y


    return x, y, w, h, angle



@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)



@torch.inference_mode()
def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    charset_test = string.digits + string.ascii_lowercase
    model, img_transform = get_parseq(device)
    hp = model.hparams

    datamodule = SceneTextDataModule("./data", '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, 512, 4, False, 0)

    test_set = SceneTextDataModule.TEST_NEW
    # test_set = sorted(set(test_set))

    results = {}
    max_width = max(map(len, test_set))

    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0

        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)

    result_groups = dict()
    result_groups.update({'New': SceneTextDataModule.TEST_NEW})

    with open('TEST_OUTPUT.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)


def get_parseq(device):
    # parseq = torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval()
    # parseq = torch.load('./model/parseq_small.bin', map_location=torch.device('cpu'))

    parseq = (
        load_from_checkpoint(
            "pretrained=parseq-tiny",
        )
        .eval()
        .to(device)
    )

    size = parseq.hparams.img_size

    img_transform = SceneTextDataModule.get_transform(size)
    # img_transform = SceneTextDataModule.get_transform([32, 128])

    return parseq, img_transform


def get_sensitive_list():
    profanity_list = set(pd.read_csv("profanity.csv", usecols=[0])["text"].str.lower())
    # profanity_list = set()
    user_word_list = set(pd.read_csv("sensitive.csv", usecols=[0])["text"].str.lower())
    print(user_word_list)

    sensitive_list = profanity_list.union(user_word_list)

    aho = ahocorasick.Automaton()
    for word in sensitive_list:
        aho.add_word(word, word)

    aho.make_automaton()

    return aho


def read_img(path):
    img_og = cv2.imread(path)

    # img_og = img_og.resize(640,640)

    img = Image.fromarray(cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB))

    # img = img.resize(640,640)

    return img, img_og



# def east_boxes(img, eastmodel, device):

#     boxes = east.detect.detect(img, eastmodel, device)

#     return boxes


def process_img(img):

    data = dict()

    img = np.array(img)

    img_meta = dict(
        org_img_size=np.array(img.shape[:2])
    )

    img = scale_aligned_short(img, 640)

    img_meta.update(dict(
        img_size=(np.array(img.shape[:2])).tolist(),
        filename="test"
    ))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    data['img_metas'] = img_meta

    # p_img = torch.from_numpy(img).permute(2,1,0).float()
    img = img.unsqueeze(0)
    data['imgs'] = img
    data['imgs'] = data['imgs'].cuda(non_blocking=True)

    return data




def fast_boxes(data, model, device):

    with torch.no_grad():
        outputs = model(**data)

    # print(type(outputs))

    results = outputs['results']

    # print(results)

    bboxes = results[0]['bboxes']

    return bboxes



def get_fast(device):
    # parser = argparse.ArgumentParser(description='Hyperparams')
    # parser.add_argument('config', default='./FAST/config/fast/tt/fast_base_tt_640_finetune_ic17mlt.py' ,help='config file path')
    # parser.add_argument('checkpoint', nargs='?', type=str, default='./FAST/pretrained/fast_base_tt_640_finetune_ic17mlt.pth')
    # parser.add_argument('--report-speed', action='store_true')
    # parser.add_argument('--print-model', action='store_true')
    # parser.add_argument('--min-score', default=None, type=float)
    # parser.add_argument('--min-area', default=None, type=int)
    # parser.add_argument('--batch-size', default=1, type=int)
    # parser.add_argument('--worker', default=4, type=int)
    # parser.add_argument('--ema', action='store_true')
    # parser.add_argument('--cpu', action='store_true')

    # args = parser.parse_args()
    # cfg = Config.fromfile(args.config)
    cfg = Config.fromfile('./FAST/config/fast/tt/fast_tiny_tt_640_finetune_ic17mlt.py')

    model = build_model(cfg.model)
    model = model.cuda()
    # checkpoint = torch.load(args.checkpoint)
    checkpoint = torch.load('./FAST/pretrained/fast_tiny_tt_640_finetune_ic17mlt.pth')
    # checkpoint = torch.load('./modified_chkpt.pth')
    # print(next(iter(checkpoint)))
    # state_dict = checkpoint['state_dict']
    # state_dict = checkpoint
    state_dict = checkpoint['ema']

    d = dict()
    for key, value in state_dict.items():
        tmp = key.replace("module.", "")
        d[tmp] = value

    model.load_state_dict(d)
    model = rep_model_convert(model)
    model = fuse_module(model)
    model.eval()


    return model




def do_recognition(parseq, img_transform, img, boxes, device):
    detected_texts = []
    bounding_boxes = []
    index = 0
    # Process each text box individually
    region_images = []

    for box in boxes:
        # print(box)
        if (len(box)<8):

            x, xpw, y, yph = box
            w = xpw - x
            h = yph - y

        else:
            x, y, w, h, angle = convert_to_xywh(box)  # Bounding box coordinates
        
        region_img = img.crop((x, y, x + w, y + h))  # Crop the region from the image

        # Preprocess the region_image as needed (resize, normalize, etc.)
        region_img = img_transform(region_img).to(device)

        # region_img =  region_img.squeeze(0)
        # print(region_img.shape)

        bounding_boxes.append([x, y, x + w, y + h])
        region_images.append(region_img)

    with torch.no_grad():
        region_images = torch.stack(region_images, dim=0)
        # print(region_images.shape)

        logits = parseq(region_images)
        logits.shape  # torch.Size([batch_size, seq_len, num_classes])

        # Greedy decoding
        preds = logits.softmax(-1)
        # labels, confidences = parseq.tokenizer.decode(logits)
        labels, confidences = parseq.tokenizer.decode(preds)

    for i in range(len(labels)):
        detected_texts.append([labels[i], confidences[i][0]])
        print(
            [
                "Decoded label = {}".format(detected_texts[i][0]),
                "confidence = {}".format(detected_texts[i][1]),
                "bounding boxes = {}".format(bounding_boxes[i]),
            ]
        )

    return detected_texts, bounding_boxes




def blur_text(detected_texts, bounding_boxes, img_og, sensitive_list):
    bbox_img = img_og.copy()
    blur_img = img_og.copy()
    rec_img  =img_og.copy()

    for i in range(len(detected_texts)):
        word = detected_texts[i][0]
        # print(detected_texts[i][1])
        # confidence_tensor = detected_texts[i][1]
        # confidence = confidence_tensor[0]
        # confidence = confidence_tensor

        # wordset = set(word_tokenize(word))
        # matches = sensitive_list.intersection(wordset)

        # if word.lower() in sensitive_list:

        # print(word.lower())

        # for _, sub in sensitive_list.iter(word.lower()):


            # if word == 'Kawaiifuu':
            #     print(sub)
        
        x, y, xpw, yph = (
                int(bounding_boxes[i][0]),
                int(bounding_boxes[i][1]),
                int(bounding_boxes[i][2]),
                int(bounding_boxes[i][3]),
        )

        cv2.rectangle(rec_img, (x, y), (xpw, yph), (0, 0, 255), 2)
        cv2.rectangle(rec_img, (x,y - (yph - y)), (xpw, y), (0,255,0), thickness=cv2.FILLED)
        cv2.putText(
                rec_img,
                word,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )


        

        if any(sub.lower() in word.lower() for _, sub in sensitive_list.iter(word.lower())) or any(Levenshtein.distance(word.lower(), x) == 1 for x in sensitive_list):

        
            x, y, xpw, yph = (
                int(bounding_boxes[i][0]),
                int(bounding_boxes[i][1]),
                int(bounding_boxes[i][2]),
                int(bounding_boxes[i][3]),
            )
            print("removing: " + word + " | bbox: {}".format(bounding_boxes[i]))
            roi = img_og[y:yph, x:xpw]
            text_height = yph - y
            kernel_size = max(1, int(1.5 * text_height))
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            img_og[y:yph, x:xpw] = blurred_roi
            cv2.rectangle(img_og, (x, y), (xpw, yph), (0, 0, 255), 2)

            # text = f"{confidence:.3f}"
            text = word
            cv2.rectangle(blur_img, (x, y), (xpw, yph), (0, 0, 255), 2)
            cv2.rectangle(blur_img, (x,y - (yph - y)), (xpw, y), (0,255,0), thickness=cv2.FILLED)
            cv2.putText(
                blur_img,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )




    for i in range (len(bounding_boxes)):
        x, y, xpw, yph = (
                int(bounding_boxes[i][0]),
                int(bounding_boxes[i][1]),
                int(bounding_boxes[i][2]),
                int(bounding_boxes[i][3]),
        )

        cv2.rectangle(bbox_img, (x, y), (xpw, yph), (0, 0, 255), 2)


    return img_og, blur_img, bbox_img, rec_img



def get_video(path):
    video = cv2.VideoCapture(path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))





def process_video(path, parseq, img_transform, fast_model, sensitive_list, device):

    start_time = time.time()

    cap = cv2.VideoCapture(path)

    output_video_path = './output/output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=True)

    tt = (time.time() - start_time)
    print(f"writer initialized {tt}")

    frame_count = 0
    start_time = time.time()
    fps_list = []

    while True:

        # Read the next frame
        ret, frame_og = cap.read()

        # Check if the video has ended
        if not ret:
            break

        frame = Image.fromarray(cv2.cvtColor(frame_og, cv2.COLOR_BGR2RGB))

        data = process_img(frame)
        t = (time.time() - start_time) - tt
        tt = (time.time() - start_time)
        print(f"image transformed {t}")


        boxes = fast_boxes(data, fast_model, device)
        t = (time.time() - start_time) - tt
        tt = (time.time() - start_time)
        print(f"FAST output {t}")


        detected_texts, bounding_boxes = do_recognition(parseq, img_transform, frame, boxes, device) 
        t = (time.time() - start_time) - tt
        tt = (time.time() - start_time)
        print(f"parseq output {t}")

        blurred_frame, blur_bbox_frame, det_bbox_frame, rec_frame = blur_text(detected_texts, bounding_boxes, frame_og, sensitive_list)
        t = (time.time() - start_time) - tt
        tt = (time.time() - start_time)
        print(f"blurring done {t}")

        out.write(blurred_frame)    
        t = (time.time() - start_time) - tt
        tt = (time.time() - start_time)
        print(f"frame written {t}")

        frame_count += 1

        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = 10 / elapsed_time
            fps_list.append(fps)
            start_time = time.time()

    average_fps = sum(fps_list) / len(fps_list)

    print(f"Average FPS: {average_fps:.2f}")

    out.release()
    cap.release()




def process_single(parseq, img_transform, model, sensitive_list, device):


    start_time = time.time()

    img, img_og = read_img("./sample_images/class.jpg")
    # img_list, img_og_list = get_test_images("./sample_images/")
    # t = (time.time() - start_time) - t
    # tt = time.time() - start_time
    # print(f"loading done {tt}")


    data = process_img(img)
    # data = process_img_list(img_list)
    # t = (time.time() - start_time) - tt
    tt = (time.time() - start_time)
    print(f"Img Processed {tt}")


    boxes = fast_boxes(data, model, device)

    t = (time.time() - start_time) - tt
    tt = (time.time() - start_time)
    print(f"FAST output {t}")



    detected_texts, bounding_boxes = do_recognition(parseq, img_transform, img, boxes, device)
    t = (time.time() - start_time) - tt
    tt = (time.time() - start_time)
    print(f"parseq op {t}")


    blurred_img, blur_bbox_image, det_bbox_img, rec_img = blur_text(
        detected_texts, bounding_boxes, img_og, sensitive_list
    )
    t = (time.time() - start_time) - tt

    print(f"blurring done {t}")

    cv2.imwrite("./output/op_img.jpg", blurred_img)
    cv2.imwrite("./output/op_blur_bbox_img.jpg", blur_bbox_image)
    cv2.imwrite("./output/op_det_bbox_img.jpg", det_bbox_img)
    cv2.imwrite("./output/rec_img.jpg", rec_img)



def list_gpu_info():
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_info.append((i, gpu_name))
    

    if gpu_info:
        for gpu_id, gpu_name in gpu_info:
            print(f"GPU {gpu_id}: {gpu_name}")
    else:
        print("No GPUs available.")




def test_fast():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_fast(device)

    cfg = Config.fromfile("./FAST/config/fast/tt/fast_base_tt_640_finetune_ic17mlt.py")
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )


    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    results = dict()

    for idx, data in enumerate(test_loader):
        print('Testing %d/%d\r' % (idx, len(test_loader)), flush=True, end='')
        # prepare input

        data['imgs'] = data['imgs'].cuda(non_blocking=True)

        # data.update(dict(cfg=cfg))
        # forward
        with torch.no_grad():
            outputs = model(**data)

        # save result
        image_names = data['img_metas']['filename']
        for index, image_name in enumerate(image_names):
            rf.write_result(image_name, outputs['results'][index])
            results[image_name] = outputs['results'][index]

    results = json.dumps(results)
    with open('./output.json', 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False)
        print("write json file success!")


    # fast_eval.do_eval()




def main():
    list_gpu_info()


    start_time = time.time()
    print(f"loading {(time.time() - start_time)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fast_model = get_fast(device)


    parseq, img_transform = get_parseq(device)
    # t = (time.time() - start_time) - t
    # print(f"got parseq {t}")


    sensitive_list = get_sensitive_list()
    # t = (time.time() - start_time) - t
    # print(f"got list {t}")

    tt = time.time() - start_time
    print(f"loading done {tt}")



    process_single(parseq, img_transform, fast_model, sensitive_list, device)

    # process_video('./video_eval/subtitled/poke.mp4', parseq, img_transform, fast_model, sensitive_list, device)




if __name__ == '__main__':

    # test_fast()

    # eval_fast.do_eval()

    main()

    # processing()

    # test()


