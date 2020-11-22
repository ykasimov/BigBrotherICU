from collections import Counter
import json
import cv2
import pickle
import os
from predict_staff import predict_class

root_folder = '/data/ikem_hackathon/cuts/frames_detections'


def process_all_folders():
    for idx, dir in enumerate(os.listdir(root_folder)):
        print(dir)
        # if dir.startswith('nurse_and_night'):
        process_folder(f'{root_folder}/{dir}', 4, safe_people=False, predict=True)


def process_folder(dir, idx, safe_people=True, predict=False):
    result_folder = 'people_cuts'
    if safe_people:
        os.makedirs(result_folder, exist_ok=True)
    image_folder = f'{dir}/pictures'
    image_folder_prediction = f'{dir}/detection'
    bbox_folder = f'{dir}/logs'
    prediction_folder = f'{dir}/predictions'
    print(prediction_folder)
    os.makedirs(prediction_folder, exist_ok=True)
    for file in os.listdir(image_folder):
        image = cv2.imread(f'{image_folder}/{file}')
        image_to_draw = cv2.imread(f'{image_folder_prediction}/{file}')
        with open(f'{bbox_folder}/{file}', 'rb') as f:
            boxes = pickle.load(f)
        image_predictions = []
        for person, box in crop_people(image, boxes):
            if safe_people:
                cv2.imwrite(f'{result_folder}/{idx}_{file}', person)
            if predict:
                prediction = predict_class(person)
                image_predictions.append(prediction)
                image = draw_bbox(image_to_draw, box, prediction)
                if prediction == "Patient":
                    image = blur_or_blacken(image, box, blur=True)
        if predict:
            people_counter = Counter(image_predictions)
            with open(f'{prediction_folder}/{file}.personnel', 'w') as f:
                json.dump(people_counter, f)
            cv2.imwrite(f'{prediction_folder}/{file}', image)


def draw_bbox(image, bbox, label):
    x_l, y_l, x_r, y_r = tuple(map(int, bbox))
    pt1 = (x_l, y_l)
    pt2 = (x_r, y_r)
    image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), 3)
    cv2.putText(image, label, (x_l, y_l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image


def crop_people(image, boxes):
    for bbox, obj_class in zip(boxes['bbox'], boxes['classes']):
        if obj_class != 'person':
            continue
        x_l, y_l, x_r, y_r = tuple(map(int, bbox))
        cut = image[y_l:y_r, x_l:x_r]
        yield cv2.resize(cut, (128, 128)), bbox


def blur_or_blacken(image, bb, blur=True, copy=False):
    x_l, y_l, x_r, y_r = tuple(map(int, bb))
    if copy:
        img = image
        image = cv2.copy(img)
    if blur:
        src = image[y_l:y_r, x_l:x_r]
        dst = cv2.GaussianBlur(src, (31, 31), sigmaX=100,sigmaY=100)
        image[y_l:y_r, x_l:x_r] = dst
    else:
        src = image[y_l:y_r, x_l:x_r]
        src[:] = (0, 0, 0)
    return image


if __name__ == "__main__":
    # process_all_folders()
    process_folder(f'{root_folder}/patient_sits_and_eats_lunch', 'doctor', safe_people=False, predict=True)
