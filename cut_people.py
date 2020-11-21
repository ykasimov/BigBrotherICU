import cv2
import pickle
import os

root_folder = '/data/ikem_hackathon/cuts/frames_detections'


def process_all_folders():
    for idx, dir in enumerate(os.listdir(root_folder)):
        print(dir)
        if dir.startswith('nurse_and_night'):
            crop_folder(f'{root_folder}/{dir}', 4)


def crop_folder(dir, idx):
    result_folder = 'people_cuts'
    os.makedirs(result_folder, exist_ok=True)
    image_folder = f'{dir}/pictures'
    bbox_folder = f'{dir}/logs'
    for file in os.listdir(image_folder):
        image = cv2.imread(f'{image_folder}/{file}')
        with open(f'{bbox_folder}/{file}', 'rb') as f:
            boxes = pickle.load(f)
        for person in crop_people(image, boxes):
            cv2.imwrite(f'{result_folder}/{idx}_{file}', person)


def crop_people(image, boxes):
    for bbox, obj_class in zip(boxes['bbox'], boxes['classes']):
        if obj_class != 'person':
            continue
        x_l, y_l, x_r, y_r = tuple(map(int, bbox))
        cut = image[y_l:y_r, x_l:x_r]
        yield cv2.resize(cut, (128, 128))


if __name__ == "__main__":
    process_all_folders()
