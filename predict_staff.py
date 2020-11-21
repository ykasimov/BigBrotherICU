from PIL import Image
import torch
from people_classifier import Net, transform

PATH = './models/med_personel_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

classes = {0: 'Doctor', 1: 'Nurse', 2: 'Patient'}


def predict(raw_image):
    image = transform(raw_image).float().unsqueeze(0)
    output = net(image)
    _, prediction = torch.max(output.data, 1)
    print(classes[prediction[0].item()])


if __name__ == '__main__':
    image = Image.open('/data/ikem_hackathon/cuts/people_cuts/nurse/4_000049.jpg')
    predict(image)
