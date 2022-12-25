import numpy as np

from Vgg16 import Vgg16_fine_tuning, params_to_update
from constant import LABELS
from data_processing.ImageTransform import ImageTransform
from data_processing.Load_data import Load_data
from lib import *
from sklearn.metrics import classification_report


def load_model(model_path):
    net = Vgg16_fine_tuning().net
    load_weights = torch.load(model_path, map_location={"cuda:0": "cpu"})
    net.load_state_dict(load_weights)

    return net


def predict_after_loading(dataloader):
    net = load_model('saved_model/vgg16_fine_tuning.pth')
    net.eval()

    epoch_loss = 0.0
    epoch_corrects = 0
    ans = []
    truth = []
    criterior = nn.CrossEntropyLoss()

    for inputs, labels in tqdm(dataloader):
        inputs = inputs
        labels = labels

        outputs = net(inputs)

        loss = criterior(outputs, labels)

        _, preds = torch.max(outputs, 1)
        ans += preds.numpy().tolist()
        truth += labels.numpy().tolist()

        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(preds == labels.data)

    epoch_loss = epoch_loss / len(dataloader.dataset)
    epoch_acc = epoch_corrects.double() / len(dataloader.dataset)

    print(" Loss: {:.4f}  Acc: {:.4f}".format(epoch_loss, epoch_acc))
    print(classification_report(truth, ans, digits = 3))

def predict_an_img(img):
    net = load_model('saved_model/vgg16_fine_tuning.pth')
    net.eval()

    transform = ImageTransform()
    img = transform(img, phase = 'test')
    img = img.unsqueeze_(0)

    output = net(img)
    max_id = np.argmax(output.detach().numpy())

    return LABELS[max_id]


if __name__ == "__main__":
    ld = Load_data()
    val_loader = ld(phase='val')

    predict_after_loading(val_loader)
