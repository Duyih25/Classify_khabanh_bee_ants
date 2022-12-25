from data_processing.Load_data import Load_data
from lib import *
from constant import *


def params_to_update(net):
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    update_param_name_1 = ["features"]
    update_param_name_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_name_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in net.named_parameters():
        if name in update_param_name_1:
            param.requires_grad = True
            params_to_update_1.append(param)
        elif name in update_param_name_2:
            param.requires_grad = True
            params_to_update_2.append(param)
        elif name in update_param_name_3:
            param.requires_grad = True
            params_to_update_3.append(param)

    return params_to_update_1, params_to_update_2, params_to_update_3

class Vgg16_fine_tuning:
    def __init__(self):
        self.net = models.vgg16(pretrained=USE_PRETRAINED)
        self.net.classifier[6] = nn.Linear(in_features=4096, out_features=3, bias=True)
        self.criterior = nn.CrossEntropyLoss()
        self.optimizer = None



    def train_model(self, save_path, dataloader_dict, num_epochs=1):
        params1, params2, params3 = params_to_update(self.net)
        self.optimizer = optim.SGD([
            {'params': params1, 'lr': 1e-5},
            {'params': params2, 'lr': 1e-5},
            {'params': params3, 'lr': 1e-3},
        ], lr=0.001, momentum=0.9)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)

        best_acc = 0

        self.net.to(device)

        for epoch in range(num_epochs + 1):
            if epoch == 0:
                print()
                continue
            print("Epoch {}/{} ".format(epoch, num_epochs))

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.net.train()

                else:
                    self.net.eval()

                epoch_loss = 0.0
                epoch_corrects = 0

                for inputs, labels in tqdm(dataloader_dict[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.net(inputs)

                        loss = self.criterior(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                        epoch_loss += loss.item() * inputs.size(0)
                        epoch_corrects += torch.sum(preds == labels.data)

                epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
                epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

                print("{} Loss: {:.4f}  Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

                if epoch_acc > best_acc and phase == 'val':
                    best_acc = epoch_acc
                    torch.save(self.net.state_dict(), save_path)


if __name__ == "__main__":
    ld = Load_data()
    train_loader = ld(phase='train')
    val_loader = ld(phase='val')

    dataloader_dict = {
        'train': train_loader,
        'val': val_loader
    }
    print("Loading model ...")
    model = Vgg16_fine_tuning()
    print("Finish loading ...")
    print()
    print("Start training ...")
    model.train_model(save_path=SAVE_PATH, dataloader_dict=dataloader_dict, num_epochs=5)
