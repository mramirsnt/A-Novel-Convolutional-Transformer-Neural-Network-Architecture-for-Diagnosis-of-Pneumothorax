import torch
import torchvision.datasets
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torchvision.models import  vgg16_bn , vgg16
from settings import TRAIN_FOLDER, TEST_FOLDER, BALANCED_PATH, RGB_BALANCED, RGB_PREPROCESSED_BALANCED, MINIMAL_RGB,NUM_CLASSES
import torch.nn as nn
from tqdm import tqdm
from pgu_metrics import PGU_Metrics
import torch.optim as optim
from settings import BATCH_SIZE, checkpoint_filepath, RGB_DIRECTORY_PATH, RGB_CLAHE_PATH, RANDOM_SEED,EPOCHS
from call_backs import SaveTheBestCallBack

dir = RGB_CLAHE_PATH
def load_data():
    train_dir = dir + '/train/'
    val_dir = dir + '/validation/'
    test_dir = dir + '/test/'
    feature_extractor = transforms.Compose([transforms.ToTensor(),transforms.Resize(256),transforms.CenterCrop(224),transforms.Normalize(
    mean=[0.48235, 0.45882, 0.40784],
    std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
    )])
    train_dataset = torchvision.datasets.ImageFolder(train_dir,transform=feature_extractor)
    val_dataset = torchvision.datasets.ImageFolder(val_dir,transform=feature_extractor)
    test_dataset = torchvision.datasets.ImageFolder(test_dir,transform=feature_extractor)
# print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return  train_loader,val_loader,test_loader

def create_model():
    model = vgg16_bn(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=NUM_CLASSES)
    return model

def evaluate(model, data_set, data_type='train', num_classes = 2, added_metrics=[]):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pgmetrics = PGU_Metrics(num_classes=num_classes, threshold=0.5, positive_class=1)

    predicted_values = []
    true_values = []
    with torch.no_grad():
        for data in data_set:

            input,label = data
            input = input.to(device)
            label = label.to(device)
            model = model.to(device)
            outputs = model(input)
            true_values.extend(label.tolist())
            predicted_values.extend(outputs.tolist())
        res = pgmetrics.calculate_metrics(added_metrics=added_metrics, data_type=data_type, real_value=true_values,
                            predicted_value=predicted_values)
        return res



def train(model,trainloader,valloader=None,epochs=0,added_metrics=[],device="cuda" if torch.cuda.is_available() else "cpu",epoch_call_backs=[]):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    train_steps = len(trainloader) // trainloader.batch_size


    tepoch = tqdm(None, smoothing=0, total=train_steps,
                  disable=False, leave=True, dynamic_ncols=True)

    pgmetrics = PGU_Metrics(num_classes=NUM_CLASSES, threshold=0.5, positive_class=1)
    for i in range(epochs):
        tepoch.reset(train_steps)
        tepoch.initial = 0
        predicted_values = []
        true_values = []


        for counter,data in enumerate(trainloader) :
            input,label = data
            input = input.to(device)
            label = label.to(device)
            model = model.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()

            true_values.extend(label.tolist())
            predicted_values.extend(output.tolist())
            tepoch.set_description(f"Epoch {i}")
            tepoch.set_postfix({'loss': loss.item()})  # , accuracy=100. * accuracy)
            tepoch.update()
        res = pgmetrics.calculate_metrics(added_metrics=added_metrics, data_type='train', real_value=true_values,
                                          predicted_value=predicted_values)
        print(f'counter = {counter}')
        val_res = None
        if valloader is not None:
            val_res = evaluate(model, added_metrics=added_metrics, data_type='validation', data_set=valloader)
            current_value = val_res['val_accuracy']
            for callback in epoch_call_backs:
                if type(callback) is SaveTheBestCallBack:
                    callback.run(current_value)
        if val_res is None:
            print(res)
        else:
            print(res, val_res)
    return model


train_loader,val_loader,test_loader = load_data()
model = create_model()
print(len(test_loader))
# save_the_best_callback = SaveTheBestCallBack(model=model, target_file=checkpoint_filepath, less_is_better=False)
# train(model,train_loader,valloader=val_loader,epochs=EPOCHS,added_metrics=['auroc', 'f1_score', 'precision', 'specificity'],epoch_call_backs=[save_the_best_callback])