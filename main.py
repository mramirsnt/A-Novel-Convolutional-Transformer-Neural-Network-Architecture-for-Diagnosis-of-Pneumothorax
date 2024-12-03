import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from explainability.vit_explain import visualize_explain
from pgu_train_evaluate import evaluate, fit

from settings import MINIMAL_RGB, RGB_DIRECTORY_PATH, RGB_BALANCED, TRAIN_FOLDER, TEST_FOLDER, VALIDATION_FOLDER
from settings import checkpoint_filepath, CRC_DIRECTORY_PATH
from settings import BATCH_SIZE, EPOCHS
from settings import RANDOM_SEED
from load_dataset import BaseClassLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler, ViTFeatureExtractor
from vit_classifier import ViTClassifier
from visualize_feature_maps import Interpret
from vit_manual_attention_classifier import ViTManualAttentionClassifier

torch.manual_seed(RANDOM_SEED)
model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import ViTModel
from torch import nn
from call_backs import CallBack, SaveTheBestCallBack


def create_model_vit(main_directory = None):
    model = ViTClassifier(model_id)
    return model

def create_model_vit_manual_attention(main_directory = None):
    #model = ViTClassifier(model_id)
    model = ViTManualAttentionClassifier(model_id,num_labels=2, attention_image_path=main_directory)
    # model = your custom model
    # model.train()
    return model


def experiment(model, train_data_set, added_metrics=[], validation_data_set=None, callback_list=[]):
    num_training_steps = len(train_data_set) * EPOCHS  # // BATCH_SIZE
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model = fit(model, optimizer, lr_scheduler=lr_scheduler,
                train_data_set=train_data_set, validation_data_set=validation_data_set,
                num_epochs=EPOCHS, added_metrics=added_metrics, epoch_call_backs=callback_list)
    return model


def prepare_data(base_class_loader, data_folder):
    train_data_set = base_class_loader.get_data_loader(path=data_folder, folder='train', batch_size=BATCH_SIZE,
                                                       shuffle=True)
    validation_data_set = base_class_loader.get_data_loader(path=data_folder, folder='validation',
                                                            batch_size=BATCH_SIZE, shuffle=False)
    test_data_set = base_class_loader.get_data_loader(path=data_folder, folder='test',
                                                      batch_size=BATCH_SIZE, shuffle=False)
    return train_data_set, validation_data_set, test_data_set


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device ====================', device)

    main_directory = RGB_DIRECTORY_PATH
    #model = create_model_vit(main_directory=main_directory)
    model = create_model_vit_manual_attention(main_directory=main_directory)
    save_the_best_callback = SaveTheBestCallBack(model=model, target_file=checkpoint_filepath, less_is_better=False)

    base_class_loader = BaseClassLoader(feature_extractor)
    train_data, val_data, test_data = prepare_data(base_class_loader, main_directory)
    model = experiment(model, train_data_set=train_data, validation_data_set=val_data,
                       added_metrics=['auroc', 'f1_score'], callback_list=[save_the_best_callback])
    # visualize_feature_maps(model=model, data_set=train_data, feature_extractor=feature_extractor)
    #heat_map = Interpret(model=model, feature_extractor=feature_extractor,
    #                     image_path=main_directory + TRAIN_FOLDER + '/normal/' + '1_test_0_.png')
    # heat_map.interpret()
    # visualize_explain(model=model, feature_extractor=feature_extractor,
    #                  image_path=main_directory + TRAIN_FOLDER + '/pnemothorax/' + '1_test_0_.png')
    evaluate(model=model, data_set=test_data, data_type='test', added_metrics=['auroc', 'f1_score'])
