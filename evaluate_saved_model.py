from load_dataset import BaseClassLoader
from settings import checkpoint_filepath, RGB_DIRECTORY_PATH, BATCH_SIZE, MINIMAL_RGB
from pgu_train_evaluate import evaluate
from transformers import get_scheduler, ViTFeatureExtractor
import torch

from vit_classifier import ViTClassifier

model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)

main_directory = MINIMAL_RGB

base_class_loader = BaseClassLoader(feature_extractor)
# print('root data folder = ', RGB_DIRECTORY_PATH)
#  replace MINIMAL_RGB with address of your dataset folder root

test_data_set = base_class_loader.get_data_loader(path=main_directory, folder='test',
                                                  batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = ViTClassifier(model_id)

#model = TheModelClass(*args, **kwargs)
print('-------------------------------------------------before load model')
model.load_state_dict(torch.load(checkpoint_filepath))
print('-------------------------------------------------after load model')

model.to(device=device)
print('-------------------------------------------------before evaluate ')

res = evaluate(model=model,data_set=test_data_set,data_type='test' ,added_metrics=['auroc', 'f1_score', 'precision', 'specificity','confusion_matrix'])

print(res)
print('-------------------------------------------------after evaluate ')
