from transformers import SwinModel
from torch import nn
from transformers.file_utils import ModelOutput
import collections.abc
import math

class SwinImageClassifierOutput(ModelOutput):
    def to_2tuple(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return (x, x)

    def window_partition(input_feature, window_size):
        """
        Partitions the given input into windows.
        """
        batch_size, height, width, num_channels = input_feature.shape
        input_feature = input_feature.view(
            batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
        )
        windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
        return windows

    def window_reverse(windows, window_size, height, width):
        """
        Merges windows to produce higher resolution features.
        """
        batch_size = math.floor(windows.shape[0] / (height * width / window_size / window_size))
        windows = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)
        return windows

    def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
        """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        """
        if drop_prob == 0.0 or not training:
            return input
        keep_prob = 1 - drop_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = input.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return input * random_tensor

class SwinClassifier(nn.Module):
    def __init__(self, model_id="microsoft/swin-tiny-patch4-window7-224", num_labels=2):
        super(SwinClassifier, self).__init__()
        self.swin = SwinModel.from_pretrained(model_id)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.swin.num_features,num_labels)

    def forward(self, pixel_values, labels=None):
        outputs = self.swin(pixel_values=pixel_values)
        logits = self.classifier(outputs[1])
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SwinImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
            )