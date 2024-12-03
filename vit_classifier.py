from transformers import ViTModel, ViTForImageClassification
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.vit.modeling_vit import ViTEncoder, ViTLayer, ViTSelfAttention, ViTAttention


class ViTClassifier(nn.Module):

    def __init__(self, model_id="google/vit-base-patch16-224-in21k", num_labels=2):

        super(ViTClassifier, self).__init__()
        self.vit = ViTModel.from_pretrained(model_id)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        #add new layers
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        #print('---------',outputs.last_hidden_state)
        logits = self.classifier(outputs[1])
        loss = None
        #print('attentions =', outputs.attentions)
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
          #print('loss == ',loss)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )