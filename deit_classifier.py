from transformers import  DeiTModel
from transformers.modeling_outputs import ImageClassifierOutput
from torch import nn

class DeitClassifier(nn.Module):
    def __init__(self,model_id = "facebook/deit-base-distilled-patch16-224",num_labels=2):
        super(DeitClassifier, self).__init__()
        self.deit = DeiTModel.from_pretrained(model_id)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.deit.config.hidden_size,num_labels)

    def forward(self, pixel_values, labels=None):
        outputs = self.deit(pixel_values=pixel_values)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )