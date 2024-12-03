from transformers import ConvNextModel
from transformers.modeling_outputs import  ImageClassifierOutputWithNoAttention
from torch import nn

class ConvNextClassifier(nn.Module):
    def __init__(self,model_id = "facebook/convnext-tiny-224",num_labels=2):
        super(ConvNextClassifier, self).__init__()
        self.convnext = ConvNextModel.from_pretrained(model_id)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.convnext.config.hidden_sizes[-1],num_labels)

    def forward(self,pixel_values, labels=None):
        outputs = self.convnext(pixel_values=pixel_values)
        logits = self.classifier(outputs[1])
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )