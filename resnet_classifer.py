from torch import nn
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers import ResNetModel



class Resnet50(nn.Module):
    def __init__(self,model_id="microsoft/resnet-34",num_labels=2):
        super(Resnet50, self).__init__()
        self.res = ResNetModel.from_pretrained(model_id)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(self.res.config.hidden_sizes[-1],num_labels))
        self.num_labels = num_labels
    def forward(self, pixel_values, labels):
        outputs = self.res(pixel_values=pixel_values)
        logits = self.classifier(outputs[1])
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states

       )

model = Resnet50()