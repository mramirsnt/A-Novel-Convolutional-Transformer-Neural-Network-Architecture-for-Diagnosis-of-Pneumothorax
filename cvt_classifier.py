from transformers import  CvtModel
from transformers.modeling_outputs import  ImageClassifierOutputWithNoAttention
from torch import nn

class CvtClassifier(nn.Module):
    def __init__(self, model_id="microsoft/cvt-13", num_labels=2):
        super(CvtClassifier, self).__init__()
        self.cvt = CvtModel.from_pretrained(model_id)
        self.layernorm = nn.LayerNorm(self.cvt.config.embed_dim[-1])
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.cvt.config.embed_dim[-1],num_labels)

    def forward(self, pixel_values, labels=None):
        outputs = self.cvt(pixel_values=pixel_values)
        sequence_output = outputs[0]
        cls_token = outputs[1]
        if self.cvt.config.cls_token[-1]:
            sequence_output = self.layernorm(cls_token)
        else:
            batch_size, num_channels, height, width = sequence_output.shape
            # rearrange "b c h w -> b (h w) c"
            sequence_output = sequence_output.view(batch_size, num_channels, height * width).permute(0, 2, 1)
            sequence_output = self.layernorm(sequence_output)
        sequence_output_mean = sequence_output.mean(dim=1)
        logits = self.classifier(sequence_output_mean)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            )