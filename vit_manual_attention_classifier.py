import torch
from transformers import ViTModel, ViTForImageClassification, ViTFeatureExtractor
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.vit.modeling_vit import ViTEncoder, ViTLayer, ViTSelfAttention, ViTAttention

from load_dataset import BaseClassLoader
from settings import BATCH_SIZE

model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)


class ViTManualAttentionClassifier(nn.Module):

    def __init__(self, model_id="google/vit-base-patch16-224-in21k", num_labels=2, attention_image_path=None):

        super(ViTManualAttentionClassifier, self).__init__()
        self.vit = ViTModel.from_pretrained(model_id)
        #self.fixed_vit = ViTModel.from_pretrained(model_id)
        self.multi_head_cross_attention = nn.MultiheadAttention(embed_dim=self.vit.config.hidden_size,
                                                                num_heads=1, batch_first=True)
        print('emdeb_dim ==', self.vit.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit.config.hidden_size),
            nn.Linear(self.vit.config.hidden_size, num_labels)
        )
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if attention_image_path is not None:
            self.fixed_image_for_attention = self.load_pixel_values(attention_image_path)
        else:
            print('--------------error ----- no fixed image for attention is presented')

    def load_pixel_values(self, attention_image_path):
        base_class_loader = BaseClassLoader(feature_extractor)
        test_data_set = base_class_loader.get_data_loader(path=attention_image_path, folder='attention',
                                                          batch_size=BATCH_SIZE, shuffle=False)
        batch = next(iter(test_data_set))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch['pixel_values'].requires_grad = False
        # print('batch[pixel_values] = ', batch['pixel_values'].requires_grad)
        return batch['pixel_values']

    def forward(self, pixel_values, labels=None):

        # fixed_image_outputs = self.vit(pixel_values=self.fixed_image_for_attention)
        outputs = self.vit(pixel_values=pixel_values)
        # print(outputs)
        fixed_image_outputs = self.vit(pixel_values=self.fixed_image_for_attention)
        #fixed_image_outputs = self.fixed_vit(pixel_values=self.fixed_image_for_attention)
        # print('outputs.last_hidden_state.shape === ', outputs.last_hidden_state.shape)
        # print('outputs.pooler_output.shape === ', outputs.pooler_output.shape)
        # print('fixed_image_outputs--- requires_grad = ', fixed_image_outputs[1].requires_grad)
        # print('outputs --- requires_grad = ', outputs[1].requires_grad)

        # define new cross attention soft (q*k)q
        # use layer normalization on both outputs
        # check order of parameters
        # print(f'outputs shape = ', outputs[1].shape)
        # print(f'fixed_image_outputs shape = ', fixed_image_outputs[1].shape)
        # print(outputs [1].shape)

        #print('shape outputs.last_hidden_state= ', outputs.last_hidden_state.shape[0])
        b_size = outputs.last_hidden_state.shape[0]
        #print('fixed_image_outputs.last_hidden_state.shape = ', fixed_image_outputs.last_hidden_state.shape)
        #print('fixed_image_outputs.last_hidden_state[:,0:b_size,:].shape = ', fixed_image_outputs.last_hidden_state[0:b_size,:,:].shape)
        outputs_att, attention_weight = self.multi_head_cross_attention(query=outputs.last_hidden_state,
                                                                        key=fixed_image_outputs.last_hidden_state[0:b_size,:,:],
                                                                        value=outputs.last_hidden_state)
        outputs_att = outputs_att.mean(dim=1)
        #print('outputs_att.shape = ', outputs_att.shape)
        # print('----outputs outputs-----',outputs)
        # print('outputs_att --- requires_grad = ', outputs_att.requires_grad)
        logits = self.classifier(outputs_att)
        #print('----logits-----',logits.shape)

        loss = None
        # print('attentions =', outputs.attentions)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # print('loss == ',loss)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
