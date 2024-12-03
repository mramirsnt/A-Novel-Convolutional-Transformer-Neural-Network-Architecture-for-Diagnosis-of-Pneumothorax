import torch
from tqdm import tqdm
from pgu_metrics import PGU_Metrics
from torch import Tensor
from call_backs import CallBack, SaveTheBestCallBack


def fit(model, optimizer, lr_scheduler, train_data_set, validation_data_set=None,
        added_metrics=[], num_epochs=1, num_classes = 2, epoch_call_backs=[]):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device ====================', device)
    #model.train()
    print('train batch size =', train_data_set.batch_size)
    train_steps = len(train_data_set) // train_data_set.batch_size

    tepoch = tqdm(None, smoothing=0, total=train_steps,
                  disable=False, leave=True, dynamic_ncols=True)

    pgmetrics = PGU_Metrics(num_classes=num_classes, threshold=0.5, positive_class=1)

    for epoch in range(num_epochs):

        tepoch.reset(train_steps)
        tepoch.initial = 0
        predicted_values = []
        true_values = []
        counter = 0
        for batch in train_data_set:
            #print(batch.items())
            counter = counter + 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            pv = outputs.logits
            #print('pv requires_grad ==================', pv.requires_grad)
            tv = batch['labels']
            true_values.extend(tv.tolist())
            predicted_values.extend(pv.tolist())

            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix({'loss': loss.item()})  # , accuracy=100. * accuracy)
            tepoch.update()

        #print('predicted_values', torch.argmax(Tensor(predicted_values), dim=-1))
        #print('true value = ', true_values)
        res = pgmetrics.calculate_metrics(added_metrics=added_metrics, data_type='train', real_value=true_values,
                                predicted_value=predicted_values)
        print(f'counter = {counter}')
        val_res = None
        if validation_data_set is not None:
            val_res = evaluate(model, added_metrics=added_metrics, data_type='validation', data_set=validation_data_set)
            current_value = val_res['val_accuracy']
            for callback in epoch_call_backs:
                if type(callback) is SaveTheBestCallBack:
                    callback.run(current_value)
        if val_res is None:
            print(res)
        else:
            print(res, val_res)
    return model




def evaluate(model, data_set, data_type='train', num_classes = 2, added_metrics=[]):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pgmetrics = PGU_Metrics(num_classes=num_classes, threshold=0.5, positive_class=1)

    predicted_values = []
    true_values = []
    with torch.no_grad():
        for batch in data_set:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            pv = outputs.logits
            tv = batch['labels']
            true_values.extend(tv.tolist())
            predicted_values.extend(pv.tolist())
        res = pgmetrics.calculate_metrics(added_metrics=added_metrics, data_type=data_type, real_value=true_values,
                            predicted_value=predicted_values)
        return res
