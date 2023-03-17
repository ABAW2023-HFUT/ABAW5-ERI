import time
import os
import numpy as np
import torch
import torch.optim as optim
from config import REACTION_LABELS
from eval import evaluate

def train(model, train_loader, optimizer, loss_fn, use_gpu=False):

    report_loss, report_size = 0, 0
    total_loss, total_size = 0, 0

    model.train()
    if use_gpu:
        model.cuda()

    for batch, batch_data in enumerate(train_loader, 1):
        
        features, feature_lens, labels, metas = batch_data
        batch_size = features[0].size(0)

        if use_gpu:
            features = [feature.cuda() for feature in features]
            feature_lens = feature_lens.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        preds = model(features)

        loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1), feature_lens)

        loss.backward()
        optimizer.step()

        report_loss += loss.item() * batch_size
        report_size += batch_size

        total_loss += report_loss
        total_size += report_size
        report_loss, report_size, start_time = 0, 0, time.time()

    train_loss = total_loss / total_size
    return train_loss


def save_model(model, model_folder, current_seed):
    model_file_name = f'model_{current_seed}.pth'
    model_file = os.path.join(model_folder, model_file_name)
    torch.save(model, model_file)
    return model_file

learning_rate = 1e-4

def scheduler(epoch):
    if epoch < 40:
        return 1
    else:
        a = 0.5 ** (((epoch - 40) // 5)+1)
        if a * learning_rate < 1e-5:
            return learning_rate / 1e-5
        else:
            if epoch % 5 == 0:
                print('Epoch {:05d}: reducing learning rate of group 0 to {}.'.format(epoch, a*learning_rate))
            return a


def train_model(task, model, data_loader, epochs, lr, model_path, current_seed, use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, reduce_lr_patience, combined, regularization=0.0):
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)
    learning_rate = lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=reduce_lr_patience,
    #                                                     factor=0.5, min_lr=1e-5, verbose=True)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=scheduler)
    best_val_loss = float('inf')
    best_val_details = None
    best_val_score = -1
    best_model_file = ''
    early_stop = 0

    for epoch in range(1, epochs + 1):
        print(f'Training for Epoch {epoch}...')
        train_loss = train(model, train_loader, optimizer, loss_fn, use_gpu)
        val_loss, val_score, val_details = evaluate(task, model, val_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu)
        _, train_score, train_details = evaluate(task, model, train_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu)
        print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f} | [{eval_metric_str}]: {train_score:>7.4f}')
        # for i in range(len(train_details)):
        #     print(f'Epoch:{epoch:>3} / {epochs} | [Val] | [{REACTION_LABELS[i]}]: {train_details[i]:>7.4f}')
        print(f'Epoch:{epoch:>3} / {epochs} | [Val] | Loss: {val_loss:>.4f} | [{eval_metric_str}]: {val_score:>7.4f}')
        # for i in range(len(val_details)):
        #     print(f'Epoch:{epoch:>3} / {epochs} | [Val] | [{REACTION_LABELS[i]}]: {val_details[i]:>7.4f}')

        print('-' * 50)
        if combined:
            if train_score < best_val_score:
                early_stop = 0
                best_val_score = train_score
                best_val_details = train_details
                best_val_loss = train_loss
                best_model_file = save_model(model, model_path, current_seed)

            else:
                early_stop += 1
                if early_stop >= early_stopping_patience:
                    print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                        f'early stop the training process!')
                    print('-' * 50)
                    break
            # lr_scheduler.step(1 - np.mean(train_score))
        else:
            if val_score > best_val_score:
                early_stop = 0
                best_val_score = val_score
                best_val_details = val_details
                best_val_loss = val_loss
                best_model_file = save_model(model, model_path, current_seed)

            else:
                early_stop += 1
                if early_stop >= early_stopping_patience:
                    print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                        f'early stop the training process!')
                    print('-' * 50)
                    break

            # lr_scheduler.step(1 - np.mean(val_score))
        lr_scheduler.step()
    if combined:
        print(f'Seed {current_seed} | '
          f'Best [Train {eval_metric_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')
        for i in range(len(best_val_details)):
            print(f'[{REACTION_LABELS[i]}]: {best_val_details[i]:>7.4f}')
        return best_val_loss, best_val_score, best_model_file
    else:
        print(f'Seed {current_seed} | '
            f'Best [Val {eval_metric_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')
        for i in range(len(best_val_details)):
            print(f'[{REACTION_LABELS[i]}]: {best_val_details[i]:>7.4f}')
        return best_val_loss, best_val_score, best_model_file
