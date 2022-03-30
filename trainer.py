# -*- coding: utf-8 -*-
import os
import argparse

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging

from data_sst2 import DataPrecessForSentence
from models.models import BertFineTuneModel, BertPrefixTuneModel
from utils import train, validate


def model_train_validate_test(train_df, dev_df, test_df, target_dir,
                              max_seq_len=100,
                              epochs=3,
                              batch_size=32,
                              lr=2e-5,
                              prefix_len=50,
                              scheduler_patience=1,
                              early_stopping_patience=1,
                              max_grad_norm=10.0,
                              if_save_model=True,
                              checkpoint=None,
                              mode='finetune'):
    '''
    Parameters
    ----------
    train_df : pandas dataframe of train set.
    dev_df : pandas dataframe of dev set.
    test_df : pandas dataframe of test set.
    target_dir : the path where you want to save model.
    max_seq_len: the max truncated length.
    epochs : the default is 3.
    batch_size : the default is 32.
    lr : learning rate, the default is 2e-05.
    prefix_len: length of prefix in prefix-tuning. the default is 50.
    scheduler_patience : the default is 1.
    early_stopping_patience: the default is 1
    max_grad_norm : the default is 10.0.
    if_save_model: if save the trained model to the target dir.
    checkpoint : the default is None.
    mode: Fine-tune or prefix-tune.
    '''

    assert mode in ['finetune', 'prefixtune'], 'Invalid mode'
    if mode == 'finetune':
        model = BertFineTuneModel()
    else:
        model = BertPrefixTuneModel(prefix_len=prefix_len)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    print(20 * '=', ' Preparing for training ', 20 * '=')
    # Path to save the model, create a folder if not exist.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading --------------------------------------#

    print('\t* Loading training data...')
    train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len=max_seq_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print('\t* Loading validation data...')
    dev_data = DataPrecessForSentence(tokenizer, dev_df, max_seq_len=max_seq_len)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    print('\t* Loading test data...')
    test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len=max_seq_len)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # -------------------- Model definition ------------------- --------------#

    print('\t* Building model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # -------------------- Preparation for training  -------------------------#

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # Implement of warm up
    # total_steps = len(train_loader) * epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=60, num_training_steps=total_steps)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.85,
                                                           patience=scheduler_patience)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    valid_aucs = []

    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        print('\t* Training will continue on existing model from epoch {}...'.format(start_epoch))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochs_count = checkpoint['epochs_count']
        train_losses = checkpoint['train_losses']
        train_accuracy = checkpoint['train_accuracy']
        valid_losses = checkpoint['valid_losses']
        valid_accuracy = checkpoint['valid_accuracy']
        valid_auc = checkpoint['valid_auc']

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, auc, _, = validate(model, dev_loader)
    print('\n* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}'.format(valid_loss,
                                                                                               (valid_accuracy * 100),
                                                                                               auc))

    # -------------------- Training epochs -----------------------------------#

    print('\n', 20 * '=', 'Training bert model on device: {}'.format(device), 20 * '=')
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print('* Training epoch {}:'.format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print('-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%'.format(epoch_time, epoch_loss,
                                                                                   (epoch_accuracy * 100)))

        print('* Validation for epoch {}:'.format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc, _, = validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        valid_accuracies.append(epoch_accuracy)
        valid_aucs.append(epoch_auc)
        print('-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}'
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))

        # Update the learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            if if_save_model:
                torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_score': best_score,
                            'epochs_count': epochs_count,
                            'train_losses': train_losses,
                            'train_accuracy': train_accuracies,
                            'valid_losses': valid_losses,
                            'valid_accuracy': valid_accuracies,
                            'valid_auc': valid_aucs
                            },
                           os.path.join(target_dir, 'best.pth.tar'))
                print('save model succesfully!\n')

        if patience_counter >= early_stopping_patience:
            print('-> Early stopping: patience limit reached, stopping...')
            break

    # run model on test set and save the prediction result to csv
    best_model = torch.load(os.path.join(target_dir, 'best.pth.tar'))
    model.load_state_dict(best_model['model'])
    print('* Test for the best model from epoch {}:'.format(best_model['epoch']))
    _, _, test_accuracy, _, all_prob = validate(model, test_loader)
    print('Test accuracy: {:.4f}\n'.format(test_accuracy))
    test_prediction = pd.DataFrame({'prob_1': all_prob})
    test_prediction['prob_0'] = 1 - test_prediction['prob_1']
    test_prediction['prediction'] = test_prediction.apply(lambda x: 0 if (x['prob_0'] > x['prob_1']) else 1,
                                                          axis=1)
    test_prediction = test_prediction[['prob_0', 'prob_1', 'prediction']]
    test_prediction.to_csv(os.path.join(target_dir, 'test_prediction.csv'), index=False)


if __name__ == '__main__':
    logging.set_verbosity_error()
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_len', type=int, default=100, help='the max truncated length.')
    parser.add_argument('--epochs', type=int, default=3, help='training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate.')
    parser.add_argument('--prefix_len', type=int, default=50, help='prefix length in prefix-tuning.')
    parser.add_argument('--scheduler_patience', type=int, default=1, help='scheduler patience.')
    parser.add_argument('--early_stopping_patience', type=int, default=1, help='early stop patience.')
    parser.add_argument('--mode', type=str, default='finetune', help='mode in either finetune or prefixtune')
    args = vars(parser.parse_args())

    data_path = './data/'
    train_df = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t', header=None, names=['similarity', 's1'])
    dev_df = pd.read_csv(os.path.join(data_path, 'dev.tsv'), sep='\t', header=None, names=['similarity', 's1'])
    test_df = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t', header=None, names=['similarity', 's1'])
    target_dir = f'./output/Bert/{args["mode"]}'
    model_train_validate_test(train_df, dev_df, test_df, target_dir, **args)
