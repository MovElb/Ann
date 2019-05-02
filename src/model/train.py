import argparse
import logging
import os
import random
import sys
from datetime import datetime
from shutil import copyfile

import msgpack
import torch

from BatchGen import BatchGen
from model import BertyModel
from utils import str2bool, score


def setup():
    parser = argparse.ArgumentParser(
            description='Train a BertyNet model.'
    )
    # System parameters
    parser.add_argument('--log_file', default='output.log',
                        help='path for log file.')
    parser.add_argument('--log_per_updates', type=int, default=10,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--data_file', default='data.msgpack',
                        help='name of preprocessed data file.')
    parser.add_argument('--meta_file', default='meta.msgpack', help='name of meta-data file')
    parser.add_argument('--data_dir', default='SQuAD2', help='path to preprocessed data directory')
    parser.add_argument('--model_dir', default='models',
                        help='path to store saved models.')
    parser.add_argument('--save_last_only', action='store_true',
                        help='only save the final models.')
    parser.add_argument('--eval_per_epoch', type=int, default=1,
                        help='perform evaluation per x epochs.')
    parser.add_argument('--seed', type=int, default=1337,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')

    # Training parameters
    parser.add_argument('--eval', action='store_true', help='turn on evaluation-only mode')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-rs', '--resume', default='best_model.pt',
                        help='previous model file name (in `model_dir`). '
                             'e.g. "checkpoint_epoch_11.pt"')
    parser.add_argument('-ro', '--resume_options', action='store_true',
                        help='use previous model options, ignore the cli and defaults.')
    parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                        help='finetune top-k embeddings.')
    parser.add_argument('--fix_embeddings', action='store_true',
                        help='if true, `tune_partial` will be ignored.')

    # Model parameters
    parser.add_argument('--rnn_padding', action='store_true',
                        help='perform rnn padding (much slower but more accurate).')
    parser.add_argument('--rnn_hidden_size', type=int, default=125)
    parser.add_argument('--attention_hidden_size', type=int, default=250)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--num_features', type=int, default=4)
    parser.add_argument('--glove_dim', type=int, default=300)
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--pos_dim', type=int, default=12)
    parser.add_argument('--ner_dim', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--threshold_ans', type=float, default=0.6)

    args = parser.parse_args()

    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    if args.resume == 'best_model.pt' and not os.path.exists(os.path.join(args.model_dir, args.resume)):
        # means we're starting fresh
        args.resume = ''

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # setup logger
    class ProgressHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)
            self.is_overwrite = False

        def emit(self, record):
            log_entry = self.format(record)
            if record.message.startswith('> '):
                sys.stdout.write('{}\r'.format(log_entry.rstrip()))
                sys.stdout.flush()
                self.is_overwrite = True
            else:
                if self.is_overwrite:
                    sys.stdout.write('\n')
                    self.is_overwrite = False
                sys.stdout.write('{}\n'.format(log_entry))

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    ch = ProgressHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    return args, log


def load_data(opt):
    """
    Args:
        opt (dict): options converted from command line arguments
    Returns:
        tuple:
            train_data - list of all train examples. Each example is a list of pre-processed info.
            dev_data - list of all dev examples. Each example is a list of pre-processed info
                without 'answer' and 'has_answer'.
            dev_answer_info - list of ['has_answer', 'answer'] info for all dev examples.
            embeddings - list of lists. Matrix of glove embeddings computed during pre-processing step.
            opt - dict of updated options

    """
    opt['use_cuda'] = opt['cuda']
    meta_filename = os.path.join(opt['data_dir'], opt['meta_file'])
    with open(meta_filename, 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embeddings = meta['embedding']
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])

    data_filename = os.path.join(opt['data_dir'], opt['data_file'])
    with open(data_filename, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train_data = data['train']

    data['dev'].sort(key=lambda x: len(x[1]))
    dev_data = [x[:-2] for x in data['dev']]
    dev_answer_info = [x[-2:] for x in data['dev']]
    return train_data, dev_data, dev_answer_info, embeddings, opt


def main():
    args, log = setup()
    log.info('[Program starts. Loading data...]')
    train_data, dev_data, dev_answer_info, embeddings, opt = load_data(vars(args))
    log.info(opt)
    log.info('[Data loaded.]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(args.model_dir, args.resume),
                                map_location="cuda" if torch.cuda.is_available() else "cpu")
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = BertyModel(opt, embeddings, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        # synchronize random seed
        random.setstate(checkpoint['random_state'])
        torch.random.set_rng_state(checkpoint['torch_state'])
        if args.cuda:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])

    else:
        model = BertyModel(opt, embeddings)
        epoch_0 = 1

    if args.cuda:
        model.cuda()

    if args.resume:
        batches = BatchGen(dev_data, batch_size=args.batch_size, evaluation=True)
        predictions = []
        is_answerable = []
        for i, batch in enumerate(batches):
            batch_pred, batch_is_answerable = model.predict(batch)
            predictions.extend(batch_pred)
            is_answerable.extend(batch_is_answerable)
            log.debug('> evaluating [{}/{}]'.format(i, len(batches)))
        em, f1 = score(predictions, is_answerable, dev_answer_info)
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        best_val_score = f1
    else:
        best_val_score = 0.0

    if args.eval:
        return

    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(train_data, batch_size=args.batch_size)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)
            if i % args.log_per_updates == 0:
                log.info('> epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                        epoch, model.updates, model.train_loss.value,
                        str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        # eval
        if epoch % args.eval_per_epoch == 0:
            batches = BatchGen(dev_data, batch_size=args.batch_size, evaluation=True)
            predictions = []
            is_answerable = []
            for i, batch in enumerate(batches):
                batch_pred, batch_is_answerable = model.predict(batch)
                predictions.extend(batch_pred)
                is_answerable.extend(batch_is_answerable)
                log.debug('> evaluating [{}/{}]'.format(i, len(batches)))
            em, f1 = score(predictions, is_answerable, dev_answer_info)
            log.warning("dev EM: {} F1: {}".format(em, f1))
        # save
        if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            model_file = os.path.join(args.model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch)
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                        model_file,
                        os.path.join(args.model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')

if __name__ == '__main__':
    main()
