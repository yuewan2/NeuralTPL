import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

from utils.dataset import MainTplGenDataset, LvgpTplGenDataset, collate_fn
from models.model import TemplateGenerator

def build_iterator(args):
	if args.generalize == 'True':
		TplGenDataset = MainTplGenDataset
	else:
		TplGenDataset = LvgpTplGenDataset

	dataset_train = TplGenDataset(mode='train', data_folder=args.data_dir,
								  intermediate_folder=args.intermediate_dir,
								  allow_noise=True, subset=False)
	dataset_val = TplGenDataset(mode='val', data_folder=args.data_dir,
							    intermediate_folder=args.intermediate_dir,
								dictionary=(dataset_train.src_itos, dataset_train.tgt_itos),
								allow_noise=False)
	dataset_test = TplGenDataset(mode='test', data_folder=args.data_dir,
								 intermediate_folder=args.intermediate_dir,
								 dictionary=(dataset_train.src_itos, dataset_train.tgt_itos),
								 allow_noise=False)

	pads = dataset_train.src_stoi['<pad>'], dataset_train.tgt_stoi['<pad>']
	sep = dataset_train.src_stoi['>>']
	train_iter = DataLoader(dataset_train, batch_size=args.batch_size_trn, shuffle=True,
							collate_fn=partial(collate_fn, sep=sep, pads=pads, device=args.device))
	val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=True,
						  collate_fn=partial(collate_fn, sep=sep, pads=pads, device=args.device))
	test_iter = DataLoader(dataset_test, batch_size=args.batch_size_val, shuffle=False,
						   collate_fn=partial(collate_fn, sep=sep, pads=pads, device=args.device))
	return train_iter, val_iter, test_iter, dataset_train.src_itos, dataset_train.tgt_itos


def build_model(args, vocab_itos_src, vocab_itos_tgt):
	src_pad_idx = np.argwhere(np.array(vocab_itos_src) == '<pad>')[0][0]
	tgt_pad_idx = np.argwhere(np.array(vocab_itos_tgt) == '<pad>')[0][0]

	model = TemplateGenerator(
		num_layers = args.num_layers, d_model = args.d_model, 
		heads = args.heads, d_ff = args.d_ff, dropout = args.dropout, 
		vocab_size_src = len(vocab_itos_src), vocab_size_tgt = len(vocab_itos_tgt),
		src_pad_idx = src_pad_idx, tgt_pad_idx = tgt_pad_idx)
	return model.to(args.device)

def load_checkpoint(args):
	checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
	print('Loading checkpoint from {}'.format(checkpoint_path))
	checkpoint = torch.load(checkpoint_path)
	model = checkpoint['model']
	optimizer = checkpoint['optim']
	step = checkpoint['step']
	step += 1
	return step, optimizer, model
