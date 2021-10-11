import torch
from torch import optim
from torch.utils.data import DataLoader
import encoder.AMDE_implementations
from losses import LOSS_FUNCTIONS
from train_logging import LOG
from encoder.molgraph_data import MolGraphDataset, molgraph_collate_fn
import argparse
import numpy as np


MODEL_CONSTRUCTOR_DICTS = {
         'AMDE': {
        'constructor': encoder.AMDE_implementations.Graph_encoder,
        'hyperparameters': {
            'message-passes': {'type': int, 'default': 2},
            'message-size': {'type': int, 'default': 25},
            'msg-depth': {'type': int, 'default': 2},
            'msg-hidden-dim': {'type': int, 'default': 50},
            'att-depth': {'type': int, 'default': 2},
            'att-hidden-dim': {'type': int, 'default': 50},
            'gather-width': {'type': int, 'default': 75},
            'gather-att-depth': {'type': int, 'default': 2},
            'gather-att-hidden-dim': {'type': int, 'default': 45},
            'gather-emb-depth': {'type': int, 'default': 2},
            'gather-emb-hidden-dim': {'type': int, 'default': 26},
            'out-depth': {'type': int, 'default': 2},
            'out-hidden-dim': {'type': int, 'default': 90},
            'out-layer-shrinkage': {'type': float, 'default': 0.6}
        }
    }
}



common_args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

common_args_parser.add_argument('--train-set', type=str, default='./Data/train.csv', help='Training dataset path')
common_args_parser.add_argument('--valid-set', type=str, default='./Data/valid.csv', help='Validation dataset path')
common_args_parser.add_argument('--test-set', type=str, default='./Data/test.csv', help='Testing dataset path')

#common_args_parser.add_argument('--train-set', type=str, default='./data_demo/sample_train.csv', help='Training dataset path')
#common_args_parser.add_argument('--valid-set', type=str, default='./data_demo/sample_valid.csv', help='Validation dataset path')
#common_args_parser.add_argument('--test-set', type=str, default='./data_demo/sample_test.csv', help='Testing dataset path')
common_args_parser.add_argument('--loss', type=str, default='CrossEntropy', choices=[k for k, v in LOSS_FUNCTIONS.items()])
common_args_parser.add_argument('--score', type=str, default='All', help='roc-auc or MSE or All')
common_args_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
common_args_parser.add_argument('--batch-size', type=int, default=128, help='Number of graphs in a mini-batch')
common_args_parser.add_argument('--learn-rate', type=float, default=1e-4)
common_args_parser.add_argument('--savemodel', action='store_true', default=False, help='Saves model with highest validation score')
common_args_parser.add_argument('--logging', type=str, default='less')

main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = main_parser.add_subparsers(help=', '.join([k for k, v in MODEL_CONSTRUCTOR_DICTS.items()]), dest='model')
subparsers.required = True

model_parsers = {}
for model_name, constructor_dict in MODEL_CONSTRUCTOR_DICTS.items():
    subparser = subparsers.add_parser(model_name, parents=[common_args_parser])
    for hp_name, hp_kwargs in constructor_dict['hyperparameters'].items():
        subparser.add_argument('--' + hp_name, **hp_kwargs, help=model_name + ' hyperparameter')
    model_parsers[model_name] = subparser


def main():
    global args
    args = main_parser.parse_args()
    args_dict = vars(args)
    # dictionary of hyperparameters that are specific to the chosen model
    model_hp_kwargs = {
        name.replace('-', '_'): args_dict[name.replace('-', '_')]   # argparse converts to "_" implicitly
        for name, v in MODEL_CONSTRUCTOR_DICTS[args.model]['hyperparameters'].items()
    }

    train_dataset = MolGraphDataset(args.train_set)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=molgraph_collate_fn)


    validation_dataset = MolGraphDataset(args.valid_set)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=molgraph_collate_fn)

    test_dataset = MolGraphDataset(args.test_set)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=molgraph_collate_fn)

    ((sample_adj_1, sample_nd_1, sample_ed_1),(sample_adj_2, sample_nd_2, sample_ed_2),sample_target,d1,d2,mask1,mask2) = train_dataset[0]

    net = MODEL_CONSTRUCTOR_DICTS[args.model]['constructor'](
        node_features_1=len(np.array(sample_nd_1[0])), edge_features_1=len(np.array(sample_ed_1[0, 0])),
        node_features_2=len(np.array(sample_nd_2[0])), edge_features_2=len(np.array(sample_ed_2[0, 0])),
        out_features=1,**model_hp_kwargs)

    optimizer = optim.Adam(net.parameters(), lr=args.learn_rate)
    criterion = LOSS_FUNCTIONS[args.loss]

    for epoch in range(args.epochs):
        net.train()
        for i_batch, batch in enumerate(train_dataloader):

            adj_1, nd_1, ed_1, adj_2, nd_2, ed_2,target,d1,d2,mask_1,mask_2 = batch
            optimizer.zero_grad()
            output = net(adj_1, nd_1, ed_1, adj_2, nd_2, ed_2,d1,d2,mask_1,mask_2)
            loss= criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 5.0)
            optimizer.step()
        with torch.no_grad():
            net.eval()
            LOG[args.logging](
                net,train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args)


if __name__ == '__main__':
    main()
