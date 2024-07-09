from __future__ import print_function

import os
import random

import numpy as np
import torch
from tqdm import tqdm

import model
import options
import wsad_dataset
from test import test
from train import train

torch.set_default_dtype(torch.float32)


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import torch.optim as optim

if __name__ == '__main__':
    args = options.parser.parse_args()

    seed = args.seed
    print('=============seed: {}, pid: {}============='.format(seed, os.getpid()))
    setup_seed(seed)
    device = torch.device("cuda")
    dataset = getattr(wsad_dataset, args.dataset)(args)
    if 'Thumos' in args.dataset_name:
        max_map = [0] * 9
    else:
        max_map = [0] * 10
    max_uct_rank_acc = 0
    ckpt_folder_path = args.path_dataset + '/aaai23osr/ckpt/' + args.group_name
    if not os.path.exists(ckpt_folder_path):
        os.makedirs(ckpt_folder_path)
    print(args)
    model = getattr(model, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)

    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_loss = 0
    lrs = [args.lr, args.lr / 5, args.lr / 5 / 5]
    print(model)
    for itr in tqdm(range(args.max_iter)):

        loss = train(itr, dataset, args, model, optimizer, device)
        total_loss += loss
        if itr % args.interval == 0 and not itr == 0:
            print('Iteration: %d, Loss: %.5f' % (itr, total_loss / args.interval))
            total_loss = 0
            # torch.save(model.state_dict(), ckpt_folder_path + '/last_' + args.model_name + '.pkl')
            torch.save(model.state_dict(), ckpt_folder_path + '/last_' + args.model_name + '.pkl')

            iou, dmap, uct_rank_acc = test(itr, dataset, args, model, device)

            if 'Thumos' in args.dataset_name:
                map_update_cond = sum(dmap[:7]) > sum(max_map[:7])
            else:
                map_update_cond = np.mean(dmap) > np.mean(max_map)
            uct_update_cond = uct_rank_acc > max_uct_rank_acc

            if args.main_evaluate_indicator == 'map':
                ckpt_save_cond = map_update_cond
            elif args.main_evaluate_indicator == 'uct':
                ckpt_save_cond = uct_update_cond
            else:
                raise "Unknown indicator!"

            if ckpt_save_cond:
                torch.save(model.state_dict(), ckpt_folder_path + '/best_' + args.model_name + '.pkl')
            if map_update_cond:
                max_map = dmap
            if uct_update_cond:
                max_uct_rank_acc = uct_rank_acc

            print(f'MAX uct_rank_acc: {max_uct_rank_acc:.3f}')

            print('----------------------------------------------------------------')
            print('For all classes (MAX):')
            print('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i], max_map[i]) for i in range(len(iou))]))
            max_map = np.array(max_map)
            if 'Thumos' in args.dataset_name:
                print('Max mAP Avg 0.1-0.5: {:.3f}, Max mAP Avg 0.1-0.7: {:.3f}, Max mAP Avg 0.1-0.9: {:.3f}'
                      .format(np.mean(max_map[:5]), np.mean(max_map[:7]), np.mean(max_map)))
            print("------------------pid: {}--------------------".format(os.getpid()))
