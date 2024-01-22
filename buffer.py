import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import (
    TensorDataset,
    ParamDiffAug,
    get_dataset,
    get_network,
    get_daparam,
    epoch,
)
import copy

def main(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    channel, im_size, num_classes, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, args=args)

    print('Hyperparameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset, args.model, str(args.lr_teacher)) # buffers/Oxford-IIIT/UNetV(1/2)/(0.01/0.001)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    masks_all = []
    indices_class = [[] for c in range(num_classes-1)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        img, (lab, msk) = dst_train[i]
        images_all.append(torch.unsqueeze(img, dim=0))
        labels_all.append(lab)
        masks_all.append(torch.unsqueeze(msk, dim=0))

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
    masks_all = torch.cat(masks_all, dim=0).to("cpu")

    for c in range(num_classes-1):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))
    
    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
    
    criterion = nn.CrossEntropyLoss().to(args.device)

    trajectories = []

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(masks_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=2, pin_memory=True)

    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'
    print("DC augmentation parameters: \n", args.dc_aug_param)

    for it in range(1, args.num_experts+1):
        ''' train synthetic data '''
        teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        teacher_net.train()
        lr = args.lr_teacher 
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2) # optimimzer_img for synthetic data
        teacher_optim.zero_grad()

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        for e in range(1, args.train_epochs+1):
            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                          criterion=criterion, args=args, aug=True)
            
            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                        criterion=criterion, args=args, aug=False)
            
            print("Itr: {}\tEpoch: {}\tTrain Loss: {:.5f}\tTest Loss: {:.5f}\tTrain DC: {:.5f}\tTest DC: {:.5f}".format(it, e, train_loss, test_loss, train_acc, test_acc))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        parameters_changed = all(
            any(torch.any(p1 != p2) for p1, p2 in zip(ts1, ts2))
            for ts1, ts2 in zip(timestamps, timestamps[1:])
        )
        print("parameters_changed in each iteration:", parameters_changed)

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument('--dataset', type=str, default='Oxford-IIIT', help='dataset')
    parser.add_argument('--seg_task', type=str, default='B', choices=['B', 'M'],
                        help='whether to use binary or multi-label segmentation')
    parser.add_argument('--model', type=str, default='UNetV1', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='num experts')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=128, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='False', choices=['True', 'False'], 
                        help='whether to use differentiable Siamese augmentation')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='oxford-iiit-pet-dataset', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regulerization')
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()
    main(args)
