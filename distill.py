import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import wandb

from utils import (
    get_time,
    get_dataset,
    get_network,
    DiffAugment,
    ParamDiffAug,
    evaluate_synset,
)

from reparam_module import ReparamModule

def main(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration+1, args.eval_it).tolist()
    channel, im_size, num_classes, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, args=args)
    mean_tensor, std_tensor = torch.tensor(mean), torch.tensor(std)

    if args.dsa: 
        args.dc_aug_param = None
    
    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param

    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation", 
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    print('Hyper-parameters: \n', args.__dict__)

    images_all = []
    labels_all = []
    masks_all = []
    indices_class = [[] for c in range(num_classes-1)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        img, (lab, msk) = dst_train[i]
        images_all.append(torch.unsqueeze(img, dim=0))
        masks_all.append(torch.unsqueeze(msk, dim=0))
        labels_all.append(lab)

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    masks_all = torch.cat(masks_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes-1):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    def get_images_masks(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle], masks_all[idx_shuffle]

    image_syn = torch.randn(size=((num_classes-1)*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
    mask_syn = torch.rand(size=((num_classes-1)*args.ipc, num_classes, im_size[0], im_size[1]), dtype=torch.float, requires_grad=False, device=args.device)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.init == "real":
        print("initialize synthetic data from random real images")
        for c in range(num_classes-1):
            images, masks = get_images_masks(c, args.ipc)
            image_syn.data[c*args.ipc:(c+1)*args.ipc] = images.detach().data
            mask_syn.data[c*args.ipc:(c+1)*args.ipc] = masks.detach().data
    else:
        print("initialize synthetic data from random noise")

    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)    
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)

    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], args.lr_lr, momentum=0.5)

    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)

    expert_dir = os.path.join(args.buffer_path, args.dataset, args.model, str(args.lr_teacher))
    print("Expert Dir: {}".format(expert_dir))

    buffer = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    
    print("Num Experts: {}".format(len(buffer)))

    best_acc = {m : 0 for m in [args.model]}

    best_std = {m : 0 for m in [args.model]}

    for it in range(0, args.Iteration+1):
        save_this_it = False

        wandb.log({"Progress": it}, step=it)

        if it in eval_it_pool:
            accs_test = []
            accs_train = []
            for it_eval in range(args.num_eval):
                net_eval = get_network(args.model, channel, num_classes, im_size).to(args.device)

                mask_save = mask_syn
                with torch.no_grad():
                    image_save = image_syn
                image_syn_eval, mask_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(mask_save.detach())

                args.lr_net = syn_lr.item()
                _, acc_train, acc_test, loss_train, loss_test = evaluate_synset(it_eval, net_eval, image_syn_eval, mask_syn_eval, testloader, args, return_loss=True)
                accs_test.append(acc_test)
                accs_train.append(acc_train)
            accs_test = np.array(accs_test)
            accs_train = np.array(accs_train)
            acc_test_mean = np.mean(accs_test)
            acc_test_std = np.std(accs_test)
            if acc_test_mean > best_acc[args.model]:
                best_acc[args.model] = acc_test_mean
                best_std[args.model] = acc_test_std
                save_this_it = True
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), args.model, acc_test_mean, acc_test_std))
            wandb.log({'Accuracy/{}'.format(args.model) : acc_test_mean}, step=it)
            wandb.log({'Max_Accuracy/{}'.format(args.model): best_acc[args.model]}, step=it)
            wandb.log({'Std/{}'.format(args.model): acc_test_std}, step=it)
            wandb.log({'Max_Std/{}'.format(args.model): best_std[args.model]}, step=it)

        if it in eval_it_pool and(save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cuda()
                mask_save = mask_syn.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(mask_syn.cpu(), os.path.join(save_dir, "masks_{}.pt".format(it)))
                
                if save_this_it: 
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))
                    torch.save(mask_syn.cpu(), os.path.join(save_dir, "masks_best.pt"))
                    
                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

        wandb.log({"Synthetic LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        
        student_net = ReparamModule(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in student_net.parameters()])

        expert_trajectory = buffer[np.random.randint(0, len(buffer))]

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn
        syn_masks = mask_syn 

        indices_chunks = []
        for epoch in range(args.syn_steps):
            if not indices_chunks:
                indices = torch.randperm(len(image_syn))
                indices_chunks = list(torch.split(indices, args.batch_syn))
            
            indices_chunk = indices_chunks.pop()

            images = syn_images[indices_chunk]
            target = syn_masks[indices_chunk]

            if args.dsa:
                images, target = DiffAugment(images, target, args.dsa_strategy, param=args.dsa_param)

            output = student_net(images, flat_param=student_params[-1])
            loss = criterion(output, target)

            grad = torch.autograd.grad(loss, student_params[-1], create_graph=True)[0]
            
            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += nn.functional.mse_loss(student_params[-1], target_params, reduction='sum')
        param_dist += nn.functional.mse_loss(starting_params, target_params, reduction='sum')

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_img.zero_grad()
        #optimizer_msk.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer_img.step()
        #optimizer_msk.step() 
        optimizer_lr.step()

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        del student_params

        if it%10==0:
            print('%s iter = %04d, loss = %.4f, syn lr = %.4f'%(get_time(), it, grand_loss.item(), syn_lr.item()))

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    
    parser.add_argument('--dataset', type=str, default="Oxford-IIIT", help='dataset')
    
    parser.add_argument('--seg_task', type=str, default='B', choices=['B', 'M'], 
                        help='whether to use binary or multi-label segmentation')
    
    parser.add_argument('--model', type=str, default='UNetV1', help='model')
    
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--num_eval', type=int, default=10, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=500, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-7, help='learning rate for udpating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch_size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--init', type=str, default='real', choices=['noise', 'real'], 
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images')
    
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'], 
                        help='whether to use differentiable Siamese augmentation')
    
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', 
                        help='differentiable Siamese augmentation strategy')
    
    parser.add_argument('--data_path', type=str, default='oxford-iiit-pet-dataset', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we start at')

    args, _ = parser.parse_known_args()
    main(args)