import time
import scipy.misc as m
import numpy as np
import cv2
import torch
import torchvision.utils as vutils
import argparse
from tqdm import *
from zmq import device
from model.spade_model import SpadeModel
from opt.configTrain import TrainOptions
from loader.dataset_loader_demo import DatasetLoaderDemo
from fusion.affineFace import *
import mask_attacks
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--pose_path', type=str, default='data/poseGuide/imgs', help='path to pose guide images')
parser.add_argument('--ref_path', type=str, default='data/reference/imgs', help='path to appearance/reference images')
parser.add_argument('--pose_lms', type=str, default='data/poseGuide/lms_poseGuide.out', help='path to pose guide landmark file')
parser.add_argument('--ref_lms', type=str, default='data/reference/lms_ref.out', help='path to reference landmark file')
args = parser.parse_args()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Model running on {device}")
    trainConfig = TrainOptions()
    opt = trainConfig.get_config()  # namespace of arguments
    # init test dataset
    dataset = DatasetLoaderDemo(gaze=(opt.input_nc == 9), imgSize=256)

    root = args.pose_path  # root to pose guide img
    path_Appears = args.pose_lms.format(root)  # root to pose guide dir&landmark
    dataset.loadBounds([path_Appears], head='{}/'.format(root))

    root = args.ref_path  # root to reference img
    path_Appears = args.ref_lms.format(root)   # root to reference dir&landmark
    dataset.loadAppears([path_Appears], '{}/'.format(root))
    dataset.setAppearRule('sequence')

    # dataloader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=12, drop_last=False)
    print('dataset size: {}\n'.format(dataset.shape()))

    # output sequence: ref1-pose1, ref1-pose2,  ref1-pose3, ... ref2-pose1, ref2-pose2, ref2-pose3, ...
    boundNew = []
    appNew = []
    for aa in dataset.appearList:
        for bb in dataset.boundList:
            boundNew.append(bb)
            appNew.append(aa)
    dataset.boundList = boundNew
    dataset.appearList = appNew

    model = SpadeModel(opt)  # define model
    model.setup(opt)  # initilize schedules (if isTrain), load pretrained models
    model.set_logger(opt) # set writer to runs/test_res
    model.eval()

    iter_start_time = time.time()
    cnt = 1

    # Initialize Metrics
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0

    # with torch.no_grad():
    for step, data in tqdm(enumerate(data_loader)):
        origin_img_src = data['img_src']
        origin_img_src = origin_img_src.to(device)

        #save original results as Y
        model.set_input(data)  # set device for data
        model.forward()
        original_output = model.fake_B.cpu().detach() #original reenactment result
        ori_sample_z = model.sample_z.cpu().detach() #original sample z 攻击的基准Y

        #对比原输出和原输入(shape=[batch_size,3,256,256])，计算mask
        mask = abs(origin_img_src.cpu() - original_output.cpu())
        mask = mask.cpu().detach()
        mask = mask[0,0,:,:]+mask[0,1,:,:]+mask[0,2,:,:] #mask shape=[256,256]
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0
        # print("mask 的 shape")
        # print(mask.shape)
        # print("mask 的值>=0.5的个数")
        # print(mask[mask>=0.5].shape)

        #输出mask>=0.5部分的original_output
        mask_output = original_output*mask
        original_output = mask_output

        ifgsm_attack = mask_attacks.IFGSMAttack(model=model,device=device,mask=mask)
        #攻击：传入data(包含x，即img_src)，和基准Y
        x_adv,perturb = ifgsm_attack.perturb(data,model.fake_B.clone().detach_())#fake_B作为Y
        # x_adv,perturb = ifgsm_attack.perturb(data,model.sample_z.clone().detach_())#sample_z作为Y

        
        # use the adversial img_src as input, to generate adversial result
        data['img_src'] = x_adv
        model.set_input(data)  # set device for data
        model.forward()
        adversial_output = model.fake_B.cpu().detach() #adversial attack result

        # fusionNet
        for i in range(data['img_src'].shape[0]):
            # img_gen = model.fake_B.cpu().detach().numpy()[i].transpose(1, 2, 0)
            img_gen = original_output.numpy()[i].transpose(1, 2, 0)
            
            # #攻击：传入data(包含x，即img_src)，和基准Y
            # x_adv,perturb = ifgsm_attack.perturb(data,model.fake_B.clone().detach_())#fake_B作为Y

            #——————生成未攻击结果———————
            img_gen = (img_gen * 0.5 + 0.5) * 255.0
            img_gen = img_gen.astype(np.uint8)
            img_gen = dataset.gammaTrans(img_gen, 2.0) # model output image, 256*256*3
            # cv2.imwrite('output_noFusion/{}.jpg'.format(cnt), img_gen)

            lms_gen = data['pt_dst'].cpu().numpy()[i] / 255.0 # [146, 2]
            img_ref = data['img_src_np'].cpu().numpy()[i]
            lms_ref = data['pt_src'].cpu().numpy()[i] / 255.0
            lms_ref_parts, img_ref_parts = affineface_parts(img_ref, lms_ref, lms_gen)

            # fusion
            fuse_parts, seg_ref_parts, seg_gen = fusion(img_ref_parts, lms_ref_parts, img_gen, lms_gen, 0.1)
            fuse_eye, mask_eye, img_eye = lightEye(img_ref, lms_ref, fuse_parts, lms_gen, 0.1)
            # res = np.hstack([img_ref, img_pose, img_gen, fuse_eye])
            cv2.imwrite('output/{}.jpg'.format(cnt), fuse_eye)

            #————————攻击后的结果————————
            # data['img_src'] = x_adv.cpu().detach().numpy()
            # data['img_src'] = x_adv
            # model.set_input(data)  # set device for data
            # model.forward()

            # adv_img_gen = model.fake_B.cpu().detach().numpy()[i].transpose(1, 2, 0)
            adv_img_gen = adversial_output.numpy()[i].transpose(1, 2, 0)
            adv_img_gen = (adv_img_gen * 0.5 + 0.5) * 255.0
            adv_img_gen = adv_img_gen.astype(np.uint8)
            adv_img_gen = dataset.gammaTrans(adv_img_gen, 2.0) # model output image, 256*256*3
            # cv2.imwrite('output_noFusion/{}.jpg'.format(cnt), img_gen)

            # fusion
            fuse_parts_adv, seg_ref_parts_adv, seg_gen_adv = fusion(img_ref_parts, lms_ref_parts, adv_img_gen, lms_gen, 0.1)
            fuse_eye_adv, mask_eye_adv, img_eye_adv = lightEye(img_ref, lms_ref, fuse_parts_adv, lms_gen, 0.1)
            # res = np.hstack([img_ref, img_pose, img_gen, fuse_eye])
            cv2.imwrite('adv_output/{}.jpg'.format(cnt), fuse_eye_adv)

            #save adv_src_img
            #!!!x_adv.shape = [6,3,256,256]!!!
            adv_src_img = np.float32(x_adv[i].cpu().detach()).transpose(1, 2, 0)
            adv_src_img = cv2.normalize(adv_src_img,None,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
            cv2.imwrite('adv_src/{}.jpg'.format(cnt), adv_src_img)
            
            #save origin_src_img
            ori_src_img = np.float32(origin_img_src[i].cpu().detach()).transpose(1, 2, 0)
            ori_src_img = cv2.normalize(ori_src_img,None,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
            cv2.imwrite('ori_src/{}.jpg'.format(cnt), ori_src_img)

            cnt += 1
            # Compare to ground-truth output
            # l1_error += F.l1_loss(fuse_eye_adv, fuse_eye)
            # l2_error += F.mse_loss(fuse_eye_adv, fuse_eye)
            # l0_error += (fuse_eye_adv - fuse_eye).norm(0)
            # min_dist += (fuse_eye_adv - fuse_eye).norm(float('-inf'))

            # Compare to input image
            l1_error += F.l1_loss(origin_img_src, x_adv)
            l2_error += F.mse_loss(origin_img_src, x_adv)
            l0_error += (origin_img_src - x_adv).norm(0)
            min_dist += (origin_img_src - x_adv).norm(float('-inf'))

            # if F.mse_loss(fuse_eye_adv, fuse_eye) > 0.05:
            #     n_dist += 1

            n_samples += 1


    iter_end_time = time.time()
    # Print metrics
    print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, 
    l1_error / n_samples, l2_error / n_samples, float(n_dist) / float(n_samples), l0_error / n_samples, min_dist / n_samples))


    print('length of dataset:', len(dataset))
    print('time per img: ', (iter_end_time - iter_start_time) / len(dataset))




