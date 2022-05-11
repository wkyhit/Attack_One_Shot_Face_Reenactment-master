import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

class IFGSMAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # PGD or I-FGSM?
        self.rand = True
    
    def perturb(self, X_nat, y):
        """
        Vanilla Attack.
        """
        origin_img_src = X_nat['img_src'].clone().detach_()#保留原始的img_src

        if self.rand:
            X_nat['img_src'] = X_nat['img_src'].to(self.device)
            X_nat['img_src'] += torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat['img_src'].shape).astype('float32')).to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()    
        
        X_nat['img_src'].to('cpu')
        
        self.model.set_input(X_nat)#设置model的数据

        for i in range(self.k):
            self.model.img_src.requires_grad = True #对img_src求梯度
            self.model.forward()
            output = self.model.fake_B
            # output.requires_grad = True

            # self.model.zero_grad()

            # Attention attack
            # loss = self.loss_fn(output_att, y)

            # Output attack
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)
            loss.requires_grad_(True) #!!解决无grad bug
            loss.backward()
            grad = self.model.img_src.grad

            # img_src_adv = self.model.img_src + self.a * grad.sign()
            img_src_adv = self.model.img_src + self.a * torch.sign(grad)


            eta = torch.clamp(img_src_adv - origin_img_src, min=-self.epsilon, max=self.epsilon)#加入的噪声
            X = torch.clamp(origin_img_src + eta, min=-1, max=1).detach_()#攻击后的img_src结果

            # Debug
            # X_adv, loss, grad, output_att, output_img = None, None, None, None, None
        #返回攻击后的img_src和noise
        return X, eta 
