# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:29:35 2019
训练网络制备双量子比特贝尔态：

我们在运行中发现第 56 个点达到了局部最优，对应验证集和测试集的保真度在 0.94。
但经作者试验证实，在不同版本的环境下运行会产生稍有不同的结果。作者猜想是因为模型未收敛，初始的随机数产生了一定的影响。
而不同版本的初始随机数是不同的。
但这无妨，我们这里采用了 1000 个训练点，每个训练点之后都会有对 320 个验证点保真度的测试。
后续研究者可以在自己环境下，运行此程序，根据找到验证集保真度最高时所对应训练集点数，
更改 environment 中训练集和验证集的定义，使得训练到上述的最佳训练点数终止训练。
再将完成训练的网络用于测试集测试即可。

本程序部分参考 张笑铭 (Xiaoming Zhang) 的已有工作，作者对其开源代码表示衷心感谢！
附：张笑铭 的 github 网址: https://github.com/93xiaoming

@author: Waikikilick
"""

from environment import Env
from Net_dql_3 import DeepQNetwork
import warnings
warnings.filterwarnings('ignore')
from time import *
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)


#--------------------------------------------------------------------------------------
#训练部分
def training():
    vf_list = [] 
    plt.ion()
    print('\ntraining...')
    for k in range(len(env.training_set)):
        vf = np.mean(validating()) #每个训练点在训练该点时，最后能达到的最大保真度。
        print(vf,k,'验证集平均保真度 & 当前的训练点：')
        vf_list.append(vf) 
        cost_x = []
        cost_y = []
        learning_num = 0
        
       
        env.rrset(k)
        RL.epsilon = 0
        
        for episode in range(ep_max):
            if episode >= ep_max-2: #最后一个回合，直接采用网络预测的最佳动作，一方面检验效果，另一方面加快训练速度。
                RL.epsilon = 1
            observation = env.reset()
            fid_max = 0
            while True: 
                action = RL.choose_action(observation)
                observation_, reward, done, fid = env.step(action)
                fid_max = max(fid_max,fid) #将最大保真度记录下来
                RL.store_transition(observation, action, reward, observation_)
                cost=RL.learn()
                cost_y.append(cost)
                cost_x.append(learning_num)
                learning_num += 1
                observation = observation_
                if done:
                    break 
        print('训练点保真度：',fid_max)#最后一回合的最大保真度
        plt.plot(cost_x,cost_y)
        plt.ioff() 
        plt.show()
    return vf_list
#--------------------------------------------------------------------------------------
#验证集部分
def validating():
    #如果在验证集中，有测试点的保真度超过 0.9 那么将该点的经验记录到记忆库里，
    #以弥补训练点长时间不能达到高保真度而带来的激励，加快训练速度。

    # print('validating...')
    
    validating_fidelity_list = np.zeros((len(env.validating_set),1)) #用来保存测试过程中，各验证点的最大保真度，用于检测训练效果。
    RL.epsilon = 1
    
    for k in range(len(env.validating_set)):
        
        env.vrrset(k)
        observation = env.reset()
        story = []
        test_fid_max = 0
        
        while True: #
            
            action = RL.choose_action(observation)
            observation_, reward, done, fid = env.step(action)
            observation = observation_
            story.append([observation, action, reward, observation_,fid])
            test_fid_max = max(test_fid_max,fid)

            if done:  
                break  
        if test_fid_max > 0.9:
            for unit in story:
                RL.store_transition(unit[0],unit[1],unit[2],unit[3])
                if unit[4]==test_fid_max:
                    break
        validating_fidelity_list[k] = test_fid_max
                        
    print('验证集最大保真度：',max( validating_fidelity_list))
    return validating_fidelity_list
#----------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
#测试部分
def testing():

    print('\ntesting...')
    
    test_fidelity_list = np.zeros((len(env.testing_set),1)) #用来保存测试过程中，各测试点的最大保真度
    RL.epsilon = 1
    
    for k in range(len(env.testing_set)):
        env.trrset(k)
        observation = env.reset()
        test_fid_max = 0
        
        while True: #
            
            action = RL.choose_action(observation)
            observation_, reward, done, fid = env.step(action)
            observation = observation_
            
            test_fid_max = max(test_fid_max,fid)

            if done:  
                break  
            
        test_fidelity_list[k] = test_fid_max
                  
    return test_fidelity_list
#----------------------------------------------------------------------------------

 
#--------------------------------------------------------------------------------------
#测试动态 J 噪声部分
def testing_noise_J():
    #记录好网络在每个训练点所执行的动作序列和相应的保真度序列。
    #将达到最大保真度时作为对该点执行的策略。截取动作序列。
    #加入噪声，查看该动作序列的表现情况。
    #一句话就是，先预测后截取，先设计再加噪声。
    
    print('\ntesting noise_J...')
    
    test_fidelity_list = np.zeros((len(env.testing_set),1)) #用来保存测试过程中，各测试点的最大保真度
    RL.epsilon = 1
    
    
    for k in range(len(env.testing_set)):
        
        env.trrset(k)
        observation = env.reset()
        action_list = []
        fid_list = []
        
        while True: #
            
            action = RL.choose_action(observation)
            action_list = np.append(action_list,action)
            observation_, reward, done, fid = env.step(action)
            fid_list = np.append(fid_list,fid)
            observation = observation_
            
            if done:  
                break  
            
        observation = env.reset()
        fid_list = list(map(float,fid_list))
        max_index = fid_list.index(max(fid_list))
        action_list = action_list[0:max_index+1]
        
        #加入噪声
        test_fid_noise = 0
        for action in action_list:
            
            observation_, reward, done, fid = env.step_noise_J(action)
            observation = observation_
            test_fid_noise = fid #选择最后一步的保真度作为本回合的保真度
            
        test_fidelity_list[k] = test_fid_noise #将最终保真度记录到矩阵中
        
    return test_fidelity_list
#----------------------------------------------------------------------------------
 

#--------------------------------------------------------------------------------------
#测试动态 h 噪声部分
def testing_noise_h():

    print('\ntesting noise_h...')
    
    test_fidelity_list = np.zeros((len(env.testing_set),1)) #用来保存测试过程中，各测试点的最大保真度
    RL.epsilon = 1
    
    
    for k in range(len(env.testing_set)):
        
        env.trrset(k)
        observation = env.reset()
        action_list = []
        fid_list = []
        
        while True: #
            
            action = RL.choose_action(observation)
            action_list = np.append(action_list,action)
            observation_, reward, done, fid = env.step(action)
            fid_list = np.append(fid_list,fid)
            observation = observation_
            
            if done:  
                break  
            
        observation = env.reset()
        fid_list = list(map(float,fid_list))
        max_index = fid_list.index(max(fid_list))
        action_list = action_list[0:max_index+1]
        
        #加入噪声
        test_fid_noise = 0
        for action in action_list:
            
            observation_, reward, done, fid = env.step_noise_h(action)
            observation = observation_
            test_fid_noise = fid #选择最后一步的保真度作为本回合的保真度
            
        test_fidelity_list[k] = test_fid_noise #将最终保真度记录到矩阵中
        
    return test_fidelity_list
#----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#测试部分（ J 静态噪声环境）
    
def testing_noise_J_s():

    print('\ntesting noise_J_s...')
    
    RL.epsilon = 1
    
    for k in range(len(env.testing_set)):
            
        env.trrset(k) #在测试完一个训练点后，将初量子态调到下一个待测试点上
        observation = env.reset()
        action_list = [] 
        fid_list = [] 
        while True:
            action = RL.choose_action(observation)
            
            action_list = np.append(action_list,action)
            observation_, reward, done, fid = env.step(action)
            fid_list = np.append(fid_list,fid)
            observation = observation_
            
            if done:
                break
            
        observation = env.reset()
        fid_list = list(map(float,fid_list))
        max_index = fid_list.index(max(fid_list))
        action_list = action_list[0:max_index+1]
        
        
        test_fid_noise = 0
        for action in action_list:
            
            observation_, reward, done, fid = env.step_noise_drift_J(action)
            observation = observation_
            test_fid_noise = fid 
            
        test_fidelity_list[k] = test_fid_noise 
    return test_fidelity_list
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#测试部分（ h 静态噪声环境）
    
def testing_noise_h_s():

    print('\ntesting noise_h_s...')
    
    RL.epsilon = 1
    
    for k in range(len(env.testing_set)):
            
        env.trrset(k) #在测试完一个训练点后，将初量子态调到下一个待测试点上
        observation = env.reset()
        action_list = [] #用来保存本回合所采取的动作，用于噪声分析
        fid_list = [] #用来保存本回合中的保真度，选择最大保真度对应的步骤作为后面噪声环境中动作的终止步骤
        while True:
            action = RL.choose_action(observation)
            
            action_list = np.append(action_list,action)
            observation_, reward, done, fid = env.step(action)
            fid_list = np.append(fid_list,fid)
            observation = observation_
            
            if done:
                break
            
        observation = env.reset()
        fid_list = list(map(float,fid_list))
        max_index = fid_list.index(max(fid_list))
        action_list = action_list[0:max_index+1]
        
        
        test_fid_noise = 0
        for action in action_list:
            
            observation_, reward, done, fid = env.step_noise_drift_h(action)
            observation = observation_
            test_fid_noise = fid 
            
        test_fidelity_list[k] = test_fid_noise 
    return test_fidelity_list
#--------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#将测试集的保真度从小到大排列出来，来展示保真度分布
def sort_fid(test_fidelity_list):
    sort_fid = []
    for i in range (test_fidelity_list.shape[0]) :
        b = test_fidelity_list[i,:]
        sort_fid  = np.append(sort_fid,b)
    sort_fid.sort()
    return sort_fid
#--------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
#主程序部分
if __name__ == "__main__":
    ep_max = 100
    noise_a = 0 #噪声的幅值
    env = Env(noise_a = noise_a
              ,training_num=1000
              ,validating_num=320
              )              
        
    RL = DeepQNetwork(env.n_actions, env.n_features
              ,learning_rate=0.000001 #0.000001
              ,reward_decay=0.9
              ,e_greedy=0.99
              ,replace_target_iter=200
              ,memory_size=100000
              ,batch_size = 320
              ,e_greedy_increment=1/((ep_max-10)*env.step_max)
              ,build_net_num=3
              ,n_l1=300
              ,n_l2=400
              ,n_l3=200
              )
    
    # begin_training = time()
    vf_list = training() #训练
    vf = np.mean(validating())
    print('最后的验证集保真度为：',vf)
    # end_training = time()
    # training_time = end_training - begin_training
    # print('traing_time =',training_time) #打印出训练用时
    
    
    # begin_testing = time()
    test_fidelity_list = testing() #测试
    # env.noise_a=0.1
    # env.noise_1=env.noise_a*env.noise_normal_1
    # env.noise_2=env.noise_a*env.noise_normal_2
    # test_fidelity_list = testing_noise_J() #测试 J 噪声的影响
    # test_fidelity_list = testing_noise_h() #测试 h 噪声的影响
    # test_fidelity_list = testing_noise_J_s() #测试 h 静态噪声的影响
    # test_fidelity_list = testing_noise_h_s() #测试 h  静态噪声的影响
    # end_testing = time()
    # testing_time = end_testing - begin_testing
    # print('\ntesting_time =',testing_time) #打印出测试用时
    
    
    print('\nmean_test_fidelity:\n',np.mean(test_fidelity_list))
    
   
    # # print(sort_fid(test_fidelity_list))  #将各测试点的保真度从小到大排列出来
    
    
    
#----------------------------------------------------------------------------------
#在测噪声数据时，将以下代码粘贴到 console 更改 env.noise_a 的值就可以得到在对应噪声环境下测试保真度
#而不必再重新训练网络，可以节省大量的时间

# env.noise_a = 0.09
# print('\nnoise_a =',env.noise_a)
# env.noise_1 = env.noise_a * env.noise_normal_1
# env.noise_2 = env.noise_a * env.noise_normal_2
# test_fidelity_list = testing_noise_J() #测试
# print('\nmean_test_fidelity:\n',np.mean(test_fidelity_list))

# #直接采用循环，测试多个点
# for i in [-0.02, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015]:
#     env.noise_a = i
#     print('\nnoise_a_h =',env.noise_a)
#     env.noise_1 = env.noise_a * env.noise_normal_1
#     env.noise_2 = env.noise_a * env.noise_normal_2
#     test_fidelity_list = testing_noise_J() #测试
#     print('\nmean_test_fidelity:\n',np.mean(test_fidelity_list))
# for i in [-0.02, -0.015, -0.01, -0.005, 0.005, 0.015]:
#     env.noise_a = i
#     print('\nnoise_a_J =',env.noise_a)
#     env.noise_1 = env.noise_a * env.noise_normal_1
#     env.noise_2 = env.noise_a * env.noise_normal_2
#     test_fidelity_list = testing_noise_J() #测试
#     print('\nmean_test_fidelity:\n',np.mean(test_fidelity_list))

#----------------------------------------------------------------------------------

    
    
    
    
