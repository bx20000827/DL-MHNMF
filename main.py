import numpy as np
import networkx as nx
from loss_function import Loss_fuc
from Update import Updating_Uno, Updating_Uo, Updating_Vo, Updating_V_
from normalized_adj import nor
import math
import time
 
import pur_score
import nmi_score
import texas
import cornell
import washington
import wisconsin
import gene 
import reality
import citeseer
import DD244
import ENZYMES8
import citeseer
import terroristrel
import INTERNET
import cora
# import BZR
import BA


if __name__ == "__main__":

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)
    
    ########### 真实数据集：
    
    # A = np.array(nx.adjacency_matrix(G).todense())
    # adjacent_matrix_initial,ground_truth_node_membership_list = texas.get_A()
    # adjacent_matrix_initial,ground_truth_node_membership_list = cornell.get_A()
    # adjacent_matrix_initial,ground_truth_node_membership_list = washington.get_A()
    # adjacent_matrix_initial,ground_truth_node_membership_list = wisconsin.get_A()
    # adjacent_matrix_initial,ground_truth_node_membership_list = gene.get_A()
    # adjacent_matrix_initial,ground_community_labels = reality.get_A()
    # adjacent_matrix_initial,ground_truth_node_membership_list = citeseer.get_A()
    # adjacent_matrix_initial,ground_community_labels = BZR.get_A()
    # adjacent_matrix_initial,ground_truth_node_membership_list = DD244.get_A()
    # adjacent_matrix_initial,ground_truth_node_membership_list = ENZYMES8.get_A()
    # adjacent_matrix_initial,ground_truth_node_membership_list = terroristrel.get_A()
    # adjacent_matrix_initial = bio_yeast.get_A()
    # adjacent_matrix_initial,ground_truth_node_membership_list = INTERNET.get_A()
    adjacent_matrix_initial,ground_truth_node_membership_list = cora.get_A()
    # adjacent_matrix_initial,ground_truth_node_membership_list = BA.get_A()



    # k = 5 # Texas
    # k = 10 # Cornell
    # k = 5 # Washington
    # k = 10 # Wisconsin
    # k = 2  # Gene
    # k = 6 # Citeseer
     # k = 2 # Reality-call
    # k = 10 3 BRZ
    # k = 20
    # k = 2  # ENZYMES8
    # k = 2 # TerroristRel
    # k = 3 # INTERNET
    k = 7 # cora
    ######### 真实数据集处理 #################################：
    G = nx.Graph(adjacent_matrix_initial)

    size = 0
    max_component = []
    delete_component = []
    for component in nx.connected_components(G):
        if len(component) == 1:
            del_component = list(component)
            delete_component.append(del_component[0])
            # print(component)
    
    delete_component.sort(reverse=True)
    for i in range(len(delete_component)):
        G.remove_node(delete_component[i])
        ground_truth_node_membership_list.pop(delete_component[i])
    
    adjacent_matrix = np.array(nx.adjacency_matrix(G).todense())
    
    G = nx.Graph(adjacent_matrix)
    n = len(G.nodes())
    ####################################################################################


    View_num = 6
    
    temp_I = np.ones(n)
    temp_mat = np.eye(n)
    
    alpha = 0.001
    
    I = np.asmatrix(temp_I).T
    e = np.asmatrix(temp_mat)
    M = I * I.T / n
    H = e - M
    
    temp_Fscore_list = []
    temp_NMI_list = []
    temp_acc_list = []
    temp_ARI_list = [] 
    temp_pur_list = []
    loss_list = []
    
    F_std_sum = 0
    NMI_std_sum = 0
    acc_std_sum = 0
    ARI_std_sum = 0
    pur_std_sum = 0
    
    # 对A进行正则化
    # start_time = time.time()
    A = []
    for i in range(View_num):
        temp_mat = np.dot(temp_mat, adjacent_matrix)
        normalized_adj = nor(temp_mat, temp_I)
        A.append(normalized_adj)
    
    for times in range(10):
            
        temp_F = 0
        temp_A = 0
        temp_N = 0
        temp_ACC = 0
        temp_pur = 0
        
        # obj = []
        # lam是正则项，theta是控制divergence 的
        for theta in [0.001, 0.01, 0.1, 1, 10]:
             for lam in [0.001, 0.01, 0.1, 1, 10]:      
                
                Uno = []
                Uo = [] 
                Vo = []
            
                V_ = np.matrix(np.random.rand(k, n))
                last_V_ = V_
                
                for i in range(View_num):
                    
                    Uno.append(np.matrix(np.random.rand(n, k)))
                    Uo.append(np.matrix(np.random.rand(n, k)))
                    Vo.append(np.matrix(np.random.rand(k, n)))
                    
                maxiter = 300
                minError = 0.0001
                
                for iter in range(maxiter):

                    # print('-----------------------------')
                    # print('开始第',iter,'次迭代')
                    
                    for View_index in range(View_num): 
                        Uno = Updating_Uno(A, Uno, Uo, V_, Vo, View_index)
                        Uo = Updating_Uo(A, Uno, Uo,  Vo, V_, View_index)
                        Vo = Updating_Vo(A, Uno, Uo, V_, Vo, theta, View_index, View_num, n, M) 
                    V_ = Updating_V_(A, Uno, Uo, Vo, V_, lam, alpha, View_index, View_num, k, n)
                    

                    if  np.max(np.abs(V_ - last_V_)) <= minError:
                        # loss_list.append(temp_loss)
                        break
                            
                    last_V_ = V_

                # # ################## 非重叠社区检测 #####################################################
                # 2.真实数据集
                U_list = V_.T.tolist()
                detected_node_menbership_list = []
                for i in range(len(U_list)):
                    detected_node_menbership_list.append(U_list[i].index(max(U_list[i])))
            
         
                NMI = nmi_score.NMI(np.asarray(ground_truth_node_membership_list),  np.asarray(detected_node_menbership_list))
                if (temp_N < NMI):
                    temp_N = NMI

                    
                pur = pur_score.Purity(np.asarray(ground_truth_node_membership_list),  np.asarray(detected_node_menbership_list))
                if (temp_pur < pur):
                    temp_pur = pur
        
        temp_NMI_list.append(temp_N)
        temp_pur_list.append(temp_pur)
         
    NMI_mean = np.mean(temp_NMI_list)
    pur_mean = np.mean(temp_pur_list)
       
    for i in range(10):
        NMI_std_sum += (NMI_mean - temp_NMI_list[i])**2
        pur_std_sum += (pur_mean - temp_pur_list[i])**2
        

    print("NMI : %f±%f" %(NMI_mean , math.sqrt(NMI_std_sum/10)))
    print("pur : %f±%f" %(pur_mean , math.sqrt(pur_std_sum/10)))      