import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import os
import pickle
from scipy.spatial.distance import pdist 
import seaborn as sns

def get_curve(feature_map,ground_truth,num_min,num_max,period):
    prec_recall_curve = []
    num = len(feature_map)
    for threshold in np.arange(num_min,num_max,period):
        all_postives = 0
        for i in np.arange(1,num):
            for j in np.arange(0,i):
                if feature_map[i,j]>= threshold:
                    if  (i-100>=j):
                        all_postives= all_postives+1
        true_positives = (feature_map >= threshold)& (ground_truth == 1)

        try:
            precision = float(np.sum(true_positives))/all_postives
            recall = float(np.sum(true_positives))/np.sum(ground_truth == 1)
            prec_recall_curve.append([threshold,precision,recall])
        except:
            break
    prec_recall_curve = np.array(prec_recall_curve)
    #print(prec_recall_curve)
    return prec_recall_curve

def main(score_files):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    assert(len(score_files) <= 8)

    print("loading KITTI00_GroundTruth")
    KITTI00_GroundTruth = np.loadtxt("./models/val_reloc/GT/kitti00GroundTruth.txt",delimiter = ",")
    print("loaded KITTI00_GroundTruth")

    for index,file_name in enumerate(score_files):

        output = open(file_name, 'rb')
        KITTI00_feats_score_00= pickle.load(output)
        output.close()

        print("doing {}".format(file_name))
        prec_recall_curve0 = get_curve(KITTI00_feats_score_00,KITTI00_GroundTruth,0.0,0.8,0.05)
        print("done {}".format(file_name))

        # prec_recall_curves.append(prec_recall_curve0)

        plt.plot(prec_recall_curve0[:,2],prec_recall_curve0[:,1],colors[index],label=(file_name.split("/")[-1]).split(".")[0] )
    #题目
    plt.title('Precision-Recall curve')
    #坐标轴名字
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #背景虚线
    plt.grid(True)
    plt.legend(loc='upper right')
    #显示
    #plt.show()
    plt.savefig("./output_dir/val_reloc.png")
    print("done")

if __name__ == '__main__':
    score_files = []
    #score_files.append('./models/val_reloc/GT/feats_score_00.pkl')
    #score_files.append('./output_dir/val_reloc/feats_score_00_float_128_128.pkl')
    #score_files.append('./output_dir/val_reloc/feats_score_00_origin_fix_128.pkl')
    #score_files.append('./output_dir/val_reloc/feats_score_00_float_128_fix_128.pkl')
    score_files.append('./output_dir/val_reloc/feats_score_00_orgin4096fix8bit_fix_128.pkl')
    #score_files.append('./output_dir/val_reloc/feats_score_00_float_4096_fix_4096.pkl')
    score_files.append('./output_dir/val_reloc/feats_score_00_origin_128.pkl')
    #score_files.append('./output_dir/val_reloc/feats_score_00_460ori_128.pkl')
    #score_files.append('./output_dir/val_reloc/feats_score_00_origin_4096.pkl')
    #score_files.append('./output_dir/val_reloc/feats_score_00_float_128_128.pkl')

    main(score_files)
