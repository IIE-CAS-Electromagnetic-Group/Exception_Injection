from abnormal_visualization import read_abnormal_label,plot_waterfall_with_annotation
import os
from signal_process import *





root_path="D:\iie\Python_Workspace\Time-Series-Library-main\数据集\电梯信号"

print("开始处理:"+root_path)

#win_size=args.seq_len
win_size=10

read_coarse_grained_data(root_path, win_size//2, win_size//2, flag="train")
read_fine_grained_data(root_path, win_size//2, win_size//2, flag="train")
read_coarse_grained_data(root_path, win_size//2, win_size//2, flag="test")
read_fine_grained_data(root_path, win_size//2, win_size//2, flag="test")
#######################异常注入可视化########################
abnormal_label_list=read_abnormal_label(os.path.join(root_path,"raw_data/abnormal_label/abnormal_process.csv"))
# 调用函数绘制图表
plot_waterfall_with_annotation(os.path.join(root_path,"fine_grained_data/test_fine_grained_data.csv"),
                               abnormal_label_list)