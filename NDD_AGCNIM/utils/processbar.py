import sys
from math import ceil, floor
import time


def load_data_process_bar(cur_length, process_length, dataset_name):
    print("\r", end="")
    print("正在加载数据集--[{}%] |".format(ceil((cur_length) /
          process_length * 100)) + "█" * cur_length +  " " * (process_length -
          cur_length) + "| 数据集:" + dataset_name , end="")
    sys.stdout.flush()
    if(cur_length == process_length):
        print("\n")


def fit_process_bar(cur_length, process_length, loss):
    print("\r", end="")
    print("正在训练--[{}]% |".format(ceil((cur_length) / process_length * 100)) + "█" * cur_length + " " * (process_length -
          cur_length) + "| 损失:" + str(loss), end="")
    sys.stdout.flush()
    if(cur_length == process_length):
        print("\n")



def evaluate_process_bar(cur_length, process_length, predict_accurcy):
    print("\r", end="")
    print("正在评估--[{}]% |".format(ceil((cur_length) / process_length * 100)) + "█" * cur_length + " " * (process_length -
          cur_length) + "|", "均方误差:%.4f"%predict_accurcy, end="")
    sys.stdout.flush()
    if(cur_length == process_length):
        print("\n")
    
def gen_data_process_bar(cur_length, process_length, dataset_name):
    print("\r", end="")
    print("正在生成数据集--[{}%] |".format(ceil((cur_length) /
          process_length * 100)) + "█" * cur_length +  " " * (process_length -
          cur_length) + "| 数据集:" + dataset_name , end="")
    sys.stdout.flush()
    if(cur_length == process_length):
        print("\n")
