import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
from . import array2D
import os
import json

class VCounter:   #统计器类 用以统计Acc，Pre等参数

    def __init__(self):
        self.name_dict = {}
        self.current_name_count = 0
        self.count = 0
        self.correct = 0
        self.data = array2D.Array2D()
        self.statusData = array2D.Array2D()
        self.Acc = 0
        '''
        statusData中存每个可能的识别结果的统计数据
        按照如下顺序存储
        row0 ------ count       该分类下正样本数
        row1 ------ prob_sum    识别为该分类的总确信度
        row2 ------ correct     该分类下阳性数(True Positives)
        row3 ------ mistake     该分类下阴性数(False Positives)
        row4 ------ misjudge    其他分类被识别为该分类数量(False Negatives)
        统计量
        row5 ------ Precision
        row6 ------ Recore
        row7 ------ F1-Score

        True Negatives由计算得出，值为
            (count_sum - count) - misjudge
        即negatives总数减去假阳性数量

        各统计量计算方法：
            准确率(Accuracy):   对总表，correct_sum / count_sum
                对模型总体准确度的评价
            精确率(Precision):  对每一分类，correct / count
                对单个分类正确识别程度的评价
            召回率(Recall):     对每一分类 correct / (correct + misjudge)
                所有识别为该分类的结果中，正确数量的占比
            F1-Score:           对每一分类 2PR / (P+R)
                Pre与Recall的调和平均数，更好地关注两者中较低值的指标
            混淆矩阵:           直接使用Data内的数据
        '''

    def __str__(self):
        pass

    def clear(self):    #清空类内数据
        self.name_dict = {}
        self.current_name_count = 0
        self.count = 0
        self.correct = 0
        self.data.clear()
        self.statusData.clear()
        self.Acc = 0
        return
    
    def checkKey(self, key_val:str):
        if key_val in self.name_dict:
            return self.name_dict[key_val]
        else:
            self.name_dict[key_val] = self.current_name_count
            self.current_name_count += 1
            self.data.add_row([])
            return self.current_name_count - 1

    def addPair(self, expect_result:str, actual_result:str, prob = 1.00):   #添加一对预期-实际对
        try:    
            expect_val = self.checkKey(expect_result)
            actual_val = self.checkKey(actual_result)
            
            self.data.add(expect_val, actual_val, 1)
            
            self.statusData.add(expect_val, 0, 1)       #计数
            self.statusData.add(expect_val, 1, prob)    #计算sum_prob
            self.count+=1


            if(expect_val == actual_val):
                self.statusData.add(expect_val, 2, 1)   #识别正确，TP+1
                self.correct+=1
            else:
                self.statusData.add(expect_val, 3, 1)   #识别错误，本类FP+1
                self.statusData.add(actual_val, 4, 1)   #识别错误，得到的类FN+1

                
        except "Failed to load typename into VCounter.":
            pass    



    def calc(self):
        self.Acc = self.correct / self.count
        for name in self.name_dict:
            #print(name)
            # print(self.statusData)
            try:
                Pre = self.statusData.get(self.name_dict[name],2) / self.statusData.get(self.name_dict[name],0)
            except:
                Pre = -1
            
            try:
                Recall = self.statusData.get(self.name_dict[name],2) / (self.statusData.get(self.name_dict[name],2) + self.statusData.get(self.name_dict[name],4))
            except:
                Recall = -1
            
            try:
                F1Score = 2 * Pre * Recall / (Pre + Recall)
            except:
                F1Score = -1

            self.statusData.set(self.name_dict[name],5,Pre)
            self.statusData.set(self.name_dict[name],6,Recall)
            self.statusData.set(self.name_dict[name],7,F1Score)
        return

    def print(self):    #打印
        self.calc()
        print('Acc:    {:.2f}'.format(self.Acc*100))
        for name in self.name_dict:
            #print(name)
            # print(self.statusData)

            Pre = self.statusData.get(self.name_dict[name],5)
            Recall = self.statusData.get(self.name_dict[name],6)
            F1Score = self.statusData.get(self.name_dict[name],7)

            print("{}   Pre:{:.2f}%   Recall:{:.2f}%    F1:{:.2f}%".format(name, Pre*100, Recall*100, F1Score*100))
        return
    
    def saveJson(self, file_name:str = "New Statistics",file_path: str = "./" ):
        add_num = 0
        while(os.path.exists(file_path+file_name+ (f"({add_num})" if add_num != 0 else "" )+ ".json")):
            str1 = file_path+file_name+ (f"({add_num})" if add_num != 0 else "" )+ ".json"
            print(f"file \"{str1}\" exists.")
            add_num += 1
        
        str2 = file_path+file_name+ (f"({add_num})" if add_num != 0 else "" )+ ".json"
        print(f"Creating file \"{str2}\"")

        try:
            with open(file_path+file_name+ (f"({add_num})" if add_num != 0 else "" )+ ".json", "w") as json_file:
                self.calc()


                statisticsInfo = []
                details = []

                for name in self.name_dict:
                    statisticsInfo.append(
                        {
                            "name" : name,
                            "Precision" : self.statusData.get(self.name_dict[name], 5) ,
                            "Recall"    : self.statusData.get(self.name_dict[name], 6) ,
                            "F1-Score"  : self.statusData.get(self.name_dict[name], 7) 
                        }
                    )

                for name in self.name_dict:
                    details.append(
                        {
                            "name" : name,
                            "Count" : self.statusData.get(self.name_dict[name], 0) ,
                            "SumProb"    : self.statusData.get(self.name_dict[name], 1) ,
                            "True Positive"  : self.statusData.get(self.name_dict[name], 2) ,
                            "False Positive"    : self.statusData.get(self.name_dict[name], 3),
                            "False Negative"    : self.statusData.get(self.name_dict[name], 4) 
                        }
                    )

                data_all = []
                for i in range(len(self.name_dict)):
                    data_raw = []
                    for j in range(len(self.name_dict)):
                        data_raw.append(self.data.get(i,j))
                    data_all.append(data_raw)

                data = {
                    "Accuracy"  : self.Acc,
                    "ItemScore" : statisticsInfo,
                    "Details"   : details,
                    "Data"      : data_all
                }

                json.dump(data, json_file, indent = 4)

                json_file.close()

        except:
            print("Failed to created File.")
