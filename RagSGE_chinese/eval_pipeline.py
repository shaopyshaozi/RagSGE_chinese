import json
import os
from tqdm import tqdm
from pathlib import Path
from .gen_gt import Gen_GT
from .ragas_eval import RAGAs_Eval

class Pipeline():
    def __init__(self):
        self.gt = Gen_GT()
    
    def run(self, question_list, contexts_list, answer_list=None, ground_truth_list=None, chat_model='gpt-4-turbo', k=10, fast=True):
        '''
        Eval_Pipeline最终调用端口, 输入为:
        question_list = [q_1, q_2,...,q_n]  全部问题列表 list(str)
        contexts_list = [c_1, c_2, ..., c_n]   其中c_i = [d_1, d_2, ..., d_m], m=一个问题对应的contexts数量, n=全部数据集问题的数量 list(list(str))
        answer_list = [a_1, a_2, ..., a_n]  每个问题对应的答案 (可选, 默认为None), 如果没有输入, 系统默认仅生成标准答案 list(str)
        ground_truth_list = [g_1, g_2, ..., g_n]   每个问题对应的标准答案 (可选, 默认为None), 如果没有输入, 由系统根据contexts和question生成  list(str)
        chat_model: 生成标准答案的GPT模型 (可选, 默认为'gpt-4-turbo')
        save_data: 是否将数据保存成结构化的数据/是否将数据储存成.json (可选, 默认为True), 会将数据保存在./full或者./gt中
        k: Top-k个contexts用于RAGAs评分 (可选, 默认为10)
        '''
        
        if answer_list is not None:
            self.eval = RAGAs_Eval()
            if ground_truth_list is None:
                print("生成标准答案中.....")
                ground_truths = []
            for index, question in enumerate(question_list):
                contexts = contexts_list[index]
                answer = answer_list[index]
                if ground_truth_list is None:
                    print(f'Question {index+1}/{len(question_list)}')
                    if fast:
                        ground_truth = self.gt.generate_gt_fast(question, contexts, chat_model=chat_model, max_item=20) # 分成20个一组
                    else:
                        ground_truth = self.gt.generate_gt(question, contexts, chat_model=chat_model)
                    ground_truths.append(ground_truth)
                else:
                    ground_truth = ground_truth_list[index]

            print("\nRAGAs评分中.....")
            if ground_truth_list is None:
                score = self.eval.run(question_list, contexts_list, answer_list, ground_truths, k=k)
            else:
                score = self.eval.run(question_list, contexts_list, answer_list, ground_truth_list, k=k)
            print("\n分数结果已保存至./result/result.xlsx中")
            return score
        else:
            #print("仅生成问题的标准答案, 如需进行RAGAs评分, 请提供答案列表answer_list")
            print("生成标准答案中.....")
            ground_truths = []                                       
            for index, question in enumerate(question_list):
                print(f'Question {index+1}/{len(question_list)}')
                contexts = contexts_list[index]
                if fast:
                    ground_truth = self.gt.generate_gt_fast(question, contexts, chat_model=chat_model, max_item=20) # 分成20个一组
                else:
                    ground_truth = self.gt.generate_gt(question, contexts, chat_model=chat_model)
                ground_truths.append(ground_truth)
                

                data = {"question": question, "contexts":contexts, "ground_truth":ground_truth}
                # 将生成的ground_truth连同question和contexts一起存入一个json中
                if not os.path.exists('./gt'):
                    os.makedirs('./gt')
                with open(f'./gt/GT_{question}.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            print("\n标准答案已保存至./gt中") 
            return ground_truths

# 三种调用方式：
# 1. 用户上传question, contexts, answer, 让系统生成标准答案和评分 (常用！！！！)
    # score = p.run(question_list, contexts_list, answer_list, ground_truth_list=None, save_data=True, k=10, fast=True)
    # 返回分数列表
# 2. 用户上传question, contexts, answer, ground_truth, 仅让系统生成评分
    # score = p.run(question_list, contexts_list, answer_list, ground_truth_list, save_data=True, k=10, fast=True)
    # 返回分数列表
# 3. 用户上传question, contexts, 并设置answer_list=None, 仅让系统生成标准答案
    # ground_truth = p.run(question_list, contexts_list, answer_list=None, ground_truth_list=None, save_data=True, k=10, fast=True)
    # 返回标准答案列表