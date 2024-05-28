import requests
import json
import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas.metrics.critique import harmfulness
from ragas import evaluate
from ragas import adapt
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

class RAGAs_Eval():
    def __init__(self):
        openai_model = ChatOpenAI(model_name="gpt-4")
        print("正在初始化RAGAs评分系统, 首次运行等待时间较长.....")
        adapt(metrics=[answer_correctness, faithfulness, answer_relevancy, context_recall, context_precision], language="chinese", llm=openai_model)

    # 因为RAGAs系统使用GPT打分, 有最大tokens限制, 需要保证Top-k个contexts没有超过tokens限制。
    def max_k(self, data, k, max_tokens=13000):
        while True:
            # 选取前k个context
            if k > len(data['contexts']):
                k = len(data['contexts'])
            data['contexts'] = data['contexts'][:k]
            contents = ''
            for response in data['contexts']:
                for res in response:
                    contents+=res

            tokens = len(contents)

            # maximum tokens is 16385, set 13000 as maximum context token limits
            if tokens<max_tokens:
                break

            k-=1

        return data, k

    # 遍历所有result文件夹中的内容，生成一个eval_dataset, 据此打分
    def top_k_ragas_eval(self, question_list, contexts_list, answer_list, ground_truth_list, k=10):
        scores = []
        for index, question in enumerate(question_list):
            print(f'Question {index+1}/{len(question_list)}')
            data = {"question": question, "contexts":contexts_list[index], "ground_truth":ground_truth_list[index], 'answer':answer_list[index]}      

            # 验证top-k个contexts是否超过max_tokens = 16385, 并选取前k个context
            data, new_k = self.max_k(data, k)

            if new_k!=k:
                print(f"Top-{k} contexts 超过最大字符限制, 本次结果 '{question}' 自动更改为Top-{new_k}")

            eval_dataset = Dataset.from_pandas(pd.DataFrame([data]))

            # 使用RAGAs评分系统打分，最后共输出4个分数
            result = evaluate(
                eval_dataset,
                metrics=[
                    answer_relevancy,
                    faithfulness,
                    context_recall,
                    context_precision,
                ],
            )
            scores.append(result)

        return scores

    # 将每一个问题的分数保存到./result/result.xlsx中
    def save(self, scores):
        results = []
        for s in scores:
            results.append(s.to_pandas())

        result = pd.concat(results).reset_index(drop=True)

        save_path = './result'
        file_name = 'result.xlsx'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        result.to_excel(os.path.join(save_path, file_name))

    def run(self, question_list, contexts_list, answer_list, ground_truth_list, k=10):
        scores = self.top_k_ragas_eval(question_list, contexts_list, answer_list, ground_truth_list, k)
        self.save(scores)
        return scores