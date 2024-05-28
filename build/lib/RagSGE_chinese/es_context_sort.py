# 测试全部200个文件并排序
import json
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm
from scipy.stats import spearmanr
from .compactjsonencoder import CompactJSONEncoder

class DOC_SORT:

    template_bin = """
        有两个给定的背景，请选择一个背景来回答下面的问题。
        如果您选择背景1，请输出"1"。如果您选择背景2，请输出"2"。
        您必须选择一个背景。
        请输出是"1"或"2"。不要输出其他内容。

        问题: {question}

        背景1: {context1}
        背景2: {context2}
        """
        
    template_01 = """
        请判断使用下面文档中内容是否可以回答问题，如果可以请回答"1"，如果不可以请回答"0"
        问题: {question}
        文档: {context}
        请不要输出"0"或"1"以外的任何内容
        """
    
    OPENAI_API_KEY = ""

    # 初始化GPT模型, 模型初始化为GPT-3.5
    def __init__(self, GPT_model_name="gpt-3.5-turbo"):
        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        prompt_bin = ChatPromptTemplate.from_template(self.template_bin)
        prompt_01 = ChatPromptTemplate.from_template(self.template_01)
        model = ChatOpenAI(openai_api_key=self.OPENAI_API_KEY, model=GPT_model_name)
        parser = StrOutputParser()
        self.chain_bin = prompt_bin | model | parser
        self.chain_01 = prompt_01 | model | parser

    def sort_01(self, contexts, question):
        '''
        contexts: 数组， 200个背景知识的数组
        questions: 问题

        输出：
        good: 有答案的contexts, 需要继续使用二分排序精确排序
        bad: 没有答案的contexts, 无需排序
        '''
        good = []
        bad = []

        for i, context in enumerate(tqdm(contexts,desc="0-1排序")):
            res = self.chain_01.invoke({
                "question": question,
                "context": context,
            })

            if res == "1":     # 可以回答，good
                good.append(i)
            elif res == "0":   # 不可以回答，bad
                bad.append(i)
            else:              # 测试用例：如果GPT给出异常输出，打印出来（测试时并未出现问题）
                if '1' in res:
                    good.append(i)
                    print(f"\n compare warning1 (可忽略): {res}")
                elif '0' in res:
                    bad.append(i)
                    print(f"\n compare warning0 (可忽略): {res}")
                else:           # 不知道回答了啥，报错，放入bad
                    bad.append(i)
                    print(f"\n compare error: {res}")
        
        return good, bad

    # 使用GPT比较两个contexts，并给出比较
    def doc_compare(self, doc1, doc2, question):
        '''
        doc1: 第一个比较的文档
        doc2: 第二个比较的文档
        question: 问题
        chain: GPT 模型输入
        doc1_high: 比较输出, True = doc1更重要, False: doc2更重要
        '''
        compare_response = self.chain_bin.invoke({
            "question": question,
            "context1": doc1,
            "context2": doc2
        })

        if compare_response == "1":     # doc1更重要，排在前面
            doc1_high = True
        elif compare_response == "2":   # doc2更重要，排在前面
            doc1_high = False
        else:                           # 测试用例：如果GPT给出异常输出，打印出来（测试时并未出现问题）
            if '1' in compare_response:
                doc1_high = True
                print(f"\n compare warning1 (可忽略): {compare_response}")
            elif '2' in compare_response:
                doc1_high = False
                print(f"\n compare warning2 (可忽略): {compare_response}")
            else:
                doc1_high = False
                print(f"\n compare error: {compare_response}")

        return doc1_high

    # 二分排序法将good contexts排序（GPT两两排序）
    def binary_insert_sort(self, contexts, question, indices=None):

        '''
        contexts: 数组， 200个背景知识的数组
        questions: 问题
        indices: (可选) 可以选择输入0-1排序后有用的数组good, 随后仅排列good数组的序号
                        默认为None，则代表全部200个contexts都需要排序

        输出：
        indices: 按照相关性从强到弱排序的一个索引数组
        comparisons: [测试用例] 输出200个一共比较次数 (1082)
        '''

        # 如果没有输入indices, 默认contexts全排（200个全排）
        if indices is None:
            indices = [x for x in range(len(contexts))]  # 排序前索引序列（0,1,...,199）与context_list对应
            
        length = len(indices)
        comparisons = 0

        # 二分排序，从前往后开始排序
        for i in tqdm(range(1, length), desc="二分排序"):
            value_to_insert = contexts[indices[i]]
            insertion_index = i

            be, en = 0, i - 1
            while be <= en:
                comparisons += 1
                mid = (be + en) // 2
                if self.doc_compare(value_to_insert, contexts[indices[mid]], question):  # 如果当前值应该排在中间值前面，更重要
                    en = mid - 1
                    insertion_index = mid
                else:   # 如果当前值应该排在中间值后面，更不重要
                    be = mid + 1
                    insertion_index = be
                #print("Compared_once")

            index_to_move = indices.pop(i)
            indices.insert(insertion_index, index_to_move)

        return indices, comparisons
    
    # 计算spearman_score
    def spearman_score(self, list1, list2=None):
        '''
        list1: GPT排序好的数组indices
        list2: 需要对比的数组, 默认为None, 原始数组index是0-199排列的, 自动输入
        
        返回值: spearman分数
        '''
        if list2 is None:
            list2 = list(range(len(list1)))
        
        coef, _ = spearmanr(list1, list2)
        return round(coef,3)
    
    # 计算top-k个元素中重复度
    def top_k_similarity(self, list1, list2=None, k=20, top=True):
        '''
        list1: GPT排序好的数组indices
        list2: 需要对比的数组, 默认为None, 原始数组index是0-199排列的, 自动输入
        k: top-k个数字, 数组最前或数组最后
        top: True=对比数组最前面的k个数,  False=对比数组最后面的k个数
        
        sim: 重合度数值
        overlap: 重合数字列列表
        '''
        overlap = []
        count = 0
        if list2 is None:
            list2 = list(range(len(list1)))

        if top:
            for item in list1[:k]:
                if item in list2[:k]:
                    overlap.append(item)
                    count+=1
        else:
            for item in list1[len(list1)-k:]:
                if item in list2[len(list1)-k:]:
                    overlap.append(item)
                    count+=1

        sim = round(count/k, 3)
        return sim, overlap
    
    def run(self, question_list, contexts_list, k=10, top=True, fast=True):
        results = []
        # 默认快速01排序，k=10, top-10
        '''
        排序调用的函数, 输入如下：
        question_list = [q_1, q_2,...,q_n]  全部问题列表 list(str)
        contexts_list = [c_1, c_2, ..., c_n]   其中c_i = [d_1, d_2, ..., d_m], m=一个问题对应的contexts数量, n=全部数据集问题的数量 list(list(str))
        k: Top-k个similarity 测试, 默认为10
        top: 选择是查找Top-k个还是Bottom-k个， 默认为True, 对比最前面的k个
        fast: 选择是否需要先0-1排序提前筛选没用的信息再进行精细排序, 默认为True需要提前0-1筛序
        save: 选择是否需要保存结果, 默认为True需要保存
        '''
        if fast: #如果快速排序
            print("开始快速排序.....")
            for index, question in enumerate(question_list):
                contexts = contexts_list[index]
                print(f'\nQuestion {index+1}/{len(question_list)}')

                # good 和 bad 是包含/不包含回答问题信息的列表，按原输入列表升序排列
                # indices 是根据上下文相关性对 good 列表进行排序的结果
                good, bad = self.sort_01(contexts=contexts, question=question)
                indices, comparisons = self.binary_insert_sort(contexts=contexts, question=question, indices=good[:])
                print(f"Sorted Good indices: {indices}")
                print(f"Bad indices: {bad}")

                # Spearman and Top-k similarity 对比 good and indices(sorted)两个列表, 排除bad列表
                # e.g. [0, 1, 5, 3, 9, 19, 11, 13, 17, 14] 和 [0, 1, 3, 5, 9, 11, 13, 14, 17, 19] ...
                score = self.spearman_score(list1=indices, list2=good)
                print('Spearmans correlation coefficient: %.3f' % score)
                if k > len(indices):
                    if len(indices)>=10:
                        print(f"k值大于有用contexts的长度，将取默认长度10")
                        k = 10
                    else:
                        print(f"k值大于有用contexts的长度，将取最大长度{len(indices)}")
                        k = len(indices)
                sim, overlap = self.top_k_similarity(list1=indices, list2=good, k=k, top=top)
                print(f"Top-{k} similarity score is: {sim}")
                print(f"Overlap indices are: {overlap}")

                res = {"question": question, "contexts": contexts, "good": good, "bad": bad, "sorted": indices, "spearman": score, "sim": sim, "overlap": overlap}
                results.append(res)
            
        else: # 如果200个统一排序
            print("开始全部两两排序.....")
            for index, question in enumerate(question_list):
                contexts = contexts_list[index]
                print(f'\nQuestion {index+1}/{len(question_list)}')
            
                indices, comparisons = self.binary_insert_sort(contexts=contexts, question=question)
                print("Sorted indices:", indices)
                score = self.spearman_score(list1=indices)
                print('Spearmans correlation coefficient: %.3f' % score)
                if k > len(indices):
                    print(f"k值大于有用contexts的长度，将取最大长度{len(indices)}")
                    k = len(indices)
                sim, overlap = self.top_k_similarity(list1=indices, k=k, top=top)            
                print(f"Top-{k} similarity score is: {sim}")
                print(f"Overlap indices are: {overlap}")

                res = {"question": question, "contexts": contexts, "good": None, "bad": None, "sorted": indices, "spearman": score, "sim": sim, "overlap": overlap}
                results.append(res)


        with open(f'./result/sorted_indices.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, cls=CompactJSONEncoder)
        
        return results


'''
输出样例：非快排
Sorted indices: [19, 10, 1, 0, 23, 22, 15, 17, 3, 2, 4, 11, 6, 21, 9, 8, 5, 13, 27, 26, 25, 18, 14, 29, 16, 28, 24, 20, 12, 7]
Spearmans correlation coefficient: 0.370
Top-10 similarity score is: 0.4
Overlap indices are: [1, 0, 3, 2]

快排：
Sorted Good indices: [55, 178, 0, 59, 169, 11, 9, 1, 5, 13, 56, 28, 53, 95, 3, 186, 21, 22, 20, 57, 25, 26, 38, 70, 98, 106, 165, 182, 54, 99, 111, 129, 140, 141, 144, 192, 87, 51, 168, 17, 19, 30, 33, 31, 32, 36, 34, 35, 39, 40, 67, 47, 41, 49, 78, 121, 122, 127, 132, 42, 45, 61, 62, 80, 69, 85, 153, 128, 135, 180, 137, 152, 154, 172, 187, 195, 198, 160]
Bad indices: [2, 4, 6, 7, 8, 10, 12, 14, 15, 16, 18, 23, 24, 27, 29, 37, 43, 44, 46, 48, 50, 52, 58, 60, 63, 64, 65, 66, 68, 71, 72, 73, 74, 75, 76, 77, 79, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 93, 94, 96, 97, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 123, 124, 125, 126, 130, 131, 133, 134, 136, 138, 139, 142, 143, 145, 146, 147, 148, 149, 150, 151, 155, 156, 157, 158, 159, 161, 162, 163, 164, 166, 167, 170, 171, 173, 174, 175, 176, 177, 179, 181, 183, 184, 185, 188, 189, 190, 191, 193, 194, 196, 197, 199]
Spearmans correlation coefficient: 0.443
Top-10 similarity score is: 0.6
Overlap indices are: [0, 11, 9, 1, 5, 13]
'''


