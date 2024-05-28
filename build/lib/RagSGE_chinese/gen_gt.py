import os
from tqdm import tqdm
import openai
import tiktoken
import random
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate


class Gen_GT():
    def __init__(self):
        # 加载openai key
        openai.api_key = os.getenv("OPENAI_API_KEY")

    # GPT 长文档问题回答 (分割文本+每个文本单独回答)
    def send(self, prompt, text_data, chat_model="gpt-3.5-turbo", model_token_limit=8192, max_tokens=2000):
        """
        该方程可以使用GPT, 结合全部的contexts回答问题, 如果contexts超过GPT token limits, 则将contexts分割后一个个输入

        一开始先输入prompt,然后通过OpenAI API将文本数据分块发送给ChatGPT。
        如果文本数据过长, 将其分割成多个块, 然后分别发送每个块。每个块都会让GPT生成一个答案

        参数：
        - prompt (str)：用于引导模型响应的提示。
        - text_data (str)：需要包含的额外文本数据。
        - max_tokens (int, 可选)：如果文本过长，分割的chunk_size, 默认值为2000
        - model_token_limit (int, 可选)：GPT模型最大token_limit, 如果全部文本长度超过该值，则删除最先输入的文本内容，默认值为8192

        返回值：
        - list：GPT的回答。
        """

        # 将文本数据tokenize
        tokenizer = tiktoken.encoding_for_model(chat_model)
        token_integers = tokenizer.encode(text_data)

        if len(token_integers) + len(tokenizer.encode(prompt))> max_tokens:
            chunk_size = max_tokens
        else:
            chunk_size = len(token_integers)
        
        # 将文本内容根据max_tokens/chunk_size切分
        chunks = [
            token_integers[i : i + chunk_size]
            for i in range(0, len(token_integers), chunk_size)
        ]
        chunks = [tokenizer.decode(chunk) for chunk in chunks]

        # 初始化输入message, 包含问题的prompt
        responses = []
        messages = [
            {"role": "user", "content": prompt},
        ]

        # 遍历全部的片段，并结合prompt+question生成答案
        for chunk in chunks:
            messages.append({"role": "user", "content": chunk})

            # 如果全部文本长度超过该值，则删除最先输入的文本内容
            while (
                sum(len(tokenizer.encode(msg["content"])) for msg in messages)
                > model_token_limit
            ):
                messages.pop(1)  # Remove the oldest chunk

            response = openai.chat.completions.create(model=chat_model, messages=messages)
            chatgpt_response = response.choices[0].message.content.strip()
            responses.append(chatgpt_response)

        return responses

    # GPT 长文档问题回答 (合并总结多个回答，最终仅生成一个回答)
    def generate_gt(self, question, context_list, chat_model='gpt-4-turbo'):
        """
        该方程与上一个方程结合使用，用于总结全部回答并最终将200个回答缩减成一个标准答案ground_truth

        参数：
        - company_name (str)：公司名
        - question_body (str)：问题主体
        - context_list (list(str))：200个contexts列表
        - chat_model (str, 可选)：生成答案使用的GPT模型，默认为'gpt-4-turbo'

        返回值：
        - str：GPT最后生成的标准答案ground_truth。
        """

        responses = []

        prompt_text = f'''
                        请根据给定的文档回答问题
                        如果文档中的内容可以回答问题，请全面详细地回答问题并在回答中使用资料中细节，包括例子，数据等；如果不可以回答问题，请回答不知道
                        问题: {question}                
                        请不要输出与回答问题无关的任何内容
                        '''

        # 输入context_list(200个)，每一个context都生成一个答案
        for context in tqdm(context_list, desc="初始答案生成中(慢速版)"):
            response=self.send(prompt=prompt_text, text_data=context, chat_model=chat_model)
            responses.append(response)

        # 循环缩减回答，最后仅输出一个回答
        print("正在循环缩减回答.....")
        while len(responses)>1:
            contents = ''
            prompt_text = f'''
                            请根据以下给定的背景资料，全面详细地回答问题，请在回答中使用资料中细节，包括例子，数据等
                            问题: {question}
                            输出格式：“答案: xxx"
                            '''

            for response in responses:
                for res in response:
                    if '不知道' in res:
                        continue
                    contents+=res

            responses = self.send(prompt=prompt_text, text_data=contents, chat_model=chat_model)

        return responses[0]
    
    # 将contexts随机分割成小的数据集, 每个数据集大小为split_size
    def split_dataset(self, contexts, max_item=20):
        contexts_datasets = []       
        length = len(contexts)
        random.shuffle(contexts)
        num_dataset = int(len(contexts)/max_item)
        for _ in range(num_dataset):
            contexts_datasets.append(contexts[:max_item])
            contexts = contexts[max_item:]
        if num_dataset*max_item<length:
            diff = length-num_dataset*max_item
            contexts_datasets.append(contexts[:diff])
        return contexts_datasets


    # GPT 长文档问题回答 快速版 (不需要先依次回答200个contexts, 直接20个为一组回答再合并)
    def generate_gt_fast(self, question, contexts, chat_model='gpt-4-turbo', max_item=20):
        """
        该方程与上一个方程结合使用，用于总结全部回答并最终将200个回答缩减成一个标准答案ground_truth

        参数：
        - question (str): 问题
        - context_list (list(str))：200个contexts列表
        - chat_model (str, 可选)：生成答案使用的GPT模型，默认为'gpt-4-turbo'

        返回值：
        - str：GPT最后生成的标准答案ground_truth。
        """

        prompt_text = '''
                        请根据给定的文档回答问题
                        如果文档中的内容可以回答问题，请全面详细地回答问题并在回答中使用资料中细节，包括例子，数据等；如果不可以回答问题，请回答不知道
                        问题: {question}     
                        文档: {context_list}           
                        请不要输出与回答问题无关的任何内容
                        输出格式：“答案: xxx"
                        '''
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        prompt = ChatPromptTemplate.from_template(prompt_text)
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=chat_model)
        parser = StrOutputParser()
        chain_bin = prompt | model | parser

        contexts_datasets = self.split_dataset(contexts[:], max_item=max_item)

        # 针对每个小的数据集生成一个问题
        answers = []
        for contexts_data in tqdm(contexts_datasets, desc='初始答案生成中(快速版)'):
            answer = chain_bin.invoke({
                    "question": question,
                    "context_list": contexts_data,
                })
            #print(answer)
            if '不知道' not in answer:
                answers.append(answer)

        # 循环缩减回答
        print("正在循环缩减回答.....")
        while len(answers)>1:
            temp = []
            if len(answers)>max_item:
                num_dataset = int(len(answers)/max_item)
                length = len(answers)
                for _ in range(num_dataset):
                    answer = chain_bin.invoke({
                            "question": question,
                            "context_list": answers[:max_item],
                        })
                    answers = answers[max_item:]
                    temp.append(answer)
                if num_dataset*max_item<length:
                    diff = length-num_dataset*max_item
                    answer = chain_bin.invoke({
                            "question": question,
                            "context_list": contexts[:diff],
                        })
                    temp.append(answer)
            else:
                answer = chain_bin.invoke({
                            "question": question,
                            "context_list": answers,
                        })
                temp.append(answer)
            answers = temp[:]
        return answers[0]


# 本文档包含两种标准答案生成方式
# 1. 针对200个contexts每一个都生成一个标准答案（200个）, 随后循环缩减答案（速度慢，一个问题大约需要15-20min, 适合有token限制时）
# ground_truth = gen.generate_gt(question, contexts, chat_model='gpt-4-turbo')
# 2. 将200个contexts分成20个一组，每一组生成一个答案（共10个答案）, 随后循环缩减答案（速度快，一个问题大约需要2min, 答案可能没有慢速版那么全面，但也还好，需要强大的GPT4）
# ground_truth = gen.generate_gt_fast(question, contexts, chat_model='gpt-4-turbo', max_item=20)

'''
东方航空主营业务
慢速答案：
答案: 东方航空公司的主营业务是航空运输，具体包括客运和货运业务。在客运方面，东航覆盖国内、地区和国际航线，实施关键运营指标如客座率和旅客周转量的管理，
并根据市场需求调整运力投放。货运业务则包括常规的客机腹舱运输以及“客改货”业务，主要由中国货运航空有限公司经营。此外，东方航空还注重服务体验，
如实施灵活的退票和改签规则，以适应旅客需求。公司战略上采用以上海为中心的“中枢网络运营”战略，建立全球覆盖的航线网络，
并强调飞行安全、服务质量与市场营销的国际化。

快速答案：
答案: 东方航空公司的主营业务是航空运输服务，涵盖客运和货运服务。东航不仅经营全球范围内近500条航线，还实施了“中枢网络运营”战略，
以上海为中心建立覆盖国内外的航空网络。此外，东方航空还涉及航空发动机维修和其他航空相关服务，并积极参与国产大飞机项目，比如购买了100架C919飞机。
公司的机队主要包括空中客车A320、A330、A340、波音737和ERJ-145等机型。东航还通过合资或控股方式经营旅行公司，以及通过担保和资金募集等方式支持其
子公司的发展。
'''

