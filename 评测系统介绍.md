# RAG系统评测

## 评测系统使用方法
评测系统使用方法，请详见 [评测系统使用说明](./评测系统使用说明.md)

## 新方案设计
目前我们的系统中，一个问题大约对应200篇contexts，数量庞大，无法十分准确且快速的依赖RAGAs系统完成模型验证。与此同时，RAGAs系统无法对修改生成器参数/修改顺序给出有效的建议。因此，将任务分为如下两步：

1. 使用GPT针对200篇文章与问题的相关程度进行排序（0-1打分/小数打分/**二分排序**）。最终目的是有一个按照相关度排列的顺序`ground_truth_contexts`，并以此用于评判我们的系统检索顺序的好坏，以此优化模型。
2. 使用优化好全部200篇`contexts`内容导入GPT，生成全面的`ground_truth`, 随后使用`top-k`个`contexts`, `question`, `answer`, `ground_truth`以及开源评价模型RAGAs对生成器和检索器评分

## 0. 从ES系统中获取数据
1. 需要从ES系统中获取对应的contexts数据，获取方式如下：

```python
#使用ES客户端
from elasticsearch import Elasticsearch

es = Elasticsearch(
[ES_SERVER_ADDRESS],  # Elasticsearch地址
basic_auth=(ES_USER_NAME, ES_PASSWORD)  # 替换username和密码
)

# 调用函数执行查询
index = "company-news"
response = es.search(
    index=index,
    _source_excludes=["content_embedding", "keywords_embedding"],
    size=query_size,
    query={
        "bool": {
            "must": [
                {
                    "match_phrase": {
                        "collection_info.collect_id": company_name
                    }
                }
            ],
            "should": [
                {
                    "match": {
                        "metadata.content": {
                            "query": question_body,
                            "boost": 8
                        }
                    }
                }
            ]
        }
    }
)
```
2. 从ES系统中寻找答案 answer:
```python
response = ''
url = "http://117.50.190.205:8001/qa"
params = {
    "message": question_body,
    "company_name": company_name
}
s = requests.Session()

while response == '':
    with s.get(url, params=params, stream=True) as resp:
        if resp.status_code == 200:
            for line in resp.iter_lines():
                if line:
                    response = line.decode('utf-8')
                    #print(line.decode('utf-8'))
        else:
            print(f'错误：服务器返回状态码 {resp.status_code}, 请检查是否关闭了VPN')

    if response=='':
        print('错误：无法获得响应信息，正在重新获取......')

s.close()
```

## 1. contexts排序 `es_context_sort.py`

### 排序系统介绍
本排序系统能够完成如下：
1. 从`ElasticSearch`中获取有关问题的200个文档
2. 先遍历全部200篇文档，让GPT0-1排序，查看文档是否包含问题的答案（可选）
3. 然后让GPT对有用的文档两两对比，完成相关度排序
4. 使用`spearman_score`和`top-k similarity`量化对比排序质量

最终输出结果包含一个长度为`n`的数组 `indices`，`n`为总文件数量。
数组内包含`0~n-1`，每个元素对应原本输入数组`context_list`的下标。在输出数组`indices`中，**越靠前的下标代表GPT认为在回答问题的时候更重要，越靠后的越不重要**

**输出样例：**
```
GPT-3.5 n=30:
输入问题：company_name = "东方航空公司"
         question_body = "主营业务"

输出样例：
# Sorted Good indices: [55, 178, 0, 59, 169, 11, 9, 1, 5, 13, 56, 28, 53, 95, 3, 186, 21, 22, 20, 57, 25, 26, 38, 70, 98, 106, 165, 182, 54, 99, 111, 129, 140, 141, 144, 192, 87, 51, 168, 17, 19, 30, 33, 31, 32, 36, 34, 35, 39, 40, 67, 47, 41, 49, 78, 121, 122, 127, 132, 42, 45, 61, 62, 80, 69, 85, 153, 128, 135, 180, 137, 152, 154, 172, 187, 195, 198, 160]
# Bad indices: [2, 4, 6, 7, 8, 10, 12, 14, 15, 16, 18, 23, 24, 27, 29, 37, 43, 44, 46, 48, 50, 52, 58, 60, 63, 64, 65, 66, 68, 71, 72, 73, 74, 75, 76, 77, 79, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 93, 94, 96, 97, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 123, 124, 125, 126, 130, 131, 133, 134, 136, 138, 139, 142, 143, 145, 146, 147, 148, 149, 150, 151, 155, 156, 157, 158, 159, 161, 162, 163, 164, 166, 167, 170, 171, 173, 174, 175, 176, 177, 179, 181, 183, 184, 185, 188, 189, 190, 191, 193, 194, 196, 197, 199]
# Spearmans correlation coefficient: 0.443
# Top-10 similarity score is: 0.6
# Overlap indices are: [0, 11, 9, 1, 5, 13]
```
#### 0-1排序：
遍历全部的文章documents  `context_i = [d_1, d_2, ..., d_n], n<=200`， 让GPT分析文章`d_i`中是否包含问题`q`的答案，如果有，则将该文档的下标保存在`good`列表中，否则保存在`bad`列表中。

#### 二分排序：

随机选择一个文档作为中点，其他文章与该文章两两排序并打分，二分训练排序

a. 针对单一问题`q_i`, 我们需要对`good`列表中的每一项进行排序，排序方法如下：

b. 初始化`sorted = []`作为排序好的数列值，并遍历`good_i：d_i`

1. 将`d_i`与已经排好的序列中点进行比较 `d_i and sorted[mid]`
2. 使用GPT with prompt, 可行的prompt:

```python
'''
There are two contexts given below, please choose one of the context to answer the question given.
If you choose context1, please output "1". If you choose context2, please output "2".
You must choose one of the context and output either "1" or "2". Please do not output anything else

Question: {question}

Context1: {context1}   # doc_i
Context2: {context2}   # sorted[mid]

# output 1: doc_i排在sorted[mid]前面
# output 2: doc_i排在sorted[mid]后面
'''
```
3. 根据二分法条件循环查找，完成排序，复杂度：`O(nlogn)`



#### 排序评价： 

在此程序中，我们使用了两种方法量化评价GPT和ES的排序，分别是`spearman_score`和`top-k similarity`，**目前方案不完善，可能会有更好的评价标准**

**spearman_score:**

它是衡量两个变量的依赖性的非参数指标。它评估的是两个变量的等级之间的相关性，而不直接关注变量的具体值，适用于定序数据及非正态分布的数据分析。简单来说，该相关系数是用来判断两组数据是否具有一致的等级顺序（排序）。

计算方法：
1. 排列等级：首先，对每一组数据进行排序，赋予等级。如果有相同的值，赋予它们的平均等级。
2. 计算等级差：然后，计算每对数据点等级之间的差异（差的平方称为`d_i^2`）。
3. 使用公式计算：最后，应用下面的公式来计算`score`：

<p align="center">
    <img src=./imgs/image.png /> 
</p>

`Spearman_score` 值的区间为 `-1~1`. `1`代表两个排列顺序正相关；`-1`代表负相关。

<p align="center">
    <img src=./imgs/image-1.png /> 
</p>

**top-k similarity:**

这个是一个自制方法，用于对比两个数组在前k个值中的重复比率，并返回重复的内容。该方法可以用于对比GPT和ES两者排序后的`indices`在最相关的top-k个元素中重复的比率，从而推断出检索器的好坏。

## 2. 生成标准答案 `gen_gt.py`
`gen_gt.py`使用GPT生成标准答案，共有快速，慢速两个版本。

**慢速版**具体实现方法如下：
1. 用户给评测系统上传
    - 一个问题 `q`
    - 一个针对该问题的contexts的集合 `C = {c_1, ..., c_n}, n=200` 
2. GPT 遍历所有文档`C`, 并针对每一个`c_i`生成一个答案`a_i`。
    - 如果GPT认为可以回答问题，则`a_i`为生成的答案
    - 如果不能回答问题，`a_i = "不知道"`
3. GPT 将200个答案 `{a_1, ..., a_200}` 合并, 循环总结，得出最终标准答案
    - 如果合并后的答案超出GPT字符限制，则自动拆分成小片段输入，得出多个答案后再次合并，以此循坏

**快速版**具体实现方法如下：
1. 用户给评测系统上传
    - 一个问题 `q`
    - 一个针对该问题的`contexts`的集合 `C = {c_1, ..., c_n}, n=200` 
2. shuffle `contexts`集合，随后将contexts分为20个一组 `{C_1 = {c_1, ..., c_20}; C_2; ... ; C_10}`, 每一组生成一个答案 `[A_1; ...; A_10]`
3. GPT 将答案 `[A_1; ...; A_10]` 合并, 循环总结，得出最终标准答案

文档所使用的prompt：
```python
# 慢速版
'''
请判断给定的文档中是否包含问题的答案，如果有请回答问题，如果没有请回答不知道
问题: {question}
请不要输出与回答问题无关的任何内容
'''
'''
请根据一下给定的背景知识，回答问题
问题: {question}
输出格式：“答案: xxx"
'''

# 快速版
'''
请根据给定的文档回答问题
如果文档中的内容可以回答问题，请全面详细地回答问题并在回答中使用资料中细节，包括例子，数据等；如果不可以回答问题，请回答不知道
问题: {question}     
文档: {context_list}           
请不要输出与回答问题无关的任何内容
输出格式：“答案: xxx"
'''
```

该程序共输出三个文件，用于后续测试：
| 文件位置 | 文件描述 | 是否输出
| :---: | :---: | :---: |
| ./gt/GT_{question}.json | 该文件储存了三个变量 <br> question, contexts, ground_truth <br> 可用于RAGAs评分 | 如果没有answer=None, 且save=True
|./full/FULL_{question}.json | 该文件储存了四个变量 <br> question, contexts, ground_truth, answer <br> 可直接用于RAGAs评分 | 如果有answer, 且save=True


**`./gt/GT_{question}.json`示例：**
```
{
    "question": "优刻得科技股份有限公司主营业务",
    "contexts": [
        "前言\n优刻得科技股份有限公司主营业务是自主研发并提供以计算、存储、网络等企业必需的基础IT产品为主的云计算服务。公司主要产品或服务有公有云、私有云、混合云、数据可信流通平台安全屋...",
        "优刻得——2023企业能力分析研究报告...",
        ...200个...        
    ],
    "ground_truth": "答案: 优刻得科技股份有限公司主营业务是自主研发并提供以计算、存储、网络等企业必需的基础IT产品为主的云计算服务，包括公有云、私有云、混合云、数据可信流通平台安全屋等产品或服务。"
}
```
**`./full/FULL_{question}.json`示例：**
```
{
    "question": "优刻得科技股份有限公司主营业务",
    "contexts": [
        "前言\n优刻得科技股份有限公司主营业务是自主研发并提供以计算、存储、网络等企业必需的基础IT产品为主的云计算服务。公司主要产品或服务有公有云、私有云、混合云、数据可信流通平台安全屋...",
        "优刻得——2023企业能力分析研究报告...",
        ...200个...        
    ],
    "ground_truth": "答案: 优刻得科技股份有限公司主营业务是自主研发并提供以计算、存储、网络等企业必需的基础IT产品为主的云计算服务，包括公有云、私有云、混合云、数据可信流通平台安全屋等产品或服务。"
    "answer": "优刻得科技股份有限公司主营业务包括：\n1. 提供公有云、混合云、私有云、专有云等综合性行业解决方案\n2. 提供技术开发、技术转让、技术咨询和技术服务\n3. 自主研发iaas、paas、大数据流通平台、ai服务平台等云计算产品\n4. 提供云计算服务、新兴软件及服务、大数据服务、云软件服务等"
}
```

## 3. RAGAs评分 `ragas_eval.py`
`ragas_eval.py`采用开源的RAGAs评分系统，为检索器和生成器提供共4个分数。
- 用户上传 `question_list`, `contexts_list`, `answer_list` 和 `ground_truth_list` `Q = {q_1, ..., q_m}, m个问题`
- 随后调用RAGAs系统评分给出分数 `S = {s_1, ..., s_m}, s_i = {四个分数}`
- 最后将`S`保存在`./result/result.xlsx`中可视化展示

**用户上传的列表格式：**
| 字段 | 说明 | 类型 | 是否必填 |
| :---: | :---: | :---: | :---: |
| question_list | 问题 | 字符串 | 是 |
| contexts_list | 测试模型收集到的背景知识 | 数组 | 是 |
| answer_list | 测试模型生成的答案 | 字符串 | 是 |
| ground_truth_list | GPT生成的标准答案 | 字符串 | 是 |

> 由于RAGAs评分系统采用LLM大模型打分，模型输入时有最大tokens限制，因此无法输入全部200个contexts。本系统默认选择Top-10 contexts 用于评价，经测试不会超过限制。也可以手动修改Top-k值，系统将自动评判是否超过tokens限制，如果超过，则自动降低k值至符合条件。

**score字段格式：**
| 字段 | 说明 | 类型 |
| :---: | :---: | :---: |
| Context precision | 衡量检索出的上下文中有用信息与无用信息的比率。该指标通过分析 <br> question 和 contexts 来计算。检索的背景是否与问题有关 | 浮点数 |
| Context recall | 用来评估是否检索到了解答问题所需的全部相关信息。这一指标依据 <br> ground_truth 和 contexts 进行计算。检索的背景是否与标准答案 <br> 相关 | 浮点数 |
| Faithfulness | 用于衡量生成答案的事实准确度。它通过对比给定上下文中正确的陈述与 <br> 生成答案中总陈述的数量来计算。这一指标结合了 contexts 和 <br> answer。生成的答案是否来源于背景 | 浮点数 |
| Answer relevancy | 评估生成答案与问题的关联程度。 | 浮点数 |

**score示例:**
```
{
 'context_precision': 0.8000, 
 'context_recall': 0.7000, 
 'faithfulness': 0.9667, 
 'answer_relevancy': 0.9601
}
```
**result.xlsx示例:**
![alt text](./imgs/image-2.png)

# 项目合并流程 Pipeline `eval_pipeline.py`
`eval_pipeline.py` 整合了标准答案生成和RAGAs系统评分两个步骤，具体使用方法请参考文档 [评测系统使用说明](./评测系统使用说明.md)

# 后续改进地方
1. ~~完善排序顺序，可以先0-1排序后再二分排序，增强头部排序准确度，降低排序时间~~
2. ~~完善GT生成结果，让GPT更详细的回答问题，可以尝试更改prompt和GPT模型~~
3. ~~完善整体pipeline，细化输入输出~~
4. ~~快速版标准答案生成~~
5. 思考下如何增强答案生成，使用非缩略生成方法，200contexts随机sample20个小dataset, 随后一次性生成答案。