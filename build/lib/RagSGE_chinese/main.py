from .es_context_sort import DOC_SORT
from .eval_pipeline import Pipeline

# Sort Context
def sort(question_list, contexts_list, k=10, top=True, fast=True):
    compare = DOC_SORT()
    results = compare.run(question_list=question_list, contexts_list=contexts_list, k=k, top=top, fast=fast)
    return results

# Generate Ground Truth
def generate(question_list, contexts_list, chat_model='gpt-4-turbo'):
    p = Pipeline()
    ground_truth = p.run(question_list, contexts_list, answer_list=None, ground_truth_list=None, chat_model=chat_model, k=10, fast=True)
    return ground_truth

# RAGAs Evaluation
def evaluate(question_list, contexts_list, answer_list, ground_truth_list):
    p = Pipeline()
    score = p.run(question_list, contexts_list, answer_list, ground_truth_list, chat_model='gpt-4-turbo', k=10, fast=True)
    return score