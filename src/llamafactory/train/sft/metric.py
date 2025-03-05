# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer

if is_jieba_available():
    import jieba  # type: ignore

if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

if is_rouge_available():
    from rouge_chinese import Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        # Calculate GAUC,Hit,NDCG score
        try:
            # print('eval_preds:', eval_preds)
            print('decoded_labels:', decoded_labels)
            print('decoded_preds:', decoded_preds)

            k_values = [1, 5, 10, 15, 20, 50]
            self.score_dict["gauc"] = cal_gauc(decoded_labels, decoded_preds)
            cal_top_k = evaluate_top_k(decoded_labels, decoded_preds, k_values)
            for k in k_values:
                k1 = "0" + str(k) if k < 10 else k
                self.score_dict[f"hit@{k1}"] = cal_top_k[k]['hit']
                self.score_dict[f"ndcg@{k1}"] = cal_top_k[k]['ndcg']
        except Exception as e:
            print(e)

        if compute_result:
            return self._dump()


def parse_items(item_strings):
    """
    将原始字符串列表转换为列表的列表，每个子列表包含单个用户的项目ID。
    """
    parsed = []
    for s in item_strings:
        items = s.strip().split(',')
        items = [item.strip() for item in items]
        parsed.append(items)
    return parsed


def cal_gauc(decoded_labels, decoded_preds):
    """
    计算GAUC指标。

    Args:
        decoded_labels: list of lists, 每个子列表包含用户的真实相关项目。
        decoded_preds: list of lists, 每个子列表包含用户的预测项目，按排名顺序。
        all_items: set, 所有可能的项目集合。

    Returns:
        float: GAUC值。
    """

    # 数据预处理
    decoded_labels = parse_items(decoded_labels)
    decoded_preds = parse_items(decoded_preds)

    # 获取所有可能的项目
    all_items = set()
    for items in decoded_labels + decoded_preds:
        all_items.update(items)

    gauc_sum = 0.0
    total_pos = 0

    for user_idx in range(len(decoded_labels)):
        true_items = set(decoded_labels[user_idx])
        pred_items = decoded_preds[user_idx]

        if not true_items:
            print(f"User {user_idx} has no positive samples. Skipping.")
            continue

        # Assign ranks to items
        rank_dict = {item: rank for rank, item in enumerate(pred_items, start=1)}

        # Calculate sum of ranks for positive items
        pos_rank_sum = sum(rank_dict.get(item, len(pred_items) + 1) for item in true_items)
        pos_len = len(true_items)
        neg_len = len(all_items) - pos_len

        if neg_len <= 0:
            print(f"User {user_idx} has no negative samples. Skipping.")
            continue

        # Apply GAUC formula
        pair_num = pos_len * (len(all_items) + 1) - pos_len * (pos_len + 1) / 2 - pos_rank_sum
        auc_u = pair_num / (pos_len * neg_len)
        gauc_sum += auc_u * pos_len
        total_pos += pos_len

    gauc = gauc_sum / total_pos if total_pos > 0 else 0.0
    return gauc


def evaluate_top_k(decoded_labels, decoded_preds, k_values):
    """
    Evaluate HR@k and NDCG@k for top-k recommendation.

    Args:
        decoded_labels: list of lists, 每个子列表包含用户的真实相关项目。
        decoded_preds: list of lists, 每个子列表包含用户的预测项目，按排名顺序。
        k_values: list, top-k values to evaluate.

    Returns:
        dict: {k: {'hit': HR_value, 'ndcg': NDCG_value}} for each k.
    """

    labels = {i: row.strip().split(",") for i, row in enumerate(decoded_labels)}
    preds = {i: row.strip().split(",") for i, row in enumerate(decoded_preds)}

    results = {k: {'hit': 0.0, 'ndcg': 0.0} for k in k_values}
    num_users = len(labels)

    for user_id, true_items in labels.items():
        if user_id not in preds:
            continue
        recommended_items = preds[user_id]
        true_items_set = set(true_items)

        for k in k_values:
            top_k_items = recommended_items[:k]
            # Calculate HR@k
            hit = any(item in true_items_set for item in top_k_items)
            results[k]['hit'] += hit # /len(true_items)

            # Calculate NDCG@k
            dcg = 0.0
            idcg = 0.0
            cnt = 0
            for i, item in enumerate(top_k_items):
                if item in true_items_set:
                    dcg += 1.0 / np.log2(i + 2)  # log2(i+2) because i starts at 0
                    idcg += 1.0 / np.log2(cnt + 2)
                    cnt += 1
            results[k]['ndcg'] += dcg / idcg if idcg > 0 else 0.0

    # Average the results
    for k in k_values:
        results[k]['hit'] /= num_users
        results[k]['ndcg'] /= num_users

    return results

