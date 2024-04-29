#coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from typing import Union
import os

from paddlehub.env import DATA_HOME
from paddlehub.utils.download import download_data
from paddlehub.datasets.base_nlp_dataset import SeqLabelingDataset
from paddlehub.text.bert_tokenizer import BertTokenizer
from paddlehub.text.tokenizer import CustomTokenizer


#@download_data(url="https://bj.bcebos.com/paddlehub-dataset/msra_ner.tar.gz")
import os  
  
# 假设你的本地数据集目录如下：  
local_dataset_dir = '/path/to/your/local/dataset'  
  
# 确保数据集目录存在  
if not os.path.exists(local_dataset_dir):  
    raise ValueError(f"Local dataset directory not found: {local_dataset_dir}")  
  
# 假设你的数据集由txt和ann文件组成，并且文件名匹配  
txt_files = [f for f in os.listdir(local_dataset_dir) if f.endswith('.txt')]  
ann_files = [f for f in os.listdir(local_dataset_dir) if f.endswith('.ann')]  
  
# 检查txt和ann文件数量是否一致  
if len(txt_files) != len(ann_files):  
    raise ValueError("Number of txt files does not match number of ann files.")  
  
# 创建一个列表来存储样本  
samples = []  
  
# 遍历所有txt和ann文件对  
for txt_file, ann_file in zip(txt_files, ann_files):  
    txt_path = os.path.join(local_dataset_dir, txt_file)  
    ann_path = os.path.join(local_dataset_dir, ann_file)  
      
    # 读取txt和ann文件内容  
    with open(txt_path, 'r', encoding='utf-8') as f_txt:  
        txt_content = f_txt.read()  
    with open(ann_path, 'r', encoding='utf-8') as f_ann:  
        ann_content = f_ann.read()  
      
    # 这里添加你的文本预处理和标签解析代码  
    # ...  
      
    # 假设你已经将文本和标签处理成了samples列表中的一个元素  
    # samples.append(processed_sample)  
      
    # 示例：直接添加未经处理的样本（仅用于演示）  
    samples.append({'text': txt_content, 'label': ann_content})  
  
# 创建自定义数据集类（如果需要的话）  
# ...  
  
# 这里可以添加使用数据集的代码，比如创建一个DataLoader等  
# ...  
  
# 打印一个样本示例  
print(samples[0])

class MSRA_NER(SeqLabelingDataset):
    """
    A set of manually annotated Chinese word-segmentation data and
    specifications for training and testing a Chinese word-segmentation system
    for research purposes.  For more information please refer to
    https://www.microsoft.com/en-us/download/details.aspx?id=52531
    """
    label_list = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

    def __init__(
            self,
            tokenizer: Union[BertTokenizer, CustomTokenizer],
            max_seq_len: int = 128,
            mode: str = 'train',
    ):
        base_path = os.path.join(DATA_HOME, "msra_ner")

        if mode == 'train':
            data_file = 'train.tsv'
        elif mode == 'test':
            data_file = 'test.tsv'
        else:
            data_file = 'dev.tsv'
        super().__init__(
            base_path=base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_file=None,
            label_list=self.label_list,
            is_file_with_header=True,
        )
