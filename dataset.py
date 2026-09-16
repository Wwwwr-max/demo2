from torch.utils.data import Dataset
import torch


class Mydata(Dataset):
    def __init__(self,root,label2id,id2label):
        self.root = root
        self.label2id = label2id
        self.id2label = id2label
        self.items,self.label_num = self.read_data(root)

    @staticmethod
    def read_data(root):
        items = []
        text_list = []  # 暂存这一句
        label_list = []  # 暂存这一句的标签
        label_num = set()  # 用来统计标签种类
        with open(root, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    text, label = line.split(maxsplit=1)
                    label_num.add(label)
                    text_list.append(text)
                    label_list.append(label)
                else:  # 是上一条句子的结尾
                    items.append({
                        'text': text_list,
                        'labels': label_list
                    })
                    text_list = []  # 再次置空存放下一个句子
                    label_list = []
                    continue
        return items, label_num
    @staticmethod
    def make_collate_fn(tokenizer, max_length):
        def collate_fn(items):
            texts = [x['text'] for x in items]
            labels = [x['labels'] for x in items]
            out = tokenizer(
                texts,
                is_split_into_words=True,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            )
            process = []
            for batch_index, label_ids in enumerate(labels):
                word_ids = out.word_ids(batch_index=batch_index)
                pre_word_ids = None
                ids = []
                for word_id in word_ids:
                    if word_id is None:
                        ids.append(-100)
                    elif word_id != pre_word_ids:
                        ids.append(label_ids[word_id])
                    else:
                        ids.append(-100)
                    pre_word_ids = word_id
                process.append(ids)
            out["labels"] = torch.tensor(process, dtype=torch.long)
            return out

        return collate_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self,index):
        sample = self.items[index]      #取出样本
        text = sample['text']
        str_labels = sample['labels']
        tag_id = [self.label2id[s] for s in str_labels]
        return {'text':text,'labels':tag_id}



