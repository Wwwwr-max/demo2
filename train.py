import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import config,dataset,model_m
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
demo2_config = config.load_config("./configs/demo2_config.json")
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)
dataset_list = [
    {
        "name":"MSRA",
        "train_root":demo2_config["MSRA_train_root"],
        "dev_root":demo2_config["MSRA_dev_root"]
    },
    {
        "name":"weibo",
        "train_root":demo2_config["weibo_train_root"],
        "dev_root":demo2_config["weibo_dev_root"]
    }
]
model_list = [
    {
        "name":"bert-base",
        "model_root":demo2_config["bert_base_path"]
    },
    {
        "name":"bert-wwm",
        "model_root":demo2_config["bert_wwm_path"]
    }
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Prf():
    def __init__(self,y_true,y_pre):
        self.y_true = y_true
        self.y_pre = y_pre
        self.p_list = []
        self.r_list = []
        self.f_list = []

    def process(self):
        self.p_list.clear()
        self.r_list.clear()
        self.f_list.clear()
        target_names = set()
        for t, _, _, _ in self.y_true:
            target_names.add(t)
        for t, _, _, _ in self.y_pre:
            target_names.add(t)
        target_names = sorted(target_names)
        for type_name in target_names:
            entities_true_type = {item for item in self.y_true if item[0] == type_name}
            entities_pred_type = {item for item in self.y_pre if item[0] == type_name}
            tp = len(entities_true_type & entities_pred_type)  # --->TP
            pred = len(entities_pred_type)  # --->TP+FP
            true = len(entities_true_type)  # --->TP+FN

            precision = tp / pred if pred > 0 else 0.0
            recall = tp / true if true > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            self.p_list.append(precision)
            self.r_list.append(recall)
            self.f_list.append(f1)
        return np.mean(self.p_list), np.mean(self.r_list), np.mean(self.f_list)

def train_t(model,loader,device,loss_fn,optim):
    model.train()
    total_loss = 0
    for step,batch in enumerate(loader):
        batch = {k:v.to(device) for k,v in batch.items()}
        output =model(**batch)
        loss = loss_fn(
            output.reshape(-1, output.size(-1)),
            batch["labels"].reshape(-1)
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    aver_loss = total_loss/len(loader)
    return aver_loss

def dev_d(device, loader, model, id2label):
    model.eval()
    total_correct = 0
    total_num = 0
    y_true = set()
    y_pre = set()
    sent_id = 0

    def extract_entities(label_ids):
        entities = set()
        start = None
        cur_type = None

        def close_entity(end):
            nonlocal start, cur_type
            if start is not None:
                entities.add((start, end, cur_type))
                start = None
                cur_type = None

        for pos, label_id in enumerate(label_ids):
            label_name = id2label[label_id]

            if label_name == "O":
                close_entity(pos)
            elif label_name.startswith("B-"):
                close_entity(pos)
                start = pos
                cur_type = label_name[2:]
            elif label_name.startswith("I-"):
                if start is not None and label_name[2:] == cur_type:
                    continue
                close_entity(pos)
            else:
                close_entity(pos)
        close_entity(len(label_ids))
        return entities

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            pre = torch.argmax(output, dim=-1)
            labels = batch["labels"]

            for true_row, pred_row in zip(labels, pre):
                valid = true_row != -100
                true_ids = true_row[valid].tolist()
                pred_ids = pred_row[valid].tolist()

                total_correct += int((true_row[valid] == pred_row[valid]).sum())
                total_num += int(valid.sum())

                for start, end, type_name in extract_entities(true_ids):
                    y_true.add((type_name, sent_id, start, end))

                for start, end, type_name in extract_entities(pred_ids):
                    y_pre.add((type_name, sent_id, start, end))

                sent_id += 1

    acc = total_correct / total_num if total_num else 0.0
    p, r, f = Prf(y_true, y_pre).process()
    return acc, p, r, f


if __name__ == "__main__":
    for data_chose in dataset_list:
        for model_chose in model_list:
            set_seed(42)
            run_name = f"{data_chose['name']}_{model_chose['name']}"
            writer = SummaryWriter(f"./log/{run_name}")
            print(f"数据集为{data_chose['name']},模型为{model_chose['name']}")
            _,labels = dataset.Mydata.read_data(data_chose["train_root"])
            labels = sorted(labels)
            label2id = {label:id for id,label in enumerate(labels)}
            id2label = {id:label for id,label in enumerate(labels)}
            tokenizer = AutoTokenizer.from_pretrained(model_chose["model_root"])
            train = dataset.Mydata(data_chose["train_root"],label2id, id2label)
            train_loader = DataLoader(train,batch_size=demo2_config["batch_size"],
                                      shuffle=True,collate_fn=dataset.make_collate_fn(tokenizer,max_length=demo2_config["max_length"]))
            dev = dataset.Mydata(data_chose["dev_root"],label2id, id2label)
            dev_loader = DataLoader(dev,batch_size=demo2_config["batch_size"],
                                    shuffle=False,collate_fn=dataset.make_collate_fn(tokenizer,max_length=demo2_config["max_length"]))

            model = model_m.Mymodel(model_chose["model_root"],num_labels=len(labels),
                                    dropout=demo2_config["dropout"])
            model.to(device)
            optim = torch.optim.AdamW(model.parameters(),lr=demo2_config["learning_rate"])
            loss_fn = torch.nn.CrossEntropyLoss()
            best_f1 = 0.0
            patience = demo2_config["patience"]
            early_stop = 0.0
            epoch = demo2_config["epoch"]
            model_save_path = os.path.join(save_dir, f"{data_chose['name']}_{model_chose['name']}_best.pt")
            for i in range(epoch):
                print(f"===== Epoch {i + 1}/{epoch} =====")
                avg_loss = train_t(model,train_loader, device, loss_fn, optim)
                acc,p,r,f = dev_d(device,dev_loader,model,id2label)
                writer.add_scalar('aver/epoch', avg_loss, i + 1)
                writer.add_scalar('acc/epoch', acc, i + 1)
                writer.add_scalar('dev/p_macro', p, i + 1)
                writer.add_scalar('dev/r_macro', r, i + 1)
                writer.add_scalar('dev/f1_macro', f, i + 1)
                print(f"train_loss:{avg_loss:.4f} | dev_acc:{acc:.4f} | p_macro:{p:.4f} | r_macro:{r:.4f} | f1_macro:{f:.4f}")
                if f > best_f1:
                    best_f1 = f
                    early_stop = 0
                    torch.save(model.state_dict(), model_save_path)
                    print(f"✅ 最优更新！保存模型到 {model_save_path}, best_f1:{best_f1:.4f}")
                else:
                    early_stop += 1
                    if early_stop >= patience:
                        print(f"触发早停patience={patience},结束该组实验")
                        break

            writer.close()