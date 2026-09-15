import torch
import model_m,train,dataset,config
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
path = [
    {
        "root":"./configs/MSRA_bert-base.json"
    },
    {
        "root":"./configs/MSRA_bert-wwm.json"
    },
    {
        "root":"./configs/weibo_bert-base.json"
    },
    {
        "root":"./configs/weibo_bert-wwm.json"
    }
]
for i in path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_config = config.load_config(i["root"])
    _, labels = dataset.Mydata.read_data(test_config["train_root"])
    labels = sorted(labels)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(test_config["model"])
    model = model_m.Mymodel(
        test_config["model"],
        num_labels=len(labels),
        dropout=test_config["dropout"],
    )
    model.load_state_dict(
        torch.load(test_config["train_model_root"], map_location=device)
    )
    model.to(device)
    test_data = dataset.Mydata(test_config["test_root"],label2id,id2label)
    test_loader = DataLoader(
        test_data,
        batch_size=test_config["batch_size"],
        shuffle=False,
        collate_fn=dataset.make_collate_fn(tokenizer,max_length=test_config["max_length"])
    )
    acc, precision, recall, f1 = train.dev_d(device,test_loader,model,id2label)
    print(
        f"实验:{test_config['experiment_name']} | "
        f"test_acc:{acc:.4f} | "
        f"p_macro:{precision:.4f} | "
        f"r_macro:{recall:.4f} | "
        f"f1_macro:{f1:.4f}"
    )

