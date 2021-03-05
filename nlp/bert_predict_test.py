import sys
sys.path.insert(0, "../")
import csv_handler as csv_handler
from nlp.bert_predict import BertModel
from nlp.logger import log_to_csv
import transform as transform

input_dir = "../../tagging/data/SUGG"
log_path = "./log.txt"
train_dataset = csv_handler.csv_readlines(input_dir + "/train.csv")
dev_dataset = csv_handler.csv_readlines(input_dir + "/dev.csv")

#model = BertModel("bert-base-uncased")
#model.train(train_dataset, num_epoch = 3)

model = BertModel("./output/model")
pred = model.predict(dev_dataset)
pred = transform.map_func(pred, lambda row : row[2])
ground = transform.map_func(dev_dataset, lambda row : int(row[2]))
log_to_csv(ground, pred, log_path)

#model.checkpoint("./output/model")
