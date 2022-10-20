import transformers

DEVICE = "cuda" #"cuda" ou "cpu"
MAX_LEN = 32 # 512 --> Obviamente o treinamento será mais rápido porque diminuiu o comprimento 
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 2
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "/root/docker_data/model.bin"
TRAINING_FILE = "/root/docker_data/train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

