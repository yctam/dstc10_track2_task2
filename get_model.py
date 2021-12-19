import sys
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel

print(transformers.__file__)
print(transformers.__version__)

model_name_or_path = sys.argv[1]

config = AutoConfig.from_pretrained(model_name_or_path)
#tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path, config=config)

results = tokenizer.tokenize("This is a Test")

print(results)

config.save_pretrained("./{}-local".format(model_name_or_path))
tokenizer.save_pretrained("./{}-local".format(model_name_or_path))
model.save_pretrained("./{}-local".format(model_name_or_path))
