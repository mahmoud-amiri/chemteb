from chemteb.models import amazon_models, cohere_bedrock_models, nomic_bert_models
from mteb.model_meta import ModelMeta


model_modules = [amazon_models, cohere_bedrock_models, nomic_bert_models]

CUSTOM_MODELS = {}

for module in model_modules:
    for mdl in vars(module).values():
        if isinstance(mdl, ModelMeta):
            CUSTOM_MODELS[mdl.name] = mdl
