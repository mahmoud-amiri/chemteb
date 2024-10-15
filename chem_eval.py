import mteb
from tqdm import tqdm
import time
from chemteb import CHEMICAL_TASKS, CUSTOM_MODELS


if __name__ == "__main__":

    models = {
        "google-bert/bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "allenai/scibert_scivocab_uncased": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
        "intfloat/e5-small": "e272f3049e853b47cb5ca3952268c6662abda68f",
        "intfloat/e5-base": "b533fe4636f4a2507c08ddab40644d20b0006d6a",
        "intfloat/e5-large": "4dc6d853a804b9c8886ede6dda8a073b7dc08a81",
        "intfloat/e5-small-v2": "dca8b1a9dae0d4575df2bf423a5edb485a431236",
        "intfloat/e5-base-v2": "1c644c92ad3ba1efdad3f1451a637716616a20e8",
        "intfloat/e5-large-v2": "b322e09026e4ea05f42beadf4d661fb4e101d311",
        "intfloat/multilingual-e5-small": "fd1525a9fd15316a2d503bf26ab031a61d056e98",
        "intfloat/multilingual-e5-base": "d13f1b27baf31030b7fd040960d60d909913633f",
        "intfloat/multilingual-e5-large": "ab10c1a7f42e74530fe7ae5be82e6d4f11a719eb",
        "nomic-ai/nomic-embed-text-v1": "0759316f275aa0cb93a5b830973843ca66babcf5",
        "nomic-ai/nomic-embed-text-v1.5": "b0753ae76394dd36bcfb912a46018088bca48be0",
        "recobo/chemical-bert-uncased": "498698d28fcf7ce5954852a0444c864bdf232b64",
        "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
        "BAAI/bge-small-en": "2275a7bdee235e9b4f01fa73aa60d3311983cfea",
        "BAAI/bge-base-en": "b737bf5dcc6ee8bdc530531266b4804a5d77b5d8",
        "BAAI/bge-large-en": "abe7d9d814b775ca171121fb03f394dc42974275",
        "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "BAAI/bge-large-en-v1.5": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "all-mpnet-base-v2": "84f2bcc00d77236f9e89c8a360a00fb1139bf47d",
        "multi-qa-mpnet-base-dot-v1": "3af7c6da5b3e1bea796ef6c97fe237538cbe6e7f",
        "all-MiniLM-L12-v2": "a05860a77cef7b37e0048a7864658139bc18a854",
        "all-MiniLM-L6-v2": "8b3219a92973c328a8e22fadcfa821b5dc75636a",
        "m3rg-iitd/matscibert": "ced9d8f5f208712c4a90f98a246fe32155b29995",
        "text-embedding-ada-002": "1",
        "text-embedding-3-small": "1",
        "text-embedding-3-large": "1",
        **{
            (k, v.revision if v.revision is not None else "no_revision_available")
            for k, v in CUSTOM_MODELS.items()
        },
    }

    now = time.time()

    for model in tqdm(models.keys()):
        evaluation = mteb.MTEB(tasks=[obj() for obj in CHEMICAL_TASKS])
        if model in CUSTOM_MODELS:
            model = CUSTOM_MODELS[model].load_model()
            model.mteb_model_meta = CUSTOM_MODELS[model]
        else:
            model = mteb.get_model(model)
        evaluation.run(model)

    elapsed = time.time() - now

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")
