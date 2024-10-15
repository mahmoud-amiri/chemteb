<h1 align="center">Chemical Text Embedding Benchmark</h1>

The Chemical Text Embedding Benchmark (ChemTEB) is a comprehensive evaluation framework designed to assess the performance of various text embedding models in the domain of chemical sciences. ChemTEB addresses the unique challenges of processing chemical literature and data, where general-purpose NLP models often fall short. The benchmark provides a wide variery of datasets and tasks tailored to chemical text. ChemTEB facilitates the development of more efficient and domain-specific NLP models, bridging the gap between state-of-the-art models and real-world applications in chemistry.

## Example Usage
To reproduce the results, first clone the repository, and then run the `chem_eval.py` script. The results will be saved in the `results` directory.
 
```bash
git clone --branch standalone --single-branch https://github.com/basf/chemteb.git
cd chemteb
python chem_eval.py
```
To properly evaluate tasks using the proprietary models, you should configure AWS authentication via the CLI by running `aws configure`. Additionally, make sure to set the `OPENAI_API_KEY` environment variable with your OpenAI API key.
