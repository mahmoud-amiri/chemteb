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

## Tasks
Below is an overview of the tasks in ChemTEB:
|Name|Type|Languages|Category|#Sentences|Median Length (Tokens)|
|----|----|---------|--------|----------|----------------------|
| [WikipediaEasy10Classification](https://huggingface.co/datasets/BASF-AI/WikipediaEasy10Classification) | Classification  | eng | s2s | 2105 | 178 |
| [WikipediaEasy5Classification](https://huggingface.co/datasets/BASF-AI/WikipediaEasy5Classification) | Classification  | eng | s2s | 1163 | 172 |
| [WikipediaEasy2GeneExpressionVsMetallurgyClassification](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2GeneExpressionVsMetallurgyClassification) | Classification  | eng | s2s | 5741 | 175 |
| [WikipediaEasy2GreenhouseVsEnantiopureClassification](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2GreenhouseVsEnantiopureClassification) | Classification  | eng | s2s | 1136 | 140 |
| [WikipediaEasy2SolidStateVsColloidalClassification](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2SolidStateVsColloidalClassification) | Classification  | eng | s2s | 2216 | 151 |
| [WikipediaEasy2SpecialClassification](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2SpecialClassification) | Classification  | eng | s2s | 1312 | 133 |
| [WikipediaEZ2Classification](https://huggingface.co/datasets/BASF-AI/WikipediaEZ2Classification) | Classification  | eng | s2s | 58921 | 164 |
| [WikipediaEZ10Classification](https://huggingface.co/datasets/BASF-AI/WikipediaEZ10Classification) | Classification  | eng | s2s | 43146 | 165 |
| [WikipediaMedium5Classification](https://huggingface.co/datasets/BASF-AI/WikipediaMedium5Classification) | Classification  | eng | s2s | 617 | 137 |
| [WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification) | Classification  | eng | s2s | 1451 | 175 |
| [WikipediaMedium2BioluminescenceVsNeurochemistryClassification](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2BioluminescenceVsNeurochemistryClassification) | Classification  | eng | s2s | 486 | 158 |
| [WikipediaMedium2ComputationalVsSpectroscopistsClassification](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2ComputationalVsSpectroscopistsClassification) | Classification  | eng | s2s | 1101 | 155 |
| [WikipediaHard2BioluminescenceVsLuminescenceClassification](https://huggingface.co/datasets/BASF-AI/WikipediaHard2BioluminescenceVsLuminescenceClassification) | Classification  | eng | s2s | 410 | 149 |
| [WikipediaHard2SaltsVsSemiconductorMaterialsClassification](https://huggingface.co/datasets/BASF-AI/WikipediaHard2SaltsVsSemiconductorMaterialsClassification) | Classification  | eng | s2s | 491 | 141 |
| [WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification](https://huggingface.co/datasets/BASF-AI/WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification) | Classification  | eng | s2s | 417 | 209 |
| [SDSGlovesClassification](https://huggingface.co/datasets/BASF-AI/SDSGlovesClassification) | Classification  | eng | s2s | 8000 | 1071 |
| [SDSEyeProtectionClassification](https://huggingface.co/datasets/BASF-AI/SDSEyeProtectionClassification) | Classification  | eng | s2s | 8000 | 1060 |
| [CoconutSMILES2FormulaBM](https://huggingface.co/datasets/BASF-AI/CoconutSMILES2FormulaBM) | BitextMining  | eng, smiles | s2s | 8000 | 11 |
| [PubChemSMILESISoTitleBM](https://huggingface.co/datasets/BASF-AI/PubChemSMILESISoTitleBM) | BitextMining  | eng, smiles | s2s | 14140 | 22 |
| [PubChemSMILESISoDescBM](https://huggingface.co/datasets/BASF-AI/PubChemSMILESISoDescBM) | BitextMining  | eng, smiles | s2p | 14140 | 45 |
| [PubChemSMILESCanonTitleBM](https://huggingface.co/datasets/BASF-AI/PubChemSMILESCanonTitleBM) | BitextMining  | eng, smiles | s2s | 30914 | 12 |
| [PubChemSMILESCanonDescBM](https://huggingface.co/datasets/BASF-AI/PubChemSMILESCanonDescBM) | BitextMining  | eng, smiles | s2p | 30914 | 24 |
| [ChemHotpotQARetrieval](https://huggingface.co/datasets/BASF-AI/ChemHotpotQARetrieval) | Retrieval  | eng | s2s | 10275 | 71 |
| [ChemNQRetrieval](https://huggingface.co/datasets/BASF-AI/ChemNQRetrieval) | Retrieval  | eng | s2s | 22960 | 81 |
| [CoconutRetrieval](https://huggingface.co/datasets/BASF-AI/CoconutRetrieval) | Retrieval  | eng, smiles | s2s | 24280 | 41 |
| [WikipediaMedium5Clustering](https://huggingface.co/datasets/BASF-AI/WikipediaMedium5Clustering) | Clustering  | eng | s2s | 617 | 137 |
| [WikipediaEasy10Clustering](https://huggingface.co/datasets/BASF-AI/WikipediaEasy10Clustering) | Clustering  | eng | s2s | 2105 | 178 |
| [WikipediaAIParagraphsParaphrasePC](https://huggingface.co/datasets/BASF-AI/WikipediaAIParagraphsParaphrasePC) | PairClassification | eng | p2p | 5408 | 104 |
| [CoconutSMILES2FormulaPC](https://huggingface.co/datasets/BASF-AI/CoconutSMILES2FormulaPC) | PairClassification | eng, smiles | s2s | 8000 | 11 |
| [PubChemAISentenceParaphrasePC](https://huggingface.co/datasets/BASF-AI/PubChemAISentenceParaphrasePC) | PairClassification | eng | s2s | 4096 | 20 |
| [PubChemSMILESCanonTitlePC](https://huggingface.co/datasets/BASF-AI/PubChemSMILESCanonTitlePC) | PairClassification | eng, smiles | s2s | 4096 | 16 |
| [PubChemSMILESCanonDescPC](https://huggingface.co/datasets/BASF-AI/PubChemSMILESCanonDescPC) | PairClassification | eng, smiles | s2p | 4096 | 23 |
| [PubChemSynonymPC](https://huggingface.co/datasets/BASF-AI/PubChemSynonymPC)| PairClassification | eng | s2s | 4096 | 8
| [PubChemWikiParagraphsPC](https://huggingface.co/datasets/BASF-AI/PubChemWikiParagraphsPC)| PairClassification | eng | p2p | 4096 | 66 |
| [PubChemSMILESIsoTitlePC](https://huggingface.co/datasets/BASF-AI/PubChemSMILESIsoTitlePC)| PairClassification  | eng, smiles | s2s | 4096 | 35 |
| [PubChemSMILESIsoDescPC](https://huggingface.co/datasets/BASF-AI/PubChemSMILESIsoDescPC)| PairClassification  | eng, smiles | s2p | 4096 | 48 |