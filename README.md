# NLP Course BLEnD Evaluation

This project is based on [BLEnD](https://junhomyung.github.io/BLEnD/) (Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages) and adapted for **SemEval 2026 Task 7, Track 1: Short Answer Questions (SAQ)**.

## Overview

BLEnD is a hand-crafted benchmark designed to evaluate LLMs' everyday knowledge across diverse cultures and languages. This repository contains evaluation code adapted for the SemEval 2026 competition.

**Official Links:**
- [BLEnD Website](https://junhomyung.github.io/BLEnD/)
- [BLEnD GitHub](https://github.com/nlee0212/BLEnD)
- [SemEval 2026 Task 7 on CodaBench](https://www.codabench.org/competitions/10281/)

## Project Structure

```
BLEnD-main/
├── data/                           # Dataset files
│   ├── annotations/                # JSON files with annotated answers
│   ├── questions/                  # CSV files with questions
│   └── prompts/                    # CSV files with prompts
├── evaluation/                     # Evaluation code
│   ├── exact_match.py              # Main logic for short answer evaluation
│   ├── evaluation_utils.py        # Data handling utilities
│   ├── evaluate.py                 # Original evaluation script
│   ├── evaluate_semeval.py         # SemEval/CodaBench evaluation script
│   ├── run_semeval.sh              # Bash script for SemEval evaluation
│   └── evaluate.sh                 # Original evaluation script
├── utils.py                        # Common utilities
├── requirements.txt                # Python dependencies
└── README.md                       # Original BLEnD documentation
```

## Evaluation Methodology

### Soft Exact Match (SEM)

The evaluation uses Soft Exact Match methodology:

1. **Matching Logic**: A model response is considered correct if it matches any of the answers provided by human annotators for the same question.

2. **Variation Handling**: The system accounts for different phrasings and formulations of the same answer.

3. **Lemmatization**: Language-specific lemmatization is used for different languages.

4. **Metrics**:
   - **SEM-B** (Soft Exact Match - Binary): Binary metric (correct/incorrect)
   - **SEM-W** (Soft Exact Match - Weighted): Weighted metric (accounts for annotator vote counts)

## Installation

### Basic Requirements

```bash
pip install -r BLEnD-main/requirements.txt
```

### Language-Specific Dependencies

For proper lemmatization across all languages, install additional packages:

```bash
cd BLEnD-main/evaluation
pip install konlpy              # Korean
pip install hausastemmer        # Hausa
pip install nlp-id              # Indonesian
pip install hazm                # Persian
pip install qalsadi             # Arabic
pip install cltk                # Greek
pip install spark-nlp==5.3.3 pyspark==3.3.1  # Spanish, Amharic
pip install jieba               # Chinese

# For Azerbaijani and Assamese
git clone https://github.com/aznlp-disc/stemmer.git
cp stemmer/word.txt ./evaluation
cp stemmer/suffix.txt ./evaluation

git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
```

See `BLEnD-main/README.md` for complete installation instructions.

## Usage

### SemEval 2026 Evaluation

For evaluating predictions in SemEval format:

```bash
cd BLEnD-main/evaluation

# Using Python script
python evaluate_semeval.py \
    --predictions_file predictions.csv \
    --annotations_dir ../data/annotations \
    --results_dir results \
    --id_col ID \
    --response_col response

# Or using bash script
./run_semeval.sh predictions.csv
```

**Input Format:**
- CSV file with columns: `ID` (question ID), `response` (model response)
- Optional: `country` column if processing multiple countries

**Output:**
- `results/evaluation_summary.csv` - Summary of results
- `results/results.json` - Results in JSON format (CodaBench compatible)
- `results/{country}_detailed_results.csv` - Detailed per-question results

### Original BLEnD Evaluation

For evaluating with the original BLEnD format:

```bash
cd BLEnD-main/evaluation
python evaluate.py \
    --model "model-name" \
    --country "US" \
    --language "English" \
    --prompt_no "inst-4" \
    --id_col ID \
    --response_col response \
    --annotation_dir ../data/annotations
```

## Supported Languages

The dataset includes 16 countries/regions with 13 languages:

| Country/Region | Language | Language Code |
|----------------|----------|---------------|
| UK | English | en |
| US | English | en |
| South Korea | Korean | ko |
| North Korea | Korean | ko |
| Algeria | Arabic | ar |
| China | Chinese | zh |
| Indonesia | Indonesian | id |
| Spain | Spanish | es |
| Mexico | Spanish | es |
| Iran | Persian | fa |
| Assam | Assamese | as |
| Greece | Greek | el |
| Ethiopia | Amharic | am |
| Northern Nigeria | Hausa | ha |
| Azerbaijan | Azerbaijani | az |
| West Java | Sundanese | su |

**Note:** SemEval 2026 Task 7 includes 26 languages (extended version).

## Data Format

### Annotations (`data/annotations/*.json`)

```json
{
  "question_id": {
    "question": "Question in local language",
    "en_question": "Question in English",
    "annotations": [
      {
        "answers": ["answer1", "answer2"],
        "en_answers": ["english_answer"],
        "count": 4
      }
    ],
    "idks": {
      "idk": 0,
      "no-answer": 0,
      "not-applicable": 0
    }
  }
}
```

### Questions (`data/questions/*.csv`)

Columns: `ID`, `Topic`, `Source`, `Question`, `Translation`

## License

This project is based on BLEnD. Please refer to the original BLEnD repository for licensing information.

## Citation

If you use this code, please cite the original BLEnD paper:

```bibtex
@article{blend2024,
  title={BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages},
  author={...},
  journal={NeurIPS 2024 Datasets and Benchmarks Track},
  year={2024}
}
```

## Acknowledgments

- Original BLEnD dataset and code: [nlee0212/BLEnD](https://github.com/nlee0212/BLEnD)
- Adapted for SemEval 2026 Task 7, Track 1
