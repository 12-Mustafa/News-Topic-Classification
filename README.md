# News Topic Classification (AG News): TF-IDF Baselines vs. Fine-Tuned BERT

Classifies news articles into four topics (**World, Sports, Business, Sci/Tech**) using the title and description.
Three classical models on TF-IDF features are compared with a fine-tuned BERT model on the full AG News dataset
(120,000 training and 7,600 test articles).

**Result:** fine-tuned BERT reached **93.2% test accuracy**, against **90.7%** for the best TF-IDF model (Linear SVM).
That cuts the error rate from 9.3% to 6.8%, about **27% fewer misclassified articles**.

| Model | Features | Accuracy | Weighted F1 |
|---|---|---|---|
| Multinomial Naive Bayes | TF-IDF (5,000 terms) | 89.1% | 0.890 |
| Logistic Regression | TF-IDF (5,000 terms) | 90.6% | 0.906 |
| Linear SVM | TF-IDF (5,000 terms) | 90.7% | 0.906 |
| **BERT (`bert-base-uncased`, fine-tuned 1 epoch)** | WordPiece tokens, max 128 | **93.2%** | **0.93** |

All numbers come from the outputs saved in [`M508.ipynb`](M508.ipynb), evaluated on the official 7,600-article test set.

---

## Why this problem

Newsrooms, analysts and companies that monitor markets receive far more articles than people can sort by hand.
An automatic topic classifier lets them route content (for example, Business news to a finance team) and track
trends by category. The task here is a balanced **multi-class text classification** problem: given a headline and
a short description, predict one of four labels.

## Dataset

- **AG News** ([Kaggle mirror](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset); originally from Zhang, Zhao & LeCun, 2015)
- 120,000 training articles and 7,600 test articles, **perfectly balanced** (30,000 per class in training, 1,900 per class in test)
- Columns: `Class Index` (1 = World, 2 = Sports, 3 = Business, 4 = Sci/Tech), `Title`, `Description`
- Titles average **6.8 words**, descriptions **31.1 words**, so title and description are combined into one input text

The CSV files are not included in this repository. Download `train.csv` and `test.csv` from Kaggle and put them
next to the notebook.

## Approach

```
Load data  ->  EDA  ->  Preprocessing  ->  TF-IDF  ->  NB / LogReg / SVM  ->  Evaluation
                              |
                              +-------->  BERT tokenizer  ->  Fine-tune BERT  ->  Evaluation
```

1. **Exploratory analysis:** class balance, title and description lengths, and the most frequent words per class.
   Each class has a clear vocabulary (for example *iraq, minister, government* for World; *software, microsoft,
   internet* for Sci/Tech), while words like *said* and *new* appear everywhere. That supports TF-IDF weighting,
   which down-weights terms common to all classes.
2. **Preprocessing:** lowercasing, removing punctuation and numbers, NLTK stopword removal, WordNet lemmatization,
   and merging title and description into one `clean_text` field.
3. **Classical models:** `TfidfVectorizer(max_features=5000)` fitted on the training set only, then Multinomial
   Naive Bayes, Logistic Regression and Linear SVM from scikit-learn.
4. **Transformer model:** `bert-base-uncased` with a 4-class classification head, fine-tuned with the Hugging Face
   `Trainer` (1 epoch, batch size 8, learning rate 2e-5, weight decay 0.01, max sequence length 128).
5. **Evaluation:** accuracy plus per-class precision, recall and F1 on the held-out test set.

## Results in detail

**Per-class F1 on the test set**

| Class | Naive Bayes | Logistic Regression | Linear SVM | BERT |
|---|---|---|---|---|
| World | 0.89 | 0.91 | 0.91 | 0.94 |
| Sports | 0.96 | 0.96 | 0.96 | 0.98 |
| Business | 0.85 | 0.87 | 0.87 | 0.90 |
| Sci/Tech | 0.86 | 0.88 | 0.88 | 0.91 |

**What the numbers say**

- **Sports is the easiest class** for every model (F1 0.96 to 0.98), because its vocabulary barely overlaps with the others.
- **Business and Sci/Tech are the hardest.** They share words such as *company, inc, microsoft, google*, so a
  headline like *"Google faces new antitrust suit over ad business"* is genuinely ambiguous (Logistic Regression
  labelled it Sci/Tech). BERT adds about +3 F1 points on World, Business and Sci/Tech, but even BERT stays at
  0.90 to 0.91 on Business and Sci/Tech, so this overlap is the main remaining source of error.
- **Logistic Regression and Linear SVM are effectively tied** (90.6% vs 90.7%, a gap of two test articles out of 7,600).
  Either is a sensible choice for a lightweight model.

## Which model to use, and the trade-off

BERT is the more accurate model, but it is not automatically the right one to deploy:

- **For BERT:** it reads words in context rather than as a bag of terms, and it scored higher on every class.
  27% fewer errors matters when thousands of articles are routed per day.
- **For TF-IDF + Linear SVM:** it trains in seconds on a CPU, predicts almost instantly, is easy to explain (you can
  list the words that drive each class), and is only 2.5 points behind. In this run, just evaluating BERT on the 7,600 test
  articles took about 5 minutes on a CPU; the linear models' prediction time was not measured, but is typically
  a small fraction of that.

The practical recommendation is a linear model where speed, cost or explainability matter, and BERT where accuracy
on ambiguous articles is worth the extra compute.

## Limitations and honest notes

- **BERT was trained on the preprocessed text** (stopwords removed, lemmatized). BERT is pre-trained on natural
  sentences, so feeding it raw title + description would likely score higher. This was not tested here.
- **Only one epoch and one random seed.** The 93.2% is a single run, so small differences (such as LogReg vs SVM)
  should not be over-read.
- **The TF-IDF baseline is deliberately simple** (unigrams, 5,000 features). Adding bigrams and a larger vocabulary
  is a well-known way to strengthen this baseline, which would likely narrow BERT's lead.
- **The test set was also passed as BERT's evaluation set.** No hyperparameters were tuned on it, but a separate
  validation split would be cleaner.
- **Out-of-domain text is harder.** On hand-written examples, *"An ice cream truck expands its route to include a
  nearby dog park"* was labelled World, because AG News has no local or lifestyle category.

## How to run

```bash
git clone https://github.com/12-Mustafa/News-Airticle-Detection.git
cd News-Airticle-Detection

pip install pandas numpy matplotlib seaborn nltk scikit-learn torch transformers datasets accelerate

# Download train.csv and test.csv from Kaggle (link above) into this folder, then:
jupyter notebook M508.ipynb
```

The classical models run in a few minutes on any laptop. Fine-tuning BERT on all 120,000 articles is slow on a CPU;
a GPU (for example a free Google Colab GPU) is strongly recommended.

## Next steps

- Fine-tune BERT on raw (unpreprocessed) text and compare with the current result
- Add bigrams to the TF-IDF baseline for a fairer comparison
- Try DistilBERT for a faster, smaller transformer
- Add a confusion matrix to show exactly which classes are confused
- Wrap the best model in a small API or Streamlit demo

## Tech stack

Python, pandas, NLTK, scikit-learn, PyTorch, Hugging Face Transformers, Matplotlib, Seaborn

## References

- Zhang, X., Zhao, J., & LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification.* NeurIPS.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
- Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.* EMNLP: System Demonstrations.
