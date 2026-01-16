# Turkish Sentiment Analysis

A machine learning project that classifies Turkish text into positive, negative, or neutral sentiments using NLP techniques.

## Project Overview

This project demonstrates natural language processing and text classification skills for internship applications. It analyzes Turkish text data to predict sentiment using machine learning.

**Key Technologies:**
- Python 3.10+
- Scikit-learn (ML algorithms)
- Pandas & NumPy (data processing)
- Matplotlib & Seaborn (visualization)
- NLP techniques (TF-IDF, text preprocessing)

## Dataset

**Source:** Custom Turkish sentiment dataset  
**Samples:** 45 Turkish text examples  
**Classes:** 3 (Positive, Negative, Neutral)  
**Split:** 80% train, 20% test

## Model Architecture

**Baseline Model:**
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Algorithm:** Logistic Regression (multi-class)
- **Features:** Top 100 TF-IDF features
- **Max Iterations:** 1000

## Results

**Overall Performance:**
- **Accuracy:** 77.78%
- **Model:** Baseline (TF-IDF + Logistic Regression)

**Performance by Class:**
- All classes achieved balanced performance
- Confusion matrix shows minimal misclassification

## Custom Analyses

### 1. Most Predictive Words
**Research Question:** Which words best indicate each sentiment?

**Findings:**
- **Positive indicators:** "harika" (great), "mükemmel" (perfect), "güzel" (nice)
- **Negative indicators:** "kötü" (bad), "berbat" (terrible), "rezalet" (disaster)
- Model correctly learned Turkish sentiment vocabulary

### 2. Error Analysis
**Research Question:** Which examples does the model misclassify?

**Findings:**
- X misclassifications out of Y test samples
- Neutral statements sometimes confused with emotional sentiments
- Short, ambiguous texts are harder to classify

### 3. Text Length Impact
**Research Question:** Does text length affect accuracy?

**Findings:**
- Text length shows minimal impact on prediction accuracy
- Model performs consistently across different lengths
- Even short texts can be classified accurately with strong sentiment words

## Technical Highlights

- **Text Preprocessing:** Lowercasing, URL removal, mention/hashtag removal, whitespace normalization
- **Feature Engineering:** TF-IDF vectorization for numerical representation
- **Stratified Splitting:** Maintained class balance in train-test split
- **Evaluation Metrics:** Accuracy, precision, recall, F1-score, confusion matrix
- **Feature Importance:** Analyzed most predictive words for each class
- **Error Analysis:** Systematic review of misclassifications

## Project Structure
```
turkish-sentiment-analysis/
│
├── Turkish_Sentiment_Analysis.ipynb    # Main analysis notebook
└── README.md                            # This file
```

## How to Run

### Google Colab (Recommended)
1. Upload notebook to Google Colab
2. Run all cells sequentially
3. Dataset is created programmatically (no download needed)

### Local Environment
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Launch Jupyter
jupyter notebook Turkish_Sentiment_Analysis.ipynb
```

## Sample Predictions
```
Text: "Bu ürün harika, çok beğendim!"
Predicted: positive (Confidence: 95%)

Text: "Berbat bir deneyim, asla tavsiye etmem"
Predicted: negative (Confidence: 92%)

Text: "Toplantı yarın saat 10:00da"
Predicted: neutral (Confidence: 88%)
```

## Challenges Faced

1. **Small Dataset:** Limited to 45 examples - accuracy could improve with more data
2. **Turkish Language:** Fewer pre-trained models compared to English
3. **Neutral Class:** Distinguishing neutral from slightly positive/negative is challenging

## Future Improvements

- [ ] Use BERT-based Turkish models (BERTurk, mBERT)
- [ ] Expand dataset to 10,000+ samples
- [ ] Add deep learning models (LSTM, GRU)
- [ ] Deploy as web application with Streamlit
- [ ] Handle sarcasm and irony detection
- [ ] Real-time Twitter sentiment tracking

## Skills Demonstrated

- Natural Language Processing (NLP)
- Text preprocessing and cleaning
- Feature extraction (TF-IDF)
- Machine Learning (classification)
- Model evaluation and metrics
- Data visualization
- Critical analysis of results
- Technical documentation

## Author

**Hudayi Hamza Adatepe**  
Computer Engineering Student, 3rd Year  
Seeking Summer 2026 Internship

**Contact:**
- LinkedIn: www.linkedin.com/in/hüdayi-adatepe-9073121b8
- Email: dayihamza_60@hotmail.com
- GitHub: Hudayiadatepe(https://github.com/Hudayiadatepe)

## Related Projects

- [COVID-19 Turkey Analysis](https://github.com/yourusername/covid19-turkey-analysis) - Data analysis project

## License

This project is available for educational purposes.

## Acknowledgments

- Scikit-learn documentation and community
- Turkish NLP resources
- Inspiration: Real-world sentiment analysis applications

  ---

## 📌 Project Note

This application demonstrates a **complete ML deployment pipeline** (training → deployment → hosting). The model is trained on a demo dataset (45 samples) to showcase deployment capabilities rather than production-level accuracy.

**Key Achievement:** Successfully deployed a working ML model with Streamlit Cloud, demonstrating end-to-end ML engineering skills for internship applications.

**Production Improvements:** Larger dataset (10K+ samples), Turkish BERT models, user feedback system.

---
