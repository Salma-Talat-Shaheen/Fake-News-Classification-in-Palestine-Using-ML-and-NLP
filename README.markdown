# Fake News Classification in Palestine Project
---
<p align="center">
  <img src="https://github.com/Salma-Talat-Shaheen/Fake-News-Classification-in-Palestine-Using-ML-and-NLP/blob/main/flowCharts/NLP_GroupC_FinalTask%20.png" width="600" height="400" />
</p>

---
## Overview
This project aims to detect fake news in Palestine using Natural Language Processing (NLP) techniques. It processes Arabic news articles, extracts features, and trains machine learning models to classify news as real or fake. The best-performing model, XGBoost with TF-IDF features (text, platform, and numerical), achieved an accuracy of 95.05%, showing strong potential for combating misinformation in Arabic media.

## Project Structure
<p align="center">
  <img src="https://github.com/Salma-Talat-Shaheen/Fake-News-Classification-in-Palestine-Using-ML-and-NLP/blob/main/flowCharts/modelFlowCharts.jpg" width="600" height="400" />
</p>
The project is implemented in a Jupyter Notebook (`fakeNewsClassification.ipynb`) with the following key sections:

1. **Setup and Data Loading**:
   - Loads required libraries (listed in `requirements.txt`).
   - Loads a dataset of Arabic news articles with columns: `Id`, `date`, `platform`, `title`, `News content`, and `Label` (Real/Fake).

2. **Data Preprocessing**:
<p align="center">
  <img src="https://github.com/Salma-Talat-Shaheen/Fake-News-Classification-in-Palestine-Using-ML-and-NLP/blob/main/flowCharts/Preprocessing.jpg" width="600" height="400" />
</p>
   - Cleans and preprocesses Arabic text using tokenization, stopword removal, and normalization with Camel Tools and Farasa stemmer.
   - Handles platform encoding and numerical feature extraction.

3. **Feature Extraction**:
<p align="center">
  <img src="https://github.com/Salma-Talat-Shaheen/Fake-News-Classification-in-Palestine-Using-ML-and-NLP/blob/main/flowCharts/featureExtraction.jpg" width="600" height="400" />
</p>
   - Uses TF-IDF for text representation.
   - Implements Word2Vec (CBOW and Skip-gram) for word embeddings.
   - Experiments with AraBERT embeddings for advanced text representation.
   - Combines text features with platform and numerical features.

4. **Model Training**:
   - Trains multiple models: Logistic Regression, SVM, Naive Bayes, Random Forest, and XGBoost.
   - Evaluates models using accuracy and F1-scores for both classes (Real and Fake).

5. **Results**:
   - The best model (XGBoost with TF-IDF on text, platform, and numerical features) achieved 95.05% accuracy.
   - Detailed performance comparison is provided in a table (see notebook section 5).

6. **Prediction on New News**:
   - Includes a `predict_news` function to classify new articles using the trained XGBoost model with Word2Vec Skip-gram embeddings.
   - Tests the model on sample news articles from provided links, correctly identifying fake and real news.

7. **Conclusion and Future Improvements**:
   - Recommends the XGBoost model with TF-IDF for deployment.
   - Suggests future work, including deep learning models (e.g., LSTMs, Transformers), sentiment analysis, and collecting more diverse data.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Salma-Talat-Shaheen/Fake-News-Classification-in-Palestine-Using-ML-and-NLP.git
   cd Fake-News-Classification-in-Palestine-Using-ML-and-NLP
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the dataset (`merged_cleaned.xlsx`) in the project directory or a mounted Google Drive folder if using Colab.

## Usage
1. Open the `fakeNewsClassification.ipynb` notebook in Jupyter or Google Colab.
2. Ensure the dataset is accessible (place `real_fake_news.csv` in the appropriate directory or mount Google Drive in Colab).
3. Run all cells sequentially to preprocess data, train models, and test predictions.
4. Use the `predict_news` function to classify new articles by providing text and platform inputs. Example:
   ```python
     sample_text = """
   مسؤولة إغاثة: وضع مستشفيات قطاع غزة كارثي المتحدثة باسم الاتحاد الدولي لجمعيات الهلال والصليب الأحمر للجزيرة: وضع مستشفيات قطاع غزة كارثي. نقص كبير بعدد سيارات الإسعاف العاملة في غزة. انعدام الوقود يعني غياب الحياة تماما بغزة. انعدام الوقود بمستشفيات غزة يعني انعدام فرص الحياة. فقدنا العديد من موظفينا بسبب القصف على غزة."""                                                                                                            
   sample_platform = "الجزيرة"
   print(predict_news(sample_text, sample_platform, models['XGBoost'], w2v_skipgram_model, encoder, num_features_cols, convert_eng_arb))


## Sample Predictions
The notebook tests the model on three sample news articles:
- **Sample 1**: A fake news article about decorating the Indonesian presidential palace with a Palestinian flag (classified as Fake).
- **Sample 2**: A fake news article about protests against Iraqi oil exports to Jordan (classified as Fake).
- **Sample 3**: A real news article about the catastrophic situation in Gaza hospitals (classified as Real).

## Results
The best model (XGBoost with TF-IDF on text, platform, and numerical features) achieved:
- **Accuracy**: 95.05%
- **F1-Score (Fake)**: 0.91
- **F1-Score (Real)**: 0.97

## Future Improvements
- Experiment with deep learning models like LSTMs or Transformers.
- Incorporate sentiment analysis as an additional feature.
- Expand the dataset for better model generalization.

## Authors
- Salma Shaheen
- Rasha Abu Rkab

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on GitHub to suggest improvements or report bugs.

## Contact
For questions or feedback, please contact the authors via GitHub issues.
