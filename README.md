# 📊 Sampling Techniques for Imbalanced Datasets

A demonstration of various under and over-sampling techniques for handling imbalanced class distributions using the KDD Cup 1999 Intrusion Detection System dataset.

## 📝 Description

This repository provides a comprehensive implementation of sampling techniques to address the class imbalance problem in machine learning. Using the KDD Cup 1999 IDS dataset as an example, it demonstrates how different sampling methods can improve classification performance when dealing with highly skewed class distributions.

The implementation includes visualizations of decision boundaries using both linear and radial basis function (RBF) kernels to illustrate the separability of classes after applying various sampling techniques.

## ✨ Features

- **Data Preprocessing**: Loads, cleans, and prepares the KDD Cup dataset for analysis
- **Class Imbalance Handling**: Implements multiple sampling techniques:
  - Original (no sampling)
  - Random Over-Sampling
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - ADASYN (Adaptive Synthetic Sampling)
  - Borderline SMOTE (both borderline-1 and borderline-2 variants)
  - SVM SMOTE
  - SMOTE-NC (for datasets with categorical features)
- **Visualization**: Generates decision boundary plots to illustrate the effect of sampling on classification
- **Performance Evaluation**: Calculates and compares metrics (recall, precision, F1, accuracy) across different sampling methods
- **Confusion Matrices**: Visualizes classification results for each sampling technique

## 🔧 Prerequisites

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- imbalanced-learn (imblearn)

## 🚀 Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/corticalstack/sampling-techniques.git
   cd sampling-techniques
   ```

2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn imbalanced-learn
   ```

3. Download the KDD Cup 1999 dataset (a 10% subset is included in the repository):
   ```bash
   # If you want the full dataset instead of the included 10% subset
   # wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz
   # gunzip kddcup.data.gz
   ```

4. Create a directory for plots:
   ```bash
   mkdir plots
   ```

## 💻 Usage

Run the main script to execute the sampling techniques demonstration:

```bash
python main.py
```

This will:
1. Load and preprocess the KDD Cup dataset
2. Apply various sampling techniques to balance the class distribution
3. Visualize the decision boundaries for each sampling method
4. Generate confusion matrices and performance metrics
5. Save visualization plots to the `plots` directory

## 📈 Output

The code generates two types of visualizations for each sampling technique:

1. **Decision Boundary Plots**: Shows how different sampling techniques affect the decision boundaries of classifiers
2. **Confusion Matrices**: Displays the classification performance for each sampling method

## 📚 Resources

- [KDD Cup 1999 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤔 FAQ

**Q: Why use sampling techniques for imbalanced datasets?**  
A: In many real-world scenarios (like intrusion detection, fraud detection, medical diagnosis), the class of interest is rare. Standard machine learning algorithms tend to be biased toward the majority class, often ignoring the minority class. Sampling techniques help balance the class distribution, improving the model's ability to detect minority classes.

**Q: Which sampling technique is best?**  
A: There's no one-size-fits-all answer. The performance of sampling techniques depends on the specific dataset and problem. This repository allows you to compare different techniques and choose the one that works best for your specific use case.
