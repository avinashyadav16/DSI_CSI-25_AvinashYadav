# Customer Segmentation Analysis

## 📋 Project Overview

This project focuses on analyzing customer data to identify distinct customer segments based on purchasing behavior and demographics. Using machine learning clustering techniques, I have develop a targeted marketing strategies for effective customer relationship management.

## 🎯 Objectives

- **Analyze customer purchasing patterns** and demographic characteristics
- **Identify distinct customer segments** using unsupervised machine learning
- **Develop targeted marketing strategies** for each segment
- **Provide actionable insights** for business decision-making

## 🗂️ Dataset

**Dataset:** Mall_Customers.csv

**🔗 [Dataset Link](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)**

The dataset contains customer information with the following features:

- **CustomerID**: Unique identifier for each customer
- **Gender**: Customer's gender (Male/Female)
- **Age**: Customer's age
- **Annual Income (k$)**: Customer's annual income in thousands of dollars
- **Spending Score (1-100)**: Score assigned based on customer behavior and spending nature

**Dataset Statistics:**

- Total Customers: 200
- Features: 5
- No missing values

## 🛠️ Methodology

### 1. Data Analysis

- Exploratory Data Analysis (EDA)
- Statistical insights and correlation analysis
- Demographic and behavioral pattern analysis

### 2. Feature Engineering

- Customer scoring system based on Age, Income, and Spending
- Categorical variable encoding
- Data preprocessing and standardization

### 3. Clustering Algorithms

- **K-Means Clustering**: Partitional clustering approach
- **Hierarchical Clustering**: Agglomerative clustering with dendrograms
- **DBSCAN**: Density-based clustering for outlier detection

### 4. Customer Scoring

- Age-based scoring (1-5 scale)
- Income-based scoring (quantile-based)
- Spending score normalization
- Combined customer value score calculation

### 5. Visualization

- Interactive dashboards and comprehensive plots
- PCA visualization for dimensionality reduction
- Cluster comparison and analysis

## 📊 Key Findings

### Customer Segments Identified

1. **High Value Customers**: High income + High spending
2. **High Income Conservative Spenders**: High income + Low spending
3. **Budget-conscious High Spenders**: Low income + High spending
4. **Price-sensitive Customers**: Low spending, price-conscious
5. **Balanced Customers**: Moderate income + Moderate spending

### Clustering Results

- **Optimal Clusters**: Determined using Silhouette Score and Elbow Method
- **Best Performing Algorithm**: K-Means clustering
- **Cluster Validation**: Multiple metrics including Silhouette Score and Calinski-Harabasz Score

## 🎯 Marketing Strategies

### High Value Customers

- Premium product lines and exclusive collections
- VIP membership programs
- Personal shopping assistance
- Luxury brand partnerships

### Conservative Spenders

- Value-based marketing
- Educational content about product benefits
- Investment-focused positioning
- Bundle deals and cashback programs

### Budget-conscious Spenders

- Affordable fashion and trendy items
- Payment plan options
- Seasonal sales and flash deals
- Student discounts

### Price-sensitive Customers

- Budget-friendly product lines
- Clear value propositions
- Bulk purchase options
- Deep discounts and clearance sales

## 🚀 Technical Implementation

### Libraries Used

```python
# Data Manipulation
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Statistical Analysis
from scipy.cluster.hierarchy import dendrogram, linkage
```

### Key Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Multiple Clustering Methods**: K-Means, Hierarchical, and DBSCAN comparison
- **Cluster Validation**: Multiple metrics for optimal cluster selection
- **Business Intelligence**: Actionable insights and marketing recommendations
- **Interactive Visualizations**: Professional dashboards and plots

## 📈 Results Summary

### Cluster Performance Metrics

- **Silhouette Score**: Measures cluster quality and separation
- **Calinski-Harabasz Score**: Evaluates cluster density and separation
- **Inertia/WCSS**: Within-cluster sum of squares for K-Means

### Business Impact

- **Revenue Opportunity**: Identified high-potential customer segments
- **Marketing Efficiency**: Targeted strategies for each customer group
- **Customer Retention**: Focus on high-value customer satisfaction
- **Growth Strategy**: Convert potential customers to active spenders

## 📁 File Structure

```
PROJECT - Customer Segmentation/
│
├── Customer_Segmentation.ipynb    # Main analysis notebook
├── Mall_Customers.csv            # Dataset
└── README.md                     # Project documentation
```

## 🔧 How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly scipy
```

### Execution

1. Clone the repository
2. Navigate to the project directory
3. Open `Customer_Segmentation.ipynb` in Jupyter Notebook or VS Code
4. Run all cells sequentially

## 📊 Visualizations Included

- **Demographic Analysis**: Age, gender, and income distributions
- **Behavioral Patterns**: Spending vs. income correlation analysis
- **Cluster Visualization**: PCA-based cluster representation
- **Comparison Charts**: Multiple clustering algorithm results
- **Business Dashboards**: Marketing strategy visualization
- **Statistical Plots**: Correlation matrices and distribution plots

## 🎯 Business Applications

### Retail Industry

- **Product Placement**: Optimize store layout based on customer segments
- **Inventory Management**: Stock products based on segment preferences
- **Pricing Strategy**: Implement segment-specific pricing models

### Marketing Teams

- **Campaign Targeting**: Develop segment-specific marketing campaigns
- **Channel Selection**: Choose optimal communication channels per segment
- **Budget Allocation**: Allocate marketing budget based on segment value

### Customer Service

- **Service Levels**: Provide differentiated service based on customer value
- **Loyalty Programs**: Design segment-specific loyalty rewards
- **Customer Experience**: Personalize experience based on segment characteristics

## 📈 Future Enhancements

### Technical Improvements

- **Real-time Clustering**: Implement streaming data processing
- **Advanced Algorithms**: Explore Gaussian Mixture Models, Spectral Clustering
- **Feature Engineering**: Include additional behavioral features
- **Model Validation**: Implement cross-validation and stability analysis

### Business Extensions

- **Predictive Analytics**: Customer lifetime value prediction
- **Churn Analysis**: Identify customers at risk of churning
- **Recommendation Systems**: Product recommendations per segment
- **A/B Testing**: Test marketing strategies effectiveness

## 📞 Contact

For questions or collaboration opportunities:

- GitHub: [@avinashyadav16](https://github.com/avinashyadav16)

---

**⭐ Star this repository if you found it helpful!**
