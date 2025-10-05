# 🎬 Netflix Content Clustering & Strategic Analysis (EDA Capstone)

## Project Overview

This project is an **Exploratory Data Analysis (EDA)** focused on preparing and analyzing the **Netflix Content Catalog (Clustering Dataset)**. The primary goal is to identify and characterize natural groupings (**clusters**) within the content library based on features like genre, duration, country, and rating. The insights derived are used to inform a data-driven content strategy.

## Problem Statement & Business Objective

| Category | Description |
| :--- | :--- |
| **Problem Statement** | To analyze the 7,787 titles in the Netflix catalog to identify and define **natural clusters and segments** of content. This classification is essential for targeted content acquisition, marketing to specific user segments, and reducing churn. |
| **Business Objective** | To **optimize content acquisition and marketing efficiency** by leveraging EDA and clustering to group titles into distinct, actionable segments. This enables the client to identify market saturation points and under-represented content types. |

---

## 💾 Dataset & Methodology

### Data Source
The data used is the **Netflix Movies and TV Shows CLUSTERING Dataset** (7,787 entries), a previous snapshot of the platform's content catalog.

### Data Wrangling & Feature Engineering
Data preparation was crucial, specifically for clustering models which require purely numerical inputs:

1.  **Missing Data Imputation:** Handled significant missing data in **`director`** (2,389 values) and **`country`** by using the placeholder **'Unknown'** to preserve data integrity.
2.  **Feature Standardization:** The non-numeric `type` and `duration` fields were standardized:
    * **`is_movie`** (Binary: 1/0)
    * **`duration_int`** (Numerical value of minutes/seasons)
    * **`main_country`** (Primary country extracted)
    * **`primary_genre`** (First genre extracted for modeling simplicity)

### Key EDA Findings

| Finding | Insight | Risk/Opportunity |
| :--- | :--- | :--- |
| **Content Mix** | Catalog is **68.7% Movies** vs. 31.3% TV Shows. | **Risk:** High churn among TV series enthusiasts. |
| **Acquisition Trend** | Exponential growth peaked in **2019-2020**. | **Opportunity:** Shift from volume to quality and strategic gap filling. |
| **Geographical Imbalance** | **United States** is overwhelmingly dominant. | **Risk:** Creates a **'localization gap'** in global markets. |
| **Metadata Quality** | Growing trend of acquisitions with **'Unknown' country** origins. | **Risk:** Inefficient content spending and poor regional targeting. |

---

## 💡 Machine Learning & Conclusion

The project prepared data for a clustering model using features like `duration_int`, `year_added`, `main_country`, and encoded genre types. The EDA confirmed that the **`duration_int`** feature is the most predictive single variable, clearly separating Movies (high minutes) from TV Shows (low seasons).

### Final Recommendations

1.  **Strategic TV Investment:** Prioritize the acquisition/production of **multi-season TV series** to address the 'completion fatigue' risk and balance the content mix.
2.  **Geographic Diversification:** Allocate resources specifically to **under-represented markets** (e.g., Latin America, Germany) to close the 'localization gap'.
3.  **Mandate Metadata Quality:** Implement stricter content ingestion controls to eliminate the acquisition of titles with 'Unknown' origins, ensuring all spending is trackable and targetable.

## ⚙️ Technical Stack

* **Language:** Python
* **Environment:** Google Colab / Jupyter Notebook
* **Core Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---
*Created by: \[Your Name]*
