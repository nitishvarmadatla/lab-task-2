Project Overview
This project involves a detailed Exploratory Data Analysis (EDA) and Machine Learning Classification study on the Netflix content catalog. The primary objective is to move beyond anecdotal understanding to provide data-driven recommendations for optimizing content strategy, driving subscriber growth, and enhancing retention.

Problem Statement & Business Objective
Category	Description
Problem Statement	To analyze the vast Netflix content catalog to identify key trends, geographical gaps, and characteristic differences between content types. This quantifiable understanding will inform the content acquisition and original production strategy.
Business Objective	To maximize subscriber growth and minimize churn by developing a strategic, data-informed content investment plan. This plan focuses on prioritizing content types, genres, and geographic origins that best align with global audience demand.

Export to Sheets
💾 Dataset & Methodology
Data Source
The dataset used is the Netflix Movies and TV Shows Dataset (sourced from Kaggle), covering content available on the platform as of 2021.

Data Wrangling & Feature Engineering
Data preparation was crucial due to significant missing values. Key steps included:

Missing Data Imputation: Handled missing data in highly sparse columns like director (approx. 30% missing), cast, and country by using the placeholder 'Unknown' to preserve data integrity.

Feature Creation: Engineered new analytic features: year_added, main_country (primary country of origin), and standardized duration metrics (duration_int and duration_type).

Exploratory Data Analysis (EDA)
The analysis was structured using the UBM Rule (Univariate, Bivariate, and Multivariate Analysis) and included over 20 visualizations to reveal trends:

Content Composition: Analyzing the Movies vs. TV Shows distribution.

Temporal Trends: Charting the acceleration of content acquisition and historical release year trends.

Geographical Focus: Identifying the dominant content production regions and areas lacking representation.

Machine Learning Model
A Logistic Regression model was built and evaluated for a Binary Classification task, predicting the content type (Movie or TV Show) based on derived characteristics (e.g., country, duration, rating).

💡 Key Findings
Content Mix Dominance: The catalog is heavily skewed toward Movies (approximately 70%), confirming that historical investment favored feature films over long-form series.

Acquisition Acceleration: Content acquisition showed a massive, exponential growth starting around 2016, indicating a strategic push to rapidly scale the library.

Geographical Imbalance: Content is overwhelmingly dominated by the United States, highlighting a potential 'localization gap' in catering to high-growth international markets.

Rating Concentration: The majority of content falls into mature audience ratings (TV-MA and TV-14).

⚙️ Technical Stack
Language: Python

Environment: Google Colab / Jupyter Notebook

Core Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn (for modeling and preprocessing)

