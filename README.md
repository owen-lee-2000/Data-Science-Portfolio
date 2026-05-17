# Graduate Data Science & Analytics Portfolio

This repository contains four core data projects developed during my graduate program. My work here covers setting up database pipelines in the cloud, cleaning and analyzing text data, isolating true cause-and-effect relationships in statistics, and building machine learning models to predict outcomes.

---

## The Projects

### 1. Cloud Data Warehousing & SQL Pipelines
* **File:** `snowflake_forum_data_pipeline.sql`
* **What I did:** Set up a clean, organized data warehouse in Snowflake. I separated the processing power into two independent areas—one just for loading data and one just for running analytics queries—so they wouldn't slow each other down. I built pipelines to automatically bring in raw files from AWS S3 buckets, set up secure permissions so analysts could read the data without accidentally modifying it, and wrote SQL queries to track how long users stay active on the platform.

### 2. Customer Review Text Mining & Sentiment Analysis
* **File:** `nlp_unstructured_text_mining_analytics.py`
* **What I did:** Built a Python script to scan raw customer reviews and pull out useful business insights. I used the VADER text tool to automatically score each review and tag it as Positive, Negative, or Neutral. I then wrote a text-cleaning function to handle formatting issues, strip out punctuation, and drop common, unhelpful words. Finally, I ran an unsupervised topic model (LDA) to look at just the positive reviews and automatically group the words into major themes to see what menu items customers loved most.

### 3. Causal Inference & Market Shock Analysis
* **File:** `causal_inference_econometric_modeling.py`
* **What I did:** Used advanced statistics to prove actual cause-and-effect instead of just looking at random correlations. The first part uses a Difference-in-Differences model to see exactly how much hotel spending jumped in Chicago due to a major concert week, using Columbus as a comparison city to prove the trend was real. The second part looks at taxi ride data to see if customers change their tipping behavior the second a fare crosses a specific $15 threshold, including a check to make sure the data wasn't being manipulated right at the line.

### 4. Predictive Machine Learning with Tree Ensembles
* **File:** `predictive_modeling_classification_script.py`
* **What I did:** Built a machine learning workflow to predict wine quality based on a mix of different features. I split the data into training and testing sets using a fixed seed so the results can be perfectly repeated by anyone running the code. I trained and compared two different models—a Random Forest and a Gradient Boosting classifier—and mapped out exactly which features had the highest impact on the predictions. Finally, I evaluated the performance using confusion matrices and charts to compare their true positive vs. false positive rates.

---

## Core Skills
* **Languages:** Python (Pandas, NumPy, Scikit-Learn, Statsmodels), SQL (Snowflake)
* **Tools & Cloud:** Snowflake Data Platform, AWS S3, Git/GitHub
* **Concepts:** Supervised Learning, Text Mining, Causal Inference, Database Access Control
