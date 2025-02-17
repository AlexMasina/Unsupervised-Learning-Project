# Anime Recommender System

## Project Overview
This project builds a recommender system for anime titles using collaborative filtering and content-based filtering techniques. The goal is to predict user ratings for anime titles they have not yet seen based on historical preferences.

## Dataset
The dataset is sourced from MyAnimeList and consists of:
- `anime.csv`: Contains metadata about anime titles (e.g., name, genre, type, episodes, rating, and member count).
- `train.csv`: Contains user ratings for anime titles.
- `test.csv`: Contains user-anime pairs for which predictions are required.
- `submission.csv`: Sample submission file showing expected format.

## Methodology
### 1. Exploratory Data Analysis (EDA):
- Visualizations including histograms, box plots, scatter plots, and correlation matrices to understand rating distributions and relationships.
- Analysis of how ratings vary across anime types and popularity.

### 2. Preprocessing & Feature Engineering:
- Handling missing values in anime metadata.
- Encoding categorical features.
- Merging user rating data with anime metadata.

### 3. Model Selection & Training:
- Collaborative filtering using Singular Value Decomposition (SVD) from Surprise library.
- Cross-validation to evaluate model performance using Root Mean Square Error (RMSE).

### 4. Prediction & Submission:
- Using the trained model to predict ratings for user-anime pairs in the test dataset.
- Formatting and exporting results to a submission file.

## Installation & Requirements
To run this project, install the necessary dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn surprise
```
## Running the Project

1. Load and explore the datasets.
2. Perform EDA to understand the data better.
3. Train the recommendation model using Regression models.
4. Generate predictions on the test dataset.
5. Export the predictions in the required format.

## Evaluation Metric

The performance of the recommendation model is evaluated using Root Mean Square Error (RMSE), which measures the standard deviation of residuals between predicted and actual ratings.

## Submission Format

The final output should be a CSV file with two columns:

```csv
ID,rating
1_2011,1.0
2_3011,4.5 
```

Where ID is a concatenation of user_id and anime_id.

## License
This project is based on open-source data and is free to use for research and educational purposes.

## Acknowledgements
MyAnimeList API for providing anime data and user ratings.
Surprise library for collaborative filtering implementation.

## Author
Nthabiseng Moyeni &
Alex Masina

## Contact
For questions or collaborations, reach out via email:
Alex.Masina@outlook.com.
nthaby.thateng@gmail.com