import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from scipy.sparse import hstack
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
anime_data = pd.read_csv('anime.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Merge data
merged_train_data = pd.merge(train_data, anime_data, on='anime_id')
merged_test_data = pd.merge(test_data, anime_data, on='anime_id')

# Preprocess data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(merged_train_data['name'])
X_test = vectorizer.transform(merged_test_data['name'])

mlb = MultiLabelBinarizer()
merged_train_data['genre'] = merged_train_data['genre'].fillna('')
merged_test_data['genre'] = merged_test_data['genre'].fillna('')
genre_encoded_train = mlb.fit_transform(merged_train_data['genre'].str.split(','))
genre_encoded_test = mlb.transform(merged_test_data['genre'].str.split(','))

# Replace non-numeric values with the median number of episodes
median_episodes = merged_train_data[merged_train_data['episodes'] != 'Unknown']['episodes'].astype(float).median()
merged_train_data['episodes'] = merged_train_data['episodes'].replace('Unknown', median_episodes).astype(float)
merged_test_data['episodes'] = merged_test_data['episodes'].replace('Unknown', median_episodes).astype(float)

# Normalize number of episodes
scaler = StandardScaler()
episodes_scaled_train = scaler.fit_transform(merged_train_data[['episodes']])
episodes_scaled_test = scaler.transform(merged_test_data[['episodes']])

# Combine all features
type_encoded_train = pd.get_dummies(merged_train_data['type'])
type_encoded_test = pd.get_dummies(merged_test_data['type'])
X_train_combined = hstack([X_train, genre_encoded_train, type_encoded_train, episodes_scaled_train])
X_test_combined = hstack([X_test, genre_encoded_test, type_encoded_test, episodes_scaled_test])

# Train the model with combined features
model = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1)
model.fit(X_train_combined, merged_train_data['rating_x'], eval_set=[(X_test_combined, merged_test_data['rating'])], early_stopping_rounds=10, verbose=False)

# Streamlit app
st.title("Anime Recommender System")
st.header("Predict Anime Ratings")

title = st.text_input("Anime Title:")
genres = st.text_input("Genres (comma-separated):")
episodes = st.number_input("Number of Episodes:", min_value=1, step=1)
anime_type = st.selectbox("Type:", options=merged_train_data['type'].unique())

def predict_rating(title, genres, episodes, anime_type):
    title_vector = vectorizer.transform([title])
    genre_vector = mlb.transform([genres.split(',')])
    type_vector = pd.get_dummies(pd.Series([anime_type])).reindex(columns=type_encoded_train.columns, fill_value=0).values
    episodes_vector = scaler.transform([[episodes]])
    combined_vector = hstack([title_vector, genre_vector, type_vector, episodes_vector])
    return model.predict(combined_vector)[0]

if st.button("Predict Rating"):
    rating = predict_rating(title, genres, episodes, anime_type)
    st.write(f"Predicted Rating for '{title}': {rating:.2f}")
