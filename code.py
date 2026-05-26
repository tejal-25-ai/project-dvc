import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer

# this function creates a dataframe with sample reviews
def create_dataframe():
    data = {
        "id" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "review" : ["This is a great product!",
                     "I'm not satisfied with this purchase.", 
                     "Excellent quality and fast shipping.", 
                     "Average product, nothing special.", 
                     "Highly recommended!", 
                     "Disappointing experience.", 
                     "Good value for money.", 
                     "Poor quality, would not recommend.", 
                     "Outstanding customer service!", 
                     "Great overall experience."]
    }
    df = pd.DataFrame(data)
    return df

# to create the data folder if not exists 

import os 

def save_dataframe(df):
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv("data/data.csv", index=False)
    print("Dataframe saved to data/data.csv")
    
# os.path.exists('data') checks if the 'data' directory exists. If it doesn't, os.makedirs('data') creates the directory. Finally, df.to_csv("data/data.csv", index=False) saves the dataframe to a CSV file in the 'data' directory without including the index.

def process_data(k):
    df = pd.read_csv("data/data.csv")

    vectorizer = CountVectorizer(max_features=k)
    vectorized_data = vectorizer.fit_transform(df['review'])
    features_names = vectorizer.get_feature_names_out() # this method is used to get the feature names from the vectorizer. It returns an array of feature names that correspond to the columns in the vectorized data.

    vectorized_df = pd.DataFrame(vectorized_data.toarray(), columns=features_names)
    processed_df = pd.concat([df, vectorized_df], axis=1)

    processed_df.to_csv("data/processed_data.csv", index=False)
    print("Processed dataframe saved to data/processed_data.csv")
    return processed_df

if __name__ == "__main__":
    df = create_dataframe()
    save_dataframe(df)
    processed_df = process_data(k=6)

    print(f"data shape:{df.shape}")
    print(f"processed data shape:{processed_df.shape}")
    print(processed_df.head())