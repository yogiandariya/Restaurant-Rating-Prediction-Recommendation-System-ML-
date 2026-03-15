import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses the restaurant dataset for recommendations.
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Keep only relevant columns for recommendations
    relevant_cols = ['Restaurant Name', 'City', 'Cuisines', 'Average Cost for two', 
                     'Price range', 'Aggregate rating', 'Votes', 'Currency']
    df = df[relevant_cols]
    
    # Handle missing values
    print("Handling missing values...")
    df = df.dropna(subset=['Restaurant Name', 'City', 'Cuisines', 'Aggregate rating'])
    
    # Fill missing price range with median
    df['Price range'] = df['Price range'].fillna(df['Price range'].median())
    
    print(f"Dataset loaded with {len(df)} restaurants")
    return df

def recommend_restaurants(user_preferences, df, top_n=10):
    """
    Recommends restaurants based on user preferences using content-based filtering.
    
    Parameters:
    - user_preferences: dict with keys 'cuisine', 'price_range', 'city', 'min_rating'
    - df: preprocessed dataframe
    - top_n: number of recommendations to return
    
    Returns:
    - DataFrame with top N recommended restaurants
    """
    print("\n" + "="*60)
    print("RESTAURANT RECOMMENDATION SYSTEM")
    print("="*60)
    print(f"\nUser Preferences:")
    print(f"  Cuisine: {user_preferences.get('cuisine', 'Any')}")
    print(f"  Price Range: {user_preferences.get('price_range', 'Any')}")
    print(f"  City: {user_preferences.get('city', 'Any')}")
    print(f"  Minimum Rating: {user_preferences.get('min_rating', 0)}")
    print("-"*60)
    
    # Start with all restaurants
    filtered_df = df.copy()
    
    # Filter by cuisine (if specified)
    if 'cuisine' in user_preferences and user_preferences['cuisine']:
        cuisine = user_preferences['cuisine']
        filtered_df = filtered_df[filtered_df['Cuisines'].str.contains(cuisine, case=False, na=False)]
        print(f"After cuisine filter ({cuisine}): {len(filtered_df)} restaurants")
    
    # Filter by city (if specified)
    if 'city' in user_preferences and user_preferences['city']:
        city = user_preferences['city']
        filtered_df = filtered_df[filtered_df['City'].str.contains(city, case=False, na=False)]
        print(f"After city filter ({city}): {len(filtered_df)} restaurants")
    
    # Filter by price range (±1 tolerance if specified)
    if 'price_range' in user_preferences and user_preferences['price_range']:
        price = user_preferences['price_range']
        filtered_df = filtered_df[
            (filtered_df['Price range'] >= price - 1) & 
            (filtered_df['Price range'] <= price + 1)
        ]
        print(f"After price range filter ({price}±1): {len(filtered_df)} restaurants")
    
    # Filter by minimum rating
    if 'min_rating' in user_preferences and user_preferences['min_rating']:
        min_rating = user_preferences['min_rating']
        filtered_df = filtered_df[filtered_df['Aggregate rating'] >= min_rating]
        print(f"After rating filter (≥{min_rating}): {len(filtered_df)} restaurants")
    
    # Sort by rating (descending) and votes (descending) for popularity
    filtered_df = filtered_df.sort_values(
        by=['Aggregate rating', 'Votes'], 
        ascending=[False, False]
    )
    
    # Return top N recommendations
    recommendations = filtered_df.head(top_n)
    
    print(f"\n{'='*60}")
    print(f"TOP {min(top_n, len(recommendations))} RECOMMENDATIONS:")
    print(f"{'='*60}\n")
    
    if len(recommendations) == 0:
        print("❌ No restaurants found matching your criteria.")
        print("Try relaxing some filters (e.g., remove cuisine or increase price range).\n")
    else:
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"{idx}. {row['Restaurant Name']}")
            print(f"   📍 Location: {row['City']}")
            print(f"   🍽️  Cuisine: {row['Cuisines']}")
            print(f"   ⭐ Rating: {row['Aggregate rating']} ({row['Votes']} votes)")
            print(f"   💰 Price Range: {int(row['Price range'])} | Avg Cost: {row['Average Cost for two']} {row['Currency']}")
            print()
    
    return recommendations

def test_recommendation_system(df):
    """
    Tests the recommendation system with sample user preferences.
    """
    print("\n" + "="*60)
    print("TESTING RECOMMENDATION SYSTEM")
    print("="*60 + "\n")
    
    # Test Case 1: Italian cuisine, mid-range price, specific city
    print("\n🧪 TEST CASE 1: Italian Food Lover in New Delhi")
    test_prefs_1 = {
        'cuisine': 'Italian',
        'price_range': 3,
        'city': 'New Delhi',
        'min_rating': 4.0
    }
    recommend_restaurants(test_prefs_1, df, top_n=5)
    
    # Test Case 2: Chinese cuisine, budget-friendly
    print("\n🧪 TEST CASE 2: Budget Chinese Food")
    test_prefs_2 = {
        'cuisine': 'Chinese',
        'price_range': 2,
        'min_rating': 3.5
    }
    recommend_restaurants(test_prefs_2, df, top_n=5)
    
    # Test Case 3: High-rated restaurants in specific city
    print("\n🧪 TEST CASE 3: Top-Rated Restaurants in Mumbai")
    test_prefs_3 = {
        'city': 'Mumbai',
        'min_rating': 4.5
    }
    recommend_restaurants(test_prefs_3, df, top_n=5)
    
    # Test Case 4: Specific cuisine with no city filter
    print("\n🧪 TEST CASE 4: Best North Indian Restaurants")
    test_prefs_4 = {
        'cuisine': 'North Indian',
        'min_rating': 4.0,
        'price_range': 3
    }
    recommend_restaurants(test_prefs_4, df, top_n=5)

if __name__ == "__main__":
    dataset_path = 'c:/Users/drash/OneDrive/Desktop/project/Cognifyz Project/Dataset .csv'
    
    # Load and preprocess data
    df = load_and_preprocess_data(dataset_path)
    
    # Test the recommendation system
    test_recommendation_system(df)
    
    print("\n" + "="*60)
    print("✅ RECOMMENDATION SYSTEM TESTING COMPLETE")
    print("="*60 + "\n")
