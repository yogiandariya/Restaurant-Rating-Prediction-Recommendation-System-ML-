import pandas as pd
import numpy as np

def load_and_explore_data(filepath):
    """
    Loads the dataset and explores geographical coordinates.
    """
    print("="*70)
    print("LOCATION-BASED ANALYSIS OF RESTAURANTS")
    print("="*70 + "\n")
    
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Keep relevant columns for location analysis
    location_cols = ['Restaurant Name', 'City', 'Locality', 'Latitude', 'Longitude', 
                     'Cuisines', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
    df = df[location_cols]
    
    # Handle missing values
    print("Handling missing values...")
    initial_count = len(df)
    df = df.dropna(subset=['City', 'Latitude', 'Longitude', 'Aggregate rating'])
    print(f"Removed {initial_count - len(df)} rows with missing location data")
    print(f"Total restaurants for analysis: {len(df)}\n")
    
    # Display coordinate ranges
    print("-"*70)
    print("GEOGRAPHICAL COORDINATE RANGES")
    print("-"*70)
    print(f"Latitude range:  {df['Latitude'].min():.4f} to {df['Latitude'].max():.4f}")
    print(f"Longitude range: {df['Longitude'].min():.4f} to {df['Longitude'].max():.4f}")
    print()
    
    return df

def analyze_by_city(df):
    """
    Groups restaurants by city and analyzes concentration.
    """
    print("="*70)
    print("RESTAURANT CONCENTRATION BY CITY")
    print("="*70 + "\n")
    
    # Count restaurants per city
    city_counts = df['City'].value_counts()
    
    print(f"Total unique cities: {len(city_counts)}\n")
    print("Top 15 Cities by Restaurant Count:")
    print("-"*70)
    for idx, (city, count) in enumerate(city_counts.head(15).items(), 1):
        percentage = (count / len(df)) * 100
        print(f"{idx:2d}. {city:25s} - {count:4d} restaurants ({percentage:5.2f}%)")
    print()
    
    return city_counts

def calculate_city_statistics(df):
    """
    Calculates statistics by city: average ratings, cuisines, price ranges.
    """
    print("="*70)
    print("STATISTICS BY CITY")
    print("="*70 + "\n")
    
    # Group by city
    city_stats = df.groupby('City').agg({
        'Aggregate rating': 'mean',
        'Price range': 'mean',
        'Votes': 'sum',
        'Restaurant Name': 'count'
    }).round(2)
    
    city_stats.columns = ['Avg Rating', 'Avg Price Range', 'Total Votes', 'Restaurant Count']
    city_stats = city_stats.sort_values('Restaurant Count', ascending=False)
    
    print("Top 15 Cities - Detailed Statistics:")
    print("-"*70)
    print(f"{'City':<25} {'Count':>6} {'Avg Rating':>12} {'Avg Price':>11} {'Total Votes':>12}")
    print("-"*70)
    
    for city, row in city_stats.head(15).iterrows():
        print(f"{city:<25} {int(row['Restaurant Count']):6d} {row['Avg Rating']:12.2f} "
              f"{row['Avg Price Range']:11.2f} {int(row['Total Votes']):12d}")
    print()
    
    return city_stats

def analyze_cuisines_by_city(df, top_n_cities=10):
    """
    Analyzes popular cuisines by city.
    """
    print("="*70)
    print("POPULAR CUISINES BY CITY")
    print("="*70 + "\n")
    
    top_cities = df['City'].value_counts().head(top_n_cities).index
    
    for city in top_cities:
        city_df = df[df['City'] == city]
        
        # Extract all cuisines (split by comma)
        all_cuisines = []
        for cuisines in city_df['Cuisines'].dropna():
            all_cuisines.extend([c.strip() for c in str(cuisines).split(',')])
        
        # Count cuisine occurrences
        cuisine_counts = pd.Series(all_cuisines).value_counts()
        
        print(f"📍 {city} (Top 5 Cuisines):")
        for idx, (cuisine, count) in enumerate(cuisine_counts.head(5).items(), 1):
            print(f"   {idx}. {cuisine}: {count} restaurants")
        print()

def identify_insights(df, city_stats):
    """
    Identifies interesting patterns and insights.
    """
    print("="*70)
    print("KEY INSIGHTS AND PATTERNS")
    print("="*70 + "\n")
    
    # Insight 1: High-density areas
    print("🔍 INSIGHT 1: High-Density Restaurant Areas")
    print("-"*70)
    top_5_cities = city_stats.head(5)
    total_restaurants = len(df)
    top_5_count = top_5_cities['Restaurant Count'].sum()
    percentage = (top_5_count / total_restaurants) * 100
    
    print(f"The top 5 cities contain {int(top_5_count)} restaurants ({percentage:.1f}% of total)")
    for city, row in top_5_cities.iterrows():
        print(f"  • {city}: {int(row['Restaurant Count'])} restaurants")
    print()
    
    # Insight 2: Rating vs Location
    print("🔍 INSIGHT 2: Cities with Highest Average Ratings")
    print("-"*70)
    # Filter cities with at least 10 restaurants for meaningful averages
    significant_cities = city_stats[city_stats['Restaurant Count'] >= 10]
    top_rated = significant_cities.nlargest(5, 'Avg Rating')
    
    for city, row in top_rated.iterrows():
        print(f"  • {city}: {row['Avg Rating']:.2f} avg rating ({int(row['Restaurant Count'])} restaurants)")
    print()
    
    # Insight 3: Price range patterns
    print("🔍 INSIGHT 3: Most Expensive Cities (by Average Price Range)")
    print("-"*70)
    most_expensive = significant_cities.nlargest(5, 'Avg Price Range')
    
    for city, row in most_expensive.iterrows():
        print(f"  • {city}: {row['Avg Price Range']:.2f} avg price range ({int(row['Restaurant Count'])} restaurants)")
    print()
    
    # Insight 4: Popularity by votes
    print("🔍 INSIGHT 4: Most Popular Cities (by Total Votes)")
    print("-"*70)
    most_popular = city_stats.nlargest(5, 'Total Votes')
    
    for city, row in most_popular.iterrows():
        print(f"  • {city}: {int(row['Total Votes']):,} total votes ({int(row['Restaurant Count'])} restaurants)")
    print()

def export_to_csv(city_stats, filename='city_statistics.csv'):
    """
    Exports city statistics to CSV for external visualization.
    """
    city_stats.to_csv(filename)
    print(f"✅ City statistics exported to: {filename}\n")

if __name__ == "__main__":
    dataset_path = 'c:/Users/drash/OneDrive/Desktop/project/Cognifyz Project/Dataset .csv'
    
    # Load and explore data
    df = load_and_explore_data(dataset_path)
    
    # Analyze by city
    city_counts = analyze_by_city(df)
    
    # Calculate statistics
    city_stats = calculate_city_statistics(df)
    
    # Analyze cuisines by city
    analyze_cuisines_by_city(df, top_n_cities=10)
    
    # Identify insights
    identify_insights(df, city_stats)
    
    # Export data
    export_to_csv(city_stats)
    
    print("="*70)
    print("✅ LOCATION-BASED ANALYSIS COMPLETE")
    print("="*70 + "\n")
