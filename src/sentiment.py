
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
nltk.download('vader_lexicon')


def read_feedback_data(file_path: str) -> pd.DataFrame:
    """
    Reads warehouse feedback data from a CSV file.

    Steps
    ------
    - Reads the CSV file into a DataFrame.
    - Ensures the feedback column is string-typed.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing warehouse feedback.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the feedback data.

    Example Usage
    -------------
    df = read_feedback_data('warehouse_feedback.csv')
    """
    df = pd.read_csv(file_path)
    if 'feedback' in df.columns:
        df['feedback'] = df['feedback'].astype(str)
    return df

"""## Perform sentiment analysis

Use VADER to analyse the sentiment of each feedback entry to quantify the positivity or negativity of the feedback.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


def perform_sentiment_analysis(feedback_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs sentiment analysis on warehouse feedback using VADER.

    Steps
    ------
    - Initialises the VADER sentiment analyser.
    - Apply the analyser to each feedback entry.
    - Extract sentiment scores (negative, neutral, positive, compound).
    - Append these scores as new columns to the original DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional sentiment score columns.

    Example Usage
    -------------
    df = perform_sentiment_analysis(feedback_df)
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = feedback_df['feedback'].apply(lambda x: analyzer.polarity_scores(x))
        feedback_df['neg_sentiment'] = sentiment_scores.apply(lambda x: x['neg'])
        feedback_df['neu_sentiment'] = sentiment_scores.apply(lambda x: x['neu'])
        feedback_df['pos_sentiment'] = sentiment_scores.apply(lambda x: x['pos'])
        feedback_df['compound_sentiment'] = sentiment_scores.apply(lambda x: x['compound'])
        return feedback_df
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return feedback_df

"""## Categorise feedback

Categorise the feedback based on the type of inefficiency mentioned (e.g., picking, storage, delays).

"""

def categorise_inefficiency(feedback: str) -> str:
    """
    Categorises feedback into warehouse inefficiency types based on keywords.

    Steps
    ------
    - Convert feedback to lowercase for consistent keyword matching.
    - Check for presence of keywords related to picking, storage, delays, damage, etc.
    - Return the corresponding inefficiency category.

    Returns
    -------
    str
        Inefficiency category label.

    Example Usage
    -------------
    category = categorise_inefficiency("Picking speed is too slow.")
    """
    feedback_lower = feedback.lower()
    try:
        if 'picking' in feedback_lower or 'picked' in feedback_lower or 'picker' in feedback_lower:
            if 'error' in feedback_lower or 'incorrect' in feedback_lower:
                return 'Picking Errors'
            elif any(word in feedback_lower for word in ['slow', 'speed', 'difficult', 'searching']):
                return 'Picking Efficiency Issues'
            else:
                return 'Picking Issues (General)'
        elif any(word in feedback_lower for word in ['storage', 'stored', 'racks', 'shelves', 'layout']):
            return 'Storage Issues'
        elif any(word in feedback_lower for word in ['delay', 'delayed', 'backlogs', 'queues', 'bottleneck']):
            return 'Delays'
        elif 'damage' in feedback_lower or 'damaged' in feedback_lower:
            return 'Damage'
        elif any(word in feedback_lower for word in ['forecast', 'volume', 'planning', 'shipments']):
            return 'Forecast Gaps'
        else:
            return 'Other Inefficiency'
    except Exception as e:
        print(f"Error categorising feedback: {e}")
        return 'Uncategorised'

"""## Analyse sentiment by category

Calculate the average sentiment score for each category of inefficiency to identify which areas are causing the most warehouse problem.
"""

def apply_inefficiency_categorisation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies inefficiency categorisation to a DataFrame of feedback.

    Steps
    ------
    - Apply the categorise_inefficiency function to each feedback entry.
    - Add a new column with the resulting category.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'inefficiency_category' column.

    Example Usage
    -------------
    df = apply_inefficiency_categorisation(feedback_df)
    """
    try:
        df['inefficiency_category'] = df['feedback'].apply(categorise_inefficiency)
        return df
    except Exception as e:
        print(f"Error applying inefficiency categorisation: {e}")
        return df

def analyse_sentiment_by_category(df: pd.DataFrame) -> pd.Series:
    """
    Analyses average sentiment by inefficiency category.

    Steps
    ------
    - Groups feedback by inefficiency category
    - Calculates average compound sentiment score for each category
    - Sorts the results in ascending order

    Returns
    -------
    pd.Series
        Series of average sentiment scores by category.

    Example Usage
    -------------
    sentiment_summary = analyse_sentiment_by_category(df)
    """
    sentiment_by_category = df.groupby('inefficiency_category')['compound_sentiment'].mean()
    sentiment_by_category_sorted = sentiment_by_category.sort_values(ascending=True)
    return sentiment_by_category_sorted