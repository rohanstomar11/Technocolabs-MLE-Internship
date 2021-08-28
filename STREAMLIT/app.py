import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('final_rf.pkl')
le_category = joblib.load('le_category.pkl')

df = pd.read_csv('kickstarter_Cpgn.csv')

dict = {
    'GBP': 0.6778660151808138,
    'USD': 1.0,
    'CAD': 1.2322339772092068,
    'AUD': 1.266933969413733,
    'NOK': 8.129581125700815,
    'EUR': 0.8914663230598857,
    'MXN': 19.048823818125943,
    'SEK': 8.44495900369602,
    'NZD': 1.3642612798700928,
    'CHF': 0.9847802417687891,
    'DKK': 6.651297224481752,
    'HKD': 7.793060053828166,
    'SGD': 1.379506902407431,
    'JPY': 112.54898550773528}


def goal_in_usd(goal, currency):
    return goal/dict[currency]


def predict_success(features):

    features = np.array(features).astype(np.float64).reshape(1, -1)

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return prediction, probability


def transform_features(df_input):

    df_input['deadline'] = df_input['deadline'].apply(pd.to_datetime)
    df_input['launched'] = df_input['launched'].apply(pd.to_datetime)
    df_input['duration'] = df_input['deadline']-df_input['launched']
    df_input['duration'] = df_input['duration'].dt.days

    df_input['name_letter'] = df_input['name'].str.len()
    df_input['name_word'] = df_input['name'].apply(
        lambda x: len(str(x).split(' ')))
    df_input['name_question'] = (df_input['name'].str[-1] == '?').astype(int)
    df_input['name_upper'] = df_input['name'].str.isupper().astype(float)

    df_input['launched_year'] = df_input['launched'].dt.year
    df_input['launched_quarter'] = df_input['launched'].dt.quarter
    df_input['launched_month'] = df_input['launched'].dt.month
    df_input['launched_week'] = df_input['launched'].dt.week
    df_input['launched_date'] = df_input['launched'].dt.day
    df_input['launched_day'] = df_input['launched'].dt.dayofweek

    competition_year = df_input.groupby(['category', 'launched_year']).count()
    competition_year = competition_year[['name']]
    competition_year.reset_index(inplace=True)
    competition_quarter = df_input.groupby(
        ['category', 'launched_year', 'launched_quarter']).count()
    competition_quarter = competition_quarter[['name']]
    competition_quarter.reset_index(inplace=True)
    competition_month = df_input.groupby(
        ['category', 'launched_year', 'launched_month']).count()
    competition_month = competition_month[['name']]
    competition_month.reset_index(inplace=True)

    year = ['category', 'launched_year', 'competition_year']
    competition_year.columns = year
    quarter = ['category', 'launched_year',
               'launched_quarter', 'competition_quarter']
    competition_quarter.columns = quarter
    month = ['category', 'launched_year',
             'launched_month', 'competition_month']
    competition_month.columns = month

    df_input = pd.merge(df_input, competition_year, on=[
                        'category', 'launched_year'], how='left')
    df_input = pd.merge(df_input, competition_quarter, on=[
                        'category', 'launched_year', 'launched_quarter'], how='left')
    df_input = pd.merge(df_input, competition_month, on=[
                        'category', 'launched_year', 'launched_month'], how='left')

    df_input.drop(columns=['name', 'deadline',
                  'launched', 'country'], inplace=True)

    df_input['currency'] = df_input['currency'] == 'USD'

    df_input['category'] = le_category.transform(df_input['category'])
    return df_input


def main():

    html_temp = """
        <div style = "background-color: #87c442; padding: 10px; border-radius: 25px; ">
            <center><h1 style = "color: black; "><i>Crowdfunding</i>: Predicting Success Of Kickstarter Projects</h1></center>
        </div><br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    NAME = st.text_input("Name of the Project")

    main_category = [
        'Art',
        'Comics',
        'Crafts',
        'Dance',
        'Design',
        'Fashion',
        'Film & Video',
        'Food',
        'Games',
        'Journalism',
        'Music',
        'Photography',
        'Publishing',
        'Technology',
        'Theater'
    ]

    MAIN_CATEGORY = main_category.index(st.selectbox(
        "Select Main Category",
        tuple(main_category)
    )) + 1

    CATEGORY = st.text_input("Category")

    COUNTRY = st.text_input("Country (US, GB, DE, etc.)")

    CURRENCY = st.text_input("Currency (USD, GBP, EUR, etc.)")

    LAUNCHED = st.date_input("Launched date")

    DEADLINE = st.date_input("Project's Deadline")

    GOAL = st.number_input("Goal")

    BACKERS = st.number_input("BACKERS")

    if st.button("PREDICT"):
        USD_GOAL_REAL = goal_in_usd(GOAL, CURRENCY)
        features = [[NAME, CATEGORY, MAIN_CATEGORY, CURRENCY,
                     DEADLINE, GOAL, LAUNCHED, BACKERS, COUNTRY, USD_GOAL_REAL]]
        df_input = pd.DataFrame(features, columns=[
                                'name', 'category', 'main_category', 'currency', 'deadline', 'goal', 'launched', 'backers', 'country', 'usd_goal_real'])
        transformed = transform_features(df_input)
        prediction, probability = predict_success(transformed)
        if prediction[0] == 1:
            st.success("This project will be successful with a probability of {}%.".format(
                round(np.max(probability)*100, 2)))
        else:
            st.success("This project will not be successful with a probability of {}%.".format(
                round(np.max(probability)*100, 2)))


if __name__ == '__main__':
    main()
