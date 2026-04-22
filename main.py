import streamlit as st
import numpy as np
import pandas as pd
import pickle #To load a saved model

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Data Used in Prediction Model']) #two pages

if app_mode == 'Home':
    st.header('this is the !')
    st.subheader('This app allows you to estimate the monthly rental price of a condo unit in Phnom Penh given information required.')
    st.image('condo_pic.jpg')
    st.subheader('Sir/Mme, You need to fill all necessary information on the sidebar in order to get a prediction price!')
    st.sidebar.header("Informations about the condo unit :")
    Area = st.sidebar.slider('Area in m2', 25, 250, 25)
    No_bed = st.sidebar.radio('Number of bedrooms', options=[1, 2, 3, 4])
    Khan = st.sidebar.selectbox('Khan where the proper located', ['Chamkar Mon', 'Chbar Ampov', 'Doun Penh', 'Mean Chey', 'Prampir Meakkakra', 'Russey Keo', 'Saensokh', 'Tuol Kouk'])
    class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7 = 0,0,0,0,0,0,0,0
    if Khan == 'Chamkar Mon':
        class_0 = 1
    elif Khan == 'Chbar Ampov':
        class_1 = 1
    elif Khan == 'Doun Penh':
        class_2 = 1
    elif Khan == 'Mean Chey':
        class_3 = 1
    elif Khan == 'Prampir Meakkakra':
        class_4 = 1
    elif Khan == 'Russey Keo':
        class_5 = 1
    elif Khan == 'Saensokh':
        class_6 = 1
    else:
        class_7 = 1

    feature_list = [Area, No_bed, class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7]
    single_sample = np.array(feature_list).reshape(1,-1)

    if st.sidebar.button('Predict'):
        loaded_model = pickle.load(open('Random_Forest_Regressor.sav', 'rb'))
        prediction = loaded_model.predict(single_sample)
        st.subheader(f'The predicted monlthly rental price is: {int(prediction[0])} $.')
        

if app_mode == 'Data Used in Prediction Model':
    st.header('This page presents the data used to train the prediction model.')
    st.markdown('Dataset :')
    df = pd.read_excel('Condo_rental_raw.xlsx')
    st.write(df.head())

    st.write('For this study, we are interested in prediction rental price with three features, Area(m2), No. bedroom and Khan.')
    df = df[['Rental price (USD/month)', 'Area (m2)', 'No. bedroom', 'Khan']]
    st.markdown('Dataset with only interested features :')
    st.write(df.head())

    st.write('Check size of data: ')
    st.write(df.shape)

    st.write('Check number of null values in each collumn: ')
    st.write(df.isnull().sum())

    st.write('As null values represent less than 6 percents of the data, we decide to remove all rows containing them: ') 
    df = df.dropna()
    df = df.reset_index(drop=True)
    st.markdown('Dataset after dropping all rows with null value :')
    st.write(df.head())

    st.write('Check null values again: ')
    st.write(df.isnull().sum())

    st.write('Check size after deleting null values')
    st.write(df.shape)

    st.write('Check information of data: ')
    st.write(df.info())

    st.write('Summary statistic (numerical values): ')
    st.write(df.describe())

    st.write('Summary statistic (cateogrical values)): ')
    st.write(df.describe(include = 'object'))

    st.write('Check unique values of Khan)): ')
    st.write(sorted(df.Khan.unique()))

    st.write('One hot encode for each variable of Khan') 
    for khan in sorted(df.Khan.unique()):
        df[khan] = (df['Khan']==khan).astype(int)
    st.markdown('Dataset after one hot encode of khan:')
    st.write(df.head())

    st.write("Remove 'Khan' column") 
    df.drop(columns='Khan', inplace=True)
    st.markdown('Dataset after removing Khan column:')
    st.write('df.head()')

    st.write('Convert dataframe to numpy array:')
    data = df.to_numpy()
    st.markdown('Dataset after converting to numpy array: ')
    st.write(data[:5])


    
