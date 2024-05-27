import streamlit as st
import pandas as pd
import pickle
# from sklearn.preprocessing import StandardScaler, OrdinalEncoder


def prepare_data(df1):
    scaler = pickle.load(open('standard_scaler.pkl', 'rb'))
    encoder = pickle.load(open('ordinal_encoder.pkl', 'rb'))

    cat_cols = df1.columns[df1.dtypes == 'object']
    num_cols = df1.columns[df1.dtypes != 'object']

    num_features = df1[num_cols]
    num_features = scaler.transform(num_features.values)
    df1[num_cols] = num_features

    cat_features = df1[cat_cols]
    cat_features = encoder.transform(cat_features.values)
    df1[cat_cols] = cat_features

    return df


model = pickle.load(open('model.pkl', 'rb'))


st.title('Diamond Price Predictor')


carat = st.number_input("Carat", value=None, placeholder="Type a number")
cut = st.selectbox("Cut", options=("Fair", "Good", "Ideal", "Premium", "Very Good"), index=None,
                   placeholder="Choose an option")
color = st.selectbox("Color", options=("D", "E", "F", "G", "H", "I", "J"), index=None, placeholder="Choose an option")
clarity = st.selectbox("Clarity", options=("I1", "IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2"),
                       index=None, placeholder="Choose an option")
depth = st.number_input("Depth", value=None, placeholder="Type a number")
table = st.number_input("Table", value=None, placeholder="Type a number")
x = st.number_input("x", value=None, placeholder="Type a number")
y = st.number_input("y", value=None, placeholder="Type a number")
z = st.number_input("z", value=None, placeholder="Type a number")

data = [[carat, cut, color, clarity, depth, table, x, y, z]]
df = pd.DataFrame(data, columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])


if st.button('Predict'):
    # preprocess
    df = prepare_data(df)
    # predict
    result = model.predict(df)
    # display
    st.metric(label="Predicted Price (in USD) ", value=result)


def reset():
    st.session_state.selection = 'Please Select'


st.button('Reset', on_click=reset)
