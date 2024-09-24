import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

pipe = joblib.load(open("model.pkl", "rb"))
df = joblib.load(open("df.pkl", "rb"))

st.markdown("<h1 style='text-align: center; color: yellow; font-weight: bold; font-size: 45px; '>Laptop Price Predictor</h1>",  unsafe_allow_html=True)
st.markdown("***")
# Company	TypeName	CPU_Frequency (GHz)	RAM (GB)	GPU_Company	Weight (kg)	Price (Euro)	Touchscreen	IPS	X_res	Y_res	ppi	CPU Brand	HDD	SSD	OS
# st.write(df.head())

col1, col2, col3 = st.columns(3)

with col1:
    company = st.selectbox("Laptop Brand", df["Company"].unique())
    cpu = st.selectbox("CPU", df["CPU Brand"].unique())
    hdd = st.selectbox('HDD(GB)',[0,128,256,512,1024,2048])
    ips = st.selectbox("IPS Panel", ["Yes", "No"])
    frequency = st.number_input("CPU Frequency(GHz)", min_value=1.0, max_value=5.0, value=2.4)

with col2:
    ram = st.selectbox("RAM(GB)", [2,4,6,8,12,16,24,32,64])
    gpu_company = st.selectbox("GPU", df["GPU_Company"].unique())
    ssd = st.selectbox('SSD(GB)',[0,8,128,256,512,1024])
    touchscreen = st.selectbox("Touch Screen", ["Yes", "No"])

with col3:
    typename = st.selectbox("Type", df["TypeName"].unique())
    os = st.selectbox('Operating System',df['OS'].unique())
    resolution = st.selectbox("Screen Resolution", ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440','3072x1920','1440x900','3840x2400','1600x900','2736x1824'])
    screen_size = st.number_input("Screen Size(Inch)", min_value=10.0, max_value=18.0, value=13.0)
    weight = st.number_input("Weight(Kg)", min_value=0.5, max_value=5.0, value=2.0)


if st.button('Predict Price'):

    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,typename,frequency,ram,gpu_company,weight,touchscreen,ips,X_res,Y_res,ppi,cpu,hdd,ssd,os], dtype=object)

    #array(['Company', 'TypeName', 'CPU_Frequency (GHz)', 'RAM (GB)',
    #   'GPU_Company', 'Weight (kg)', 'Touchscreen', 'IPS', 'X_res',
    #   'Y_res', 'ppi', 'CPU Brand', 'HDD', 'SSD', 'OS'], dtype=object)

    #query = query.reshape(1, -1)
    query = query.reshape(1,15)
    #st.header(pipe.predict(query))
    st.title("The predicted price: "+ str(int(np.exp(((pipe.predict(query)[0]))))) + " €")
    st.header("In Rupees "  + str(93.2 * int(np.exp(pipe.predict(query)[0]))))
    # st.header(int(pipe.predict(query)[0]))
    # st.header(pipe.predict(query)[0])

    