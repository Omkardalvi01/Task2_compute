import pickle
import streamlit as st
import pandas as pd

pickle_in = open('model.pkl','rb')
scaler,encoder,pca,model = pickle.load(pickle_in)


def prediction(dataframe):
    pca_data = pca.transform(dataframe)
    predict = model.predict(pca_data)
    return predict
    
    


def main():
    st.title('Automobile Price Predictor')

    make = st.selectbox('Manufacturer',options = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
       'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
       'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche', 'renault',
       'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'], placeholder="Choose an option")
    body_style = st.selectbox('Body Style',options=['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'])
    asp = st.selectbox('Aspiration', options=['std','turbo'])
    l = st.number_input('Length',value=None, min_value=0.0,)
    w = st.number_input('Width',value=None, min_value=0.0)
    weight = st.number_input('Curb Weight',value=None, min_value=0.0)
    fuel_type = st.selectbox('Fuel-Type',options=['gas','diesel'])
    fuel_sys = st.selectbox('Fuel-System',options=['mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi'])
    engine_loc = st.selectbox('Engine Location', options=['front', 'rear'])
    drive_wheels = st.selectbox('Drive Wheels',options=['rwd', 'fwd', '4wd'])
    hp = st.number_input('Horsepower',value=None, min_value=0.0)
    engine_type = st.selectbox('Engine Type',options=['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv'])
    num_of_cyliner = st.selectbox('num_of_clinders', options=[2,3,4,5,6,8,12])
    bore = st.number_input('Bore',value=None, min_value=0.0)
    city_mpg = st.number_input('City MPG',value=None, min_value=0.0)
    highway_mpg = st.number_input('Highway MPG', value=None, min_value=0.0)
    whl_bs = st.number_input('Wheel Base',value=None, min_value=0.0)
    engine_size = st.number_input('Engine Size', value=None, min_value=0)
    price = "Hello"

    cat_data = {
            'fuel-type': [fuel_type],
            'make': [make],
            'aspiration': [asp],
            'body-style': [body_style],
            'drive-wheels': [drive_wheels],
            'engine-location': [engine_loc],
            'engine-type': [engine_type],
            'fuel-system': [fuel_sys]}

    cat_df = pd.DataFrame(cat_data)

    encoded_data = encoder.transform(cat_df)

    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_df.columns))

    num_data = {
        'city-mpg' : [city_mpg],
        'bore' : [bore],
        'num-of-cylinders' : [num_of_cyliner],
        'highway-mpg' : [highway_mpg],
        'width' : [w],
        'horsepower' : [hp],
        'engine-size' : [engine_size],
        'length' : [l],
        'curb-weight' : [weight],
        'wheel-base' : [whl_bs]
    }

    num_df = pd.DataFrame(num_data)
    num_scaled = scaler.transform(num_df)
    num_scaled_df = pd.DataFrame(num_scaled, columns=num_df.columns)
    df = pd.concat([num_scaled_df, encoded_df],axis=1)
    order = ['wheel-base', 'length', 'width', 'curb-weight', 'num-of-cylinders',
       'engine-size', 'bore', 'horsepower', 'city-mpg', 'highway-mpg',
       'fuel-type_diesel', 'fuel-type_gas', 'make_alfa-romero', 'make_audi',
       'make_bmw', 'make_chevrolet', 'make_dodge', 'make_honda', 'make_isuzu',
       'make_jaguar', 'make_mazda', 'make_mercedes-benz', 'make_mercury',
       'make_mitsubishi', 'make_nissan', 'make_peugot', 'make_plymouth',
       'make_porsche', 'make_renault', 'make_saab', 'make_subaru',
       'make_toyota', 'make_volkswagen', 'make_volvo', 'aspiration_std',
       'aspiration_turbo', 'body-style_convertible', 'body-style_hardtop',
       'body-style_hatchback', 'body-style_sedan', 'body-style_wagon',
       'drive-wheels_4wd', 'drive-wheels_fwd', 'drive-wheels_rwd',
       'engine-location_front', 'engine-location_rear', 'engine-type_dohc',
       'engine-type_l', 'engine-type_ohc', 'engine-type_ohcf',
       'engine-type_ohcv', 'engine-type_rotor', 'fuel-system_1bbl',
       'fuel-system_2bbl', 'fuel-system_4bbl', 'fuel-system_idi',
       'fuel-system_mfi', 'fuel-system_mpfi', 'fuel-system_spdi',
       'fuel-system_spfi']
    df = df.reindex(columns=order).copy()
    if st.button("Predict"):
        result = prediction(dataframe=df)
        price = f'Price: {result[0]}'
        st.write(price)
         
if __name__ == '__main__':
    main()