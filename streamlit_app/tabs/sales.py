import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from fastapi import FastAPI, Depends, HTTPException, status
import os
import json
from datetime import time
from extra_streamlit_components import tab_bar, TabBarItemData
from PIL import Image

title = "Sell"
sidebar_name = "Sell"
prePath = st.session_state.PrePath
new_products_folder_path = "data/predict"

cat_code = ["10","40","50","60","1140","1160","1180","1280","1281","1300","1301","1302","1320","1560","1920","1940","2060","2220","2280","2403","2462","2522","2582","2583","2585","2705","2905"]
cat_label = [
    "Livres occasion",
    "Jeux consoles neuf",
    "Accesoires gaming",
    "Consoles de jeux",
    "Objets pop culture",
    "Cartes de jeux",
    "Jeux de rôle et figurines",
    "Jouets enfant",
    "Jeux enfant",
    "Modélisme",
    "Chaussettes enfant",
    "Jeux de plein air",
    "Puériculture",
    "Mobilier",
    "Linge de maison",
    "Epicerie",
    "Décoration",
    "Animalerie",
    "Journaux et revues occasion",
    "Lots livres et magazines",
    "Jeux videos occasion",
    "Fournitures papeterie",
    "Mobilier de jardin",
    "Piscine et accessoires",
    "Outillage de jardin",
    "Livres neufs",
    "Jeux PC",
    ]

@st.cache_data        
def find_designation(prod_index):
    return X_test_df["designation"].loc[prod_index]

@st.cache_data
def find_cat_label(cat_sel):
    global cat_code, cat_label
    return cat_label[cat_code.index(cat_sel)]

def select_product():   
    return st.selectbox("designation:",X_test_df.index, format_func = find_designation, label_visibility="hidden")

def display_image(index):
    imageid = X_test_df["imageid"].loc[index]
    productid = X_test_df["productid"].loc[index]
    image_path = prePath+new_products_folder_path + \
        f"/image_test/image_{imageid}_product_{productid}.jpg"
    image = Image.open(image_path)
    st.image(image, caption='Selected image', use_column_width=True)

def select_category(index):
    global cat_code
    index = cat_code.index(index)
    return st.selectbox("Category:",cat_code, index = index, format_func = find_cat_label)

def run():
    
    st.write("")
    st.title(title)
    st.markdown('''
                ---
                ''')

    if st.session_state.UserAuthorization <1:
        st.write("You are not logged in or not authorized to sell product !")
        return
    
    if "sale_step" not in st.session_state:
        st.session_state.sale_step = 1
 
    chosen_id = tab_bar(data=[
        TabBarItemData(id="tab1", title="Step 1", description="Suggest New Products"),
        TabBarItemData(id="tab2", title="Step 2", description="Predict Categories"),
        TabBarItemData(id="tab3", title="Step 3", description="Confirm and Sell")],
        default="tab1")
    
    # Step 1: Suggest new products
    if (chosen_id == "tab1"):
        global X_test_df
        
        st.subheader("Step 1: Suggest New Products")
        num_products = st.number_input("Number of Products to Suggest", min_value=1, max_value=10, value=2)
        st.write("session state 1 :",st.session_state.sale_step)
        if (st.session_state.sale_step>=1):
            response1 = requests.get(
                'http://'+st.session_state.api_flows+':8003/new_product_proposal',
                headers={'Content-Type': 'application/json', 'Authorization': f"Bearer {st.session_state.token}"},
                data=json.dumps({
                    "num_products": num_products,
                    "new_products_folder_path": new_products_folder_path,
                    "api_secured": True
                })
            )
            st.write("new_product_proposal - status_code:", response1.status_code)
            if response1.status_code == 200:
                X_test_path = new_products_folder_path+"/X_test_update.csv"           
                try:
                    X_test_df = pd.read_csv(prePath+X_test_path, index_col=0)
                    st.write(X_test_df[["designation","description"]].to_html(index=False), unsafe_allow_html=True)
                    if st.session_state.sale_step==1:
                        st.session_state.sale_step = 2
                    st.success(response1.json().get("message", "No message in response"))
                    selected = select_product()
                    display_image(selected)

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error suggesting products: {e}")
            else:
                st.error("Failed to suggest products")
                return
   

        
    # Step 2: Predict categories for the suggested products
    if (chosen_id == "tab2"):
        global new_classes_df, df_concatenated
        
        st.subheader("Step 2: Predict Categories")
        st.write("session state 2 :",st.session_state.sale_step)
        if (st.session_state.sale_step >= 2):
            response2 = requests.post(
                'http://'+st.session_state.api_predict+':8000/prediction',
                headers={'Content-Type': 'application/json', 'Authorization': f"Bearer {st.session_state.token}"},
                data=json.dumps({
                    "dataset_path": f"{new_products_folder_path}/X_test_update.csv",
                    "images_path": f"{new_products_folder_path}/image_test",
                    "prediction_path": f"{new_products_folder_path}",
                    "api_secured": True
                    })
                )
            st.write("prediction - status_code:", response2.status_code)
            if response2.status_code == 200:
                try:
                    if "new_classes_df" not in st.session_state:
                        new_classes_df = pd.read_csv(prePath+new_products_folder_path+"/new_classes.csv")
                        st.session_state.new_classes_df = new_classes_df
                    else:
                        nc = pd.read_csv(prePath+new_products_folder_path+"/new_classes.csv")
                        new_classes_df['cat_pred'] = nc['cat_pred']
                        st.session_state.new_classes_df = new_classes_df
                        
                    if st.session_state.sale_step==2:
                        st.session_state.sale_step = 3
                    st.success(response2.json().get("message", "No message in response"))
                except Exception as e:
                    st.session_state.sale_step = 2
                    raise HTTPException(status_code=500, detail=f"Error suggesting products: {e}")
            else:
                st.session_state.sale_step = 2
                st.error("Failed to predict categories")
                return
                        
            df_concatenated= X_test_df.copy()
            df_concatenated['cat_real'] = np.nan
            df_concatenated['cat_pred'] = np.nan
            for i in range(len(df_concatenated)):
                df_concatenated['cat_real'].iloc[i] = new_classes_df['cat_real'].iloc[i].astype(int)
                df_concatenated['cat_pred'].iloc[i] = new_classes_df['cat_pred'].iloc[i].astype(int)
            
            # df_concatenated[["cat_real","cat_pred","designation","description"]]= st.data_editor(df_concatenated[["cat_real","cat_pred","designation","description"]], key="editor 1") # = st.data_editor(df_concatenated[["designation","description","cat_real","cat_pred"]])     
            selected = select_product()
            new_cat = select_category(df_concatenated.loc[selected, "cat_real"].astype(int).astype(str))
            df_concatenated["cat_real"].loc[selected]=int(new_cat)
            df_concatenated[["cat_real","cat_pred","designation","description"]]= st.data_editor(df_concatenated[["cat_real","cat_pred","designation","description"]], key="editor 2") # = st.data_editor(df_concatenated[["designation","description","cat_real","cat_pred"]])     
            display_image(selected)
            
            # Mettre à jour les DataFrames d'origine avec les valeurs actualisées
            for i in range(len(df_concatenated)):
                new_classes_df['cat_real'].iloc[i] = df_concatenated['cat_real'].iloc[i]
                new_classes_df['cat_pred'].iloc[i] = df_concatenated['cat_pred'].iloc[i]

            X_test_df['designation'] = df_concatenated['designation']
            X_test_df['description'] = df_concatenated['description']
            
            # Sauvegarder le DataFrame X_test_df mis à jour dans le fichier CSV
            csv_path = prePath + new_products_folder_path
            X_test_df.to_csv(csv_path+"/X_test_update.csv")
            new_classes_df.to_csv(csv_path+"/new_classes.csv", index=False)
            
            
    # Step 3: Confirm selling the products with chosen categories
    if (chosen_id == "tab3"):
        st.subheader("Step 3: Confirm and Sell")
        st.write("session state 3 :",st.session_state.sale_step)   
        if (st.session_state.sale_step >= 3):
            response3 = requests.get(
                'http://'+st.session_state.api_flows+':8003/add_new_products',
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {st.session_state.token}'},
                data=json.dumps({
                    "new_products_origin_path": f"{new_products_folder_path}",
                    "new_products_dest_path": "data/preprocessed",
                    "api_secured": True
                })
            )
            
            st.write("add_new_products - status_code:", response3.status_code)
            if response3.status_code == 200:
                if st.session_state.sale_step==3:
                    st.session_state.sale_step = 1
                st.write("session state 4 :",st.session_state.sale_step)
                st.success(response3.json().get("message", "No message in response"))
                del st.session_state.new_classes_df
                st.write(df_concatenated[["designation","description","cat_real","cat_pred"]].to_html(index=False), unsafe_allow_html=True)
                st.write("#### **If you want to sell more products, go to step 1**")
                






# if __name__ == "__main__":
#     run()

