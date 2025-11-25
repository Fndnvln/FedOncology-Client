import streamlit as st
import flwr as fl
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="FedOncology Clinic", page_icon="🎗️")
st.title("🎗️ FedOncology: Clinic Node")

# Initialize Model
if 'model' not in st.session_state:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(30,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.session_state['model'] = model

# --- TABS FOR NAVIGATION ---
tab1, tab2 = st.tabs(["🧠 Collaborative Training", "🩺 Diagnose Patients"])

# ==========================================
# TAB 1: TRAINING
# ==========================================
with tab1:
    st.header("Participate in Federation")
    
    # [NEW] Input for Server Address (Allows connecting via ngrok)
    server_address = st.text_input("Server Address (HQ)", value="127.0.0.1:8080", help="Paste the ngrok address here if connecting remotely.")

    train_file = st.file_uploader("Upload Clinic History (CSV)", type="csv", key="train_up")

    if train_file:
        train_file.seek(0)
        df = pd.read_csv(train_file)
        st.success(f"Loaded {len(df)} records for training.")

        if 'diagnosis' in df.columns:
            # Preprocessing
            df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
            
            # Handle M/B mapping if needed
            if df['diagnosis'].dtype == 'object':
                df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
            
            X = df.drop('diagnosis', axis=1).values
            y = df['diagnosis'].values
            
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            if st.button("🚀 Join Federation & Train"):
                class TumorClient(fl.client.NumPyClient):
                    def get_parameters(self, config):
                        return st.session_state['model'].get_weights()
                    def fit(self, parameters, config):
                        st.session_state['model'].set_weights(parameters)
                        st.session_state['model'].fit(X, y, epochs=1, batch_size=32, verbose=0)
                        return st.session_state['model'].get_weights(), len(X), {}
                    def evaluate(self, parameters, config):
                        st.session_state['model'].set_weights(parameters)
                        loss, acc = st.session_state['model'].evaluate(X, y, verbose=0)
                        return loss, len(X), {"accuracy": acc}

                with st.spinner(f"Connecting to HQ at {server_address}..."):
                    try:
                        # [UPDATED] Uses the variable from the text input
                        fl.client.start_numpy_client(
                            server_address=server_address, 
                            client=TumorClient()
                        )
                        st.success("✅ Training Complete! Model Updated.")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
                        st.warning("Make sure the Server is running and the address is correct.")
        else:
            st.error("Training data must have a 'diagnosis' column.")

# ==========================================
# TAB 2: DIAGNOSIS
# ==========================================
with tab2:
    st.header("Predict Patient Risk")
    st.markdown("Upload unseen patient data (no diagnosis column) to test the model.")
    
    test_file = st.file_uploader("Upload Test Data (CSV)", type="csv", key="test_up")
    
    if test_file:
        test_file.seek(0)
        df_test = pd.read_csv(test_file)
        
        # Clean same as training
        clean_df = df_test.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1, errors='ignore')
        
        st.info(f"Ready to diagnose {len(clean_df)} patients.")
        
        if st.button("🔎 Run Diagnosis"):
            # Scale data (Using local fit for demo purposes)
            scaler = StandardScaler()
            X_new = scaler.fit_transform(clean_df.values)
            
            # Predict using the Global Model
            predictions = st.session_state['model'].predict(X_new)
            
            # Display Results nicely
            results = clean_df.copy()
            results['Malignancy Risk'] = predictions
            results['Prediction'] = results['Malignancy Risk'].apply(lambda x: "Malignant" if x > 0.5 else "Benign")
            
            st.dataframe(results[['Malignancy Risk', 'Prediction']].style.highlight_max(axis=0))
            
            # Summary Chart
            st.bar_chart(results['Prediction'].value_counts())