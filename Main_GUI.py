from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter as tk


# ===============================
# Core Python Libraries
# ===============================
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from scipy.special import expit  # sigmoid

# ===============================
# Data Visualization
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image, ImageTk
import cv2
# ===============================
# System & Utilities
# ===============================
import os
import joblib
import numpy as np
from tqdm import tqdm
import re

# ===============================
# NLP: NLTK 
# ===============================
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.util import ngrams

# NLTK Downloads
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ===============================
# Scikit-learn: Preprocessing, Models, Metrics
# ===============================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier

# ===============================
# TensorFlow / Keras
# ===============================
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ===============================
# Transformers (Hugging Face)
# ===============================
import torch
from transformers import (XLNetTokenizer, XLNetForSequenceClassification)
from transformers import  XLNetModel
# ===============================
# Custom Modules
# ===============================
from metrics_calculator import MetricsCalculator
from graphs import GraphPlotter

# ===============================
# Model Directory
# ===============================
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

model_name='xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetModel.from_pretrained(model_name)
model.eval()

def upload_dataset():
    """Load the dataset from a CSV file"""
    global file_path,df
    text.delete('1.0', END)
    file_path = filedialog.askopenfilename(initialdir = "Dataset")

    df = pd.read_csv(file_path)
    text.insert(END,str(df.head())+"\n\n")


def preprocess_data(df, save_path=None, target_cols=None):

    global label_encoders
    label_encoders = {}  # dictionary to hold encoders for each target column

    if save_path and os.path.exists(save_path):
        print(f"Loading existing preprocessed file: {save_path}")
        df = pd.read_csv(save_path)
    else:
        print("Preprocessing data" + (f" and saving to: {save_path}" if save_path else " (no saving)"))
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = str(text).lower()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
            return ' '.join(tokens)

        # Separate target columns
        target_df = None
        if target_cols:
            existing_targets = [col for col in target_cols if col in df.columns]
            target_df = df[existing_targets].copy()
            df = df.drop(columns=existing_targets)

        # Process text columns
        text_columns = df.select_dtypes(include='object').columns
        for col in text_columns:
            df[f'processed_{col}'] = df[col].apply(clean_text)

        # Drop original text columns
        df.drop(columns=text_columns, inplace=True)

        # Reattach target columns
        if target_df is not None:
            for col in target_df.columns:
                df[col] = target_df[col]

        # Save only if path is specified
        if save_path:
            df.to_csv(save_path, index=False)

    # Select processed and numerical columns
    processed_text_cols = [col for col in df.columns if col.startswith('processed_')]
    non_text_cols = [col for col in df.columns if col not in processed_text_cols + (target_cols if target_cols else [])]

    # Join processed text columns into one string
    X_text = df[processed_text_cols].astype(str).agg(' '.join, axis=1)

    # Combine with numerical columns if any
    X_numeric = df[non_text_cols].values if non_text_cols else None
    if X_numeric is not None and len(X_numeric) > 0:
        X = [f"{text} {' '.join(map(str, numeric))}" for text, numeric in zip(X_text, X_numeric)]
    else:
        X = X_text.tolist()

    # Encode multiple target columns
    Y_dict = {}
    if target_cols:
        for col in target_cols:
            if col in df.columns:
                le = LabelEncoder()
                Y_dict[col] = le.fit_transform(df[col])
                label_encoders[col] = le

    return X, Y_dict

def plot_target_distributions(Y_dict):

    # Convert to DataFrame
    y_df = pd.DataFrame(Y_dict)

    # Loop through each target column
    for col in y_df.columns:

        # Print total number of rows
        total_rows = len(y_df[col])
        print(f"{col} → Total Rows: {total_rows}")

        # Plot count distribution
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=y_df[col])

        # Add count labels on each bar
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f'{height}',                     # text
                (p.get_x() + p.get_width()/2, height),  # position
                ha='center', va='bottom', 
                fontsize=10, fontweight='bold'
            )

        plt.title(f'Class Distribution: {col}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()



def Preprocess_Dataset_button():
    global df,X, Y_dict
    global labels1,labels2, labels3
    global metrics_calculator_lb1,metrics_calculator_lb2, metrics_calculator_lb3
    
    
    text.delete('1.0', END)

    X, Y_dict = preprocess_data(df, save_path="model/cleaned_data.csv", target_cols=["Ticket Priority","Customer Satisfaction Rating","Resolution"])

    labels_vars = {}          # labels lists
    labels_names = {}         # actual target names

    for i, (col, le) in enumerate(label_encoders.items(), start=1):
        var_name = f"labels{i}"
        labels_vars[var_name] = list(le.classes_)   # class names list
        labels_names[var_name] = col                # actual target name

    for var_name, labels in labels_vars.items():

            text.insert(END, f"{var_name}: {labels_names[var_name]}\n")

            for idx, class_name in enumerate(labels):
                text.insert(END, f"labels{idx}: {class_name}\n")

            text.insert(END, "\n")

            # ------------ SAVE LABELS TO .npy FILE ------------
            save_path = os.path.join(MODEL_DIR, f"{var_name}.npy")
            np.save(save_path, np.array(labels), allow_pickle=True)

    text.insert(END, "\nLabels saved successfully!\n\n")

    labels1 = labels_vars.get("labels1")
    metrics_calculator_lb1 = MetricsCalculator(labels1,text_widget=text)

    labels2 = labels_vars.get("labels2")
    metrics_calculator_lb2 = MetricsCalculator(labels2,text_widget=text)

    labels3 = labels_vars.get("labels3")
    metrics_calculator_lb3 = MetricsCalculator(labels3,text_widget=text)
    plot_target_distributions(Y_dict)

def eda_nlp_analysis():
    global df,X, Y_dict
    text.delete('1.0', END)

    X_text=X
    num_words=100
    top_n_words=20
    
    text.insert(END,"Generating NLP EDA Visualizations..."+"\n\n")

    # Flatten all tokens from all texts
    all_tokens = [word for doc in X_text for word in word_tokenize(doc)]

    # --- 1. WordCloud ---
    word_freq = Counter(all_tokens)
    wc = WordCloud(width=800, height=400, max_words=num_words, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Top {num_words} Words - WordCloud")
    plt.show()

    # --- 2. Top-N Frequent Words ---
    common_words = word_freq.most_common(top_n_words)
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(f"Top {top_n_words} Most Frequent Words")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.show()

    # --- 3. Document Length Histogram ---
    doc_lengths = [len(word_tokenize(doc)) for doc in X_text]
    plt.figure(figsize=(10, 5))
    sns.histplot(doc_lengths, bins=20, kde=True, color='teal')
    plt.title("Distribution of Document Lengths (in words)")
    plt.xlabel("Number of Words per Document")
    plt.ylabel("Frequency")
    plt.show()

    # --- 4. POS Tag Frequency ---
    all_pos = [tag for _, tag in pos_tag(all_tokens)]
    pos_counts = Counter(all_pos).most_common()
    pos_tags, pos_freqs = zip(*pos_counts)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(pos_tags), y=list(pos_freqs), palette="coolwarm")
    plt.title("Part of Speech (POS) Tag Frequency")
    plt.xlabel("POS Tag")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

    # --- 5. Bigram Frequency Plot ---
    bigrams = list(ngrams(all_tokens, 2))
    bigram_freq = Counter(bigrams).most_common(top_n_words)
    bigram_labels = [' '.join(b) for b, _ in bigram_freq]
    bigram_counts = [count for _, count in bigram_freq]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=bigram_counts, y=bigram_labels, palette="magma")
    plt.title(f"Top {top_n_words} Bigrams")
    plt.xlabel("Count")
    plt.ylabel("Bigram")
    plt.show()

def xlnet_feature_extraction(texts, model_name='xlnet-base-cased', batch_size=32, pooling='mean'):


    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting XLNet embeddings"):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = model(**encoded_input)

        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        if pooling == 'mean':
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            embeddings = sum_embeddings / sum_mask
        elif pooling == 'cls':
            embeddings = token_embeddings[:, -1, :]  
        else:
            raise ValueError("Pooling must be 'mean' or 'cls'")

        all_embeddings.append(embeddings.cpu().numpy())

    X = np.vstack(all_embeddings)
    return X, model


def feature_extraction(X_text, method='XLNet_Embeddings', model_dir='model', is_train=True):
    x_file = os.path.join(model_dir, f'X_{method}.pkl')

    text.insert(END, f"[INFO] Feature extraction method: {method}, Train mode: {is_train}"+"\n\n")

    if is_train:
        if os.path.exists(x_file):
            text.insert(END, f"[INFO] Loading cached embeddings from {x_file}"+"\n\n")
            X = joblib.load(x_file)
        else:
            text.insert(END, "[INFO] Computing embeddings..."+"\n\n")
            X, model = xlnet_feature_extraction(X_text, pooling='mean')
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(X, x_file)
    else:
        text.insert(END, "[INFO] Performing embedding extraction for testing..."+"\n\n")
        X, model = xlnet_feature_extraction(X_text, pooling='mean')

    return X
    
def feature_extraction_button():
    global X,features
    global features_dict,labels_dict
    text.delete('1.0', END)

    features = feature_extraction(X, method='XLNet_Embeddings', is_train=True)

    original_features = {}
    original_labels = {}

    for i, (key, y) in enumerate(Y_dict.items(), start=1):
        text.insert(END,f"\n🔹 Processing '{key}' (Original)..."+"\n\n")

        original_features[f'features{i}'] = features
        original_labels[f'Y{i}'] = y

        text.insert(END,f"Dataset: features{i}.shape = {features.shape}, Y{i}.shape = {y.shape}"+"\n\n")

    features_dict = {
        'Ticket Priority': original_features['features1'],
        'Customer Satisfaction Rating': original_features['features2'],
        'Resolution': original_features['features3']
    }

    labels_dict = {
        'Ticket Priority': original_labels['Y1'],
        'Customer Satisfaction Rating': original_labels['Y2'],
        'Resolution': original_labels['Y3']
    }



def train_ml_models(Algorithm_prefix, features_dict, Y_dict, algorithm):
    ml_models = {}

    model_mapping = {
        "QDA": QuadraticDiscriminantAnalysis,
        "LDA": LinearDiscriminantAnalysis,
        "HGB": HistGradientBoostingClassifier,
        "NC": NearestCentroid,
        "GPC": GaussianProcessClassifier,
        "SGD": SGDClassifier
    }

    if algorithm not in model_mapping:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    for target_name, y_encoded in Y_dict.items():
        X = features_dict[target_name]
        model_cls = model_mapping[algorithm]

        if algorithm == "QDA":
            mdl = model_cls(reg_param=1.0, tol=1e-1)
        elif algorithm == "LDA":
            mdl = model_cls()
        elif algorithm == "HGB":
            mdl = model_cls()
        elif algorithm == "NC":
            mdl = model_cls()
        elif algorithm == "SGD":
            mdl = model_cls(max_iter=2000, tol=1e-3)
        else:
            mdl = model_cls()

        model_path = f"model/{Algorithm_prefix}_{target_name}_{algorithm}_model.pkl"
        algo_name = f"{Algorithm_prefix} {algorithm} [{target_name}]"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        if os.path.exists(model_path):
            print(f"Loading existing {algorithm} model for {target_name}...")
            mdl = joblib.load(model_path)
        else:
            print(f"Training {algorithm} for target: {target_name}...")
            mdl.fit(X_train, y_train)
            joblib.dump(mdl, model_path)

        y_pred = mdl.predict(X_test)
        try:
            y_score = mdl.predict_proba(X_test)
        except:
            y_score = None

        if target_name == "Ticket Priority":
            metrics_calculator_lb1.calculate_metrics(algo_name, y_pred, y_test, y_score)
        elif target_name == "Customer Satisfaction Rating":
            metrics_calculator_lb2.calculate_metrics(algo_name, y_pred, y_test, y_score)
        elif target_name == "Resolution":
            metrics_calculator_lb3.calculate_metrics(algo_name, y_pred, y_test, y_score)
        else:
            print(f"No metrics calculator defined for target: {target_name}")

        ml_models[f"{target_name}_{algorithm}"] = mdl

    return ml_models
    
def existing_classifier1():
    text.delete('1.0', END)
   
    global features_dict,labels_dict
    models = train_ml_models("XLNet_Embeddings", features_dict, labels_dict, "QDA")

def existing_classifier2():
    text.delete('1.0', END)

    global features_dict,labels_dict
    models = train_ml_models("XLNet_Embeddings", features_dict, labels_dict, "LDA")

def existing_classifier3():
    text.delete('1.0', END)

    global features_dict,labels_dict
    models = train_ml_models("XLNet_Embeddings", features_dict, labels_dict, "SGD")

def existing_classifier4():
    text.delete('1.0', END)

    global features_dict,labels_dict
    models = train_ml_models("XLNet_Embeddings", features_dict, labels_dict, "NC")


def Proposed_classifier():
    text.delete('1.0', END)

    global features_dict,labels_dict
    models = train_ml_models("XLNet_Embeddings", features_dict, labels_dict, "HGB")


def Prediction():
    text.delete('1.0', END)

    filename = filedialog.askopenfilename(initialdir="Dataset")

    df_test1 = pd.read_csv(filename)
    df_result = df_test1.copy()

    # ---- Preprocess ----
    df_test, _ = preprocess_data(df_test1)
    features_test = feature_extraction(df_test, method='XLNet_Embeddings', is_train=None)

    # ---- Load Saved Labels (.npy files) ----
    labels1 = np.load("model/labels1.npy", allow_pickle=True)
    labels2 = np.load("model/labels2.npy", allow_pickle=True)
    labels3 = np.load("model/labels3.npy", allow_pickle=True)

    # ---- Prediction for Ticket Priority ----
    model_path = r"model\XLNet_Embeddings_Ticket Priority_HGB_model.pkl"
    model = joblib.load(model_path)
    y_pred = model.predict(features_test)
    mapped_labels = [labels1[i] for i in y_pred]
    df_result['Predicted_Ticket_Priority'] = mapped_labels

    # ---- Prediction for Customer Satisfaction ----
    model_path = r"model\XLNet_Embeddings_Customer Satisfaction Rating_HGB_model.pkl"
    model = joblib.load(model_path)
    y_pred = model.predict(features_test)
    mapped_labels = [labels2[i] for i in y_pred]
    df_result['Predicted_Customer Satisfaction Rating'] = mapped_labels

    # ---- Prediction for Resolution ----
    model_path = r"model\XLNet_Embeddings_Resolution_HGB_model.pkl"
    model = joblib.load(model_path)
    y_pred = model.predict(features_test)
    mapped_labels = [labels3[i] for i in y_pred]
    df_result['Predicted_Resolution'] = mapped_labels

    # ---- Print predictions row-by-row ----
    pred_cols = [c for c in df_result.columns if c.startswith("Predicted_")]

    for i, row in df_result.iterrows():
        text.insert(END, f"\n\nRow {i+1}:\n")

        # Print original (non-predicted) columns with their values
        for col in df_result.columns:
            if col not in pred_cols:
                text.insert(END, f"{col}: {row[col]}\n")

        # Print predicted values in next lines
        for col in pred_cols:
            text.insert(END, f"{col}: {row[col]}\n")

        text.insert(END, "\n")


def graph():
    text.delete('1.0', END)

    global labels1, labels2, labels3

    metrics_map = {
        "labels1": metrics_calculator_lb1,
        "labels2": metrics_calculator_lb2,
        "labels3": metrics_calculator_lb3
    }

    for label_key, calculator in metrics_map.items():

        # Copy table
        df_h = calculator.metrics_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
        df_h = df_h.round(3)

        # --- Extract target name automatically from [Target Name] ---
        first_alg_name = df_h['Algorithm'].iloc[0]
        target_name = re.search(r'\[(.*?)\]', first_alg_name).group(1)

        # --- Print title once ---
        text.insert(END, f"=== Performance for {label_key} ({target_name}) ===\n\n")

        # --- Remove […] from algorithm column ---
        df_h['Algorithm'] = df_h['Algorithm'].str.replace(r'\s*\[.*?\]', '', regex=True)

        # Print neatly
        text.insert(END, df_h.to_string(index=False))
        text.insert(END, "\n\n")
        
        
def close():
    main.destroy()


# ===============================
# GUI
# ===============================
main = tkinter.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

# LIGHT BLUE BACKGROUND
main.config(bg='#E6F2FF')

# ===============================
# TITLE
# ===============================
font=('times',20,'bold')

title = Label(main,
text='XL-NET DRIVEN TRIVARIATE CLASSIFICATION MODEL FOR HIGH PRECISION CUSTOMER SATISFACTION PREDICTION',
bg='gold2', fg='black')
title.config(font=font)
title.config(height=3,width=120)

title.place(relx=0.5,y=20,anchor='n')

# ===============================
# BUTTON FONT
# ===============================
ff=('times',14,'bold')

# GAP SETTINGS
start_y = 160
gap_y = 90
x1 = 120
x2 = 470

# ===============================
# ROW 1
# ===============================
Button(main,text="Dataset",command=upload_dataset,width=22,height=2,font=ff,bg='white').place(x=x1,y=start_y)

Button(main,text="Preprocessing",command=Preprocess_Dataset_button,width=22,height=2,font=ff,bg='white').place(x=x2,y=start_y)

# ===============================
# ROW 2
# ===============================
Button(main,text="EDA",command=eda_nlp_analysis,width=22,height=2,font=ff,bg='white').place(x=x1,y=start_y+gap_y)

Button(main,text="XLNet Feature Extraction",command=feature_extraction_button,width=22,height=2,font=ff,bg='white').place(x=x2,y=start_y+gap_y)

# ===============================
# ROW 3
# ===============================
Button(main,text="QDA Classifier",command=existing_classifier1,width=22,height=2,font=ff,bg='white').place(x=x1,y=start_y+gap_y*2)

Button(main,text="LDA Classifier",command=existing_classifier2,width=22,height=2,font=ff,bg='white').place(x=x2,y=start_y+gap_y*2)

# ===============================
# ROW 4
# ===============================
Button(main,text="SGD Classifier",command=existing_classifier3,width=22,height=2,font=ff,bg='white').place(x=x1,y=start_y+gap_y*3)

Button(main,text="NC Classifier",command=existing_classifier4,width=22,height=2,font=ff,bg='white').place(x=x2,y=start_y+gap_y*3)

# ===============================
# ROW 5
# ===============================
Button(main,text="HGB Classifier",command=Proposed_classifier,width=22,height=2,font=ff,bg='white').place(x=x1,y=start_y+gap_y*4)

Button(main,text="Prediction",command=Prediction,width=22,height=2,font=ff,bg='white').place(x=x2,y=start_y+gap_y*4)

# ===============================
# ROW 6
# ===============================
Button(main,text="Comparison Tables",command=graph,width=22,height=2,font=ff,bg='white').place(x=x1,y=start_y+gap_y*5)

Button(main,text="Exit",command=close,width=22,height=2,font=ff,bg='white').place(x=x2,y=start_y+gap_y*5)

# ===============================
# TEXT BOX
# ===============================
font1=('times',18,'bold')

text=Text(main,height=30,width=65)
scroll=Scrollbar(text)

text.configure(yscrollcommand=scroll.set)
text.place(x=850,y=160)
text.config(font=font1)

main.mainloop()