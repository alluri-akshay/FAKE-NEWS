from django.shortcuts import render,redirect

# Create your views here.
from django.contrib import messages









from django.shortcuts import render,redirect
from django.contrib import messages
from  users.models import user

def admin_login(request):
    a_email = 'admin@gmail.com'
    a_password = 'admin'

    if request.method == 'POST':
        admin_name = request.POST.get('email')
        admin_password = request.POST.get('pwd')

        if a_email == admin_name and a_password == admin_password:
            messages.success(request, 'Login successful')
            print('Admin login successful...')
            return redirect('admin_console')
        else:
            messages.error(request, 'Login credentials were incorrect.')
            print('Invalid login attempt...')
            return redirect('admin_login')

    return render(request, 'admin/admin_login.html')


def admin_console(req):
    all_users = user.objects.all().count()
    pending_users = user.objects.filter(user_status = 'pending').count()
    rejected_users = user.objects.filter(user_status = 'Rejected').count()
    accepted_users = user.objects.filter(user_status = 'Accepected ').count()
    messages.success(req, 'Login Succesfully')
    return render(req, 'admin/admindashboard.html', {'a' : pending_users, 'b' : all_users, 'c' : rejected_users, 'd' : accepted_users})

def pending_users(req):
    users = user.objects.filter(user_status = 'pending')
    context = {'u' : users}
    return render(req, 'admin/pending_users.html', context)

def accepted_users(req, id):
    return redirect('pending_users')

def reject_user(req,id):
    return redirect('pending_users')

def delete_users(req, id):
    return redirect('all_users')

from django.core.paginator import Paginator

def all_users(request):
    a = user.objects.all()
    paginator = Paginator(a, 5) 
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request,'admin/all_users.html',{'u':post})


def Admin_Accept_Button(request, id):
    users = user.objects.get(user_id=id)
    users.user_status = "Accepted"
    users.save()
    messages.success(request, "Status Changed Successfully")
    messages.warning(request, "Accepted")
    return redirect('pending_users')

def Admin_Reject_Btn(request, id):
    users = user.objects.get(user_id=id)
    users.user_status = "Rejected"
    users.save()
    messages.success(request, "Status Changed Successfully")
    messages.warning(request, "Rejected")
    return redirect('pending_users')


# def fned_model(req):
#     return render(req,'admin/fned_model')




# users/views.py

from django.shortcuts import render
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset and preprocess
df = pd.read_csv('D:/7/fake_news/dataset/fake_news_dataset.csv')
df['text'] = df['Article Title'] + ' ' + df['Content']
X = df['text']
y = df['Label']

# Split data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization
max_words = 8000
max_len = 150
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_raw)
X_train_seq = tokenizer.texts_to_sequences(X_train_raw)
X_test_seq = tokenizer.texts_to_sequences(X_test_raw)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# CNN characterizer
def build_simple_characterizer():
    input_layer = Input(shape=(max_len,))
    embedding = Embedding(input_dim=max_words, output_dim=64)(input_layer)
    conv = Conv1D(filters=32, kernel_size=5, activation='relu')(embedding)
    pool = GlobalMaxPooling1D()(conv)
    output = Dense(64, activation='relu')(pool)
    return Model(inputs=input_layer, outputs=output)

class SimpleTruthPredictor:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

# Build and evaluate
characterizer = build_simple_characterizer()
X_train_features = characterizer.predict(X_train_pad)
X_test_features = characterizer.predict(X_test_pad)

truth_predictor = SimpleTruthPredictor()
truth_predictor.fit(X_train_features, y_train)
probs = truth_predictor.predict(X_test_features)
preds = (probs > 0.5).astype(int)

# Evaluation metrics
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

# View function (no prediction)
def fned(request):
    context = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }
    return render(request, 'admin/fned.html', context)


def cka(req):
    return render(req,'admin/cka.html')




# users/views.py

import os
import pandas as pd
import numpy as np
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ——— Configuration ———
MAX_WORDS = 10000
MAX_LEN   = 200

# **Absolute path to your dataset**:
CSV_PATH  = r"D:\7\fake_news\dataset\fake_news_dataset.csv"


def build_news_characterizer():
    inp    = Input(shape=(MAX_LEN,))
    embed  = Embedding(input_dim=MAX_WORDS, output_dim=128)(inp)
    c1     = Conv1D(64, 3, activation='relu')(embed)
    p1     = GlobalMaxPooling1D()(c1)
    c2     = Conv1D(64, 5, activation='relu')(embed)
    p2     = GlobalMaxPooling1D()(c2)
    c3     = Conv1D(64, 7, activation='relu')(embed)
    p3     = GlobalMaxPooling1D()(c3)
    merged = concatenate([p1, p2, p3])
    out    = Dense(128, activation='relu')(merged)
    return Model(inputs=inp, outputs=out)


def build_ensemble_coordinator(input_dim):
    inp = Input(shape=(input_dim,))
    x   = Dense(128, activation='relu')(inp)
    x   = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=out)


class TruthPredictor:
    def __init__(self):
        self.models = [
            RandomForestClassifier(n_estimators=100),
            SVC(probability=True)
        ]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)

    def predict(self, X):
        # average the positive-class probabilities
        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        return np.mean(probs, axis=0)


def cka(request):
    # 1) Check dataset file
    if not os.path.isfile(CSV_PATH):
        return render(request, 'admin/cka.html', {
            'error': f"Dataset not found at {CSV_PATH}"
        })

    # 2) Load & preprocess
    df = pd.read_csv(CSV_PATH)
    df['text'] = df['Article Title'].fillna('') + ' ' + df['Content'].fillna('')
    X, y = df['text'], df['Label']

    # 3) Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Tokenize & pad
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_tr)
    seq_tr = tokenizer.texts_to_sequences(X_tr)
    seq_te = tokenizer.texts_to_sequences(X_te)
    pad_tr = pad_sequences(seq_tr, maxlen=MAX_LEN)
    pad_te = pad_sequences(seq_te, maxlen=MAX_LEN)

    # 5) Feature extraction (CNN)
    char_model = build_news_characterizer()
    feat_tr    = char_model.predict(pad_tr)
    feat_te    = char_model.predict(pad_te)

    # 6) Train ensemble coordinator
    coord = build_ensemble_coordinator(feat_tr.shape[1])
    coord.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    coord.fit(feat_tr, y_tr, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

    # 7) Train truth predictor
    tp = TruthPredictor()
    tp.fit(feat_tr, y_tr)

    # 8) Predict & ensemble
    coord_prob = coord.predict(feat_te).flatten()
    truth_prob = tp.predict(feat_te)
    final_pred = np.round((coord_prob + truth_prob) / 2).astype(int)

    # 9) Compute metrics
    acc = accuracy_score(y_te, final_pred)
    pre = precision_score(y_te, final_pred)
    rec = recall_score(y_te, final_pred)
    f1m = f1_score(y_te, final_pred)

    # 10) Pass metrics in context
    context = {
        'accuracy':  f"{acc:.4f}",
        'precision': f"{pre:.4f}",
        'recall':    f"{rec:.4f}",
        'f1_score':  f"{f1m:.4f}",
    }
    return render(request, 'admin/cka.html', context)






import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from django.shortcuts import render

def graph(req):
    # Models and their metrics
    models = ['FNED', 'CKA']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = {
      
        
        'FNED': [70, 63, 100, 77],
        'CKA': [100	,100, 100, 100],
    }

    # Bar width
    bar_width = 0.15
    index = np.arange(len(metrics))

    # Plot each metric as a bar chart for each model
    plt.figure(figsize=(12, 8))
    for i, model in enumerate(models):
        values = scores[model]
        plt.bar(index + i * bar_width, values, bar_width, label=model)

    # Add labels, legend, and title
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title('Comparison of Metrics Across Models', fontsize=16)
    plt.xticks(index + bar_width * (len(models) / 2), metrics, fontsize=12)
    plt.legend()

    # Annotate each bar with its value
    for i, model in enumerate(models):
        for j, value in enumerate(scores[model]):
            plt.text(index[j] + i * bar_width, value + 0.01, f'{value:.2f}', ha='center', fontsize=10)

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    # Provide image data for template
    context = {
        'image_data': image_data
    }

    return render(req, 'admin/graph.html', context)

