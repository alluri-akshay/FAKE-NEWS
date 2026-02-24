import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.paginator import Paginator
from django.conf import settings
from users.models import user
from admins.models import SystemLog

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'fake_news_dataset.csv')

# --- Helper Functions for Model Evaluation ---

def get_real_metrics(model_type='FNED'):
    """Calculates real metrics from the dataset for the specified model."""
    if not os.path.exists(CSV_PATH):
        return None

    try:
        df = pd.read_csv(CSV_PATH)
        df['text'] = df['Article Title'].fillna('') + ' ' + df['Content'].fillna('')
        X, y = df['text'], df['Label']
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == 'FNED':
            # Simplified FNED (CNN + Logistic Regression)
            max_words, max_len = 8000, 150
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(X_train_raw)
            X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=max_len)
            X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test_raw), maxlen=max_len)

            # Build characterizer
            inp = Input(shape=(max_len,))
            emb = Embedding(input_dim=max_words, output_dim=64)(inp)
            conv = Conv1D(filters=32, kernel_size=5, activation='relu')(emb)
            pool = GlobalMaxPooling1D()(conv)
            feat_model = Model(inputs=inp, outputs=Dense(64, activation='relu')(pool))
            
            # Extract features
            feat_train = feat_model.predict(X_train_pad)
            feat_test = feat_model.predict(X_test_pad)

            # Predict
            clf = LogisticRegression(max_iter=1000)
            clf.fit(feat_train, y_train)
            preds = clf.predict(feat_test)

        else:  # CKA
            # Ensemble model (Multi-scale CNN + Random Forest/SVM)
            max_words, max_len = 10000, 200
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(X_train_raw)
            X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=max_len)
            X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test_raw), maxlen=max_len)

            # Build Multi-scale CNN
            inp = Input(shape=(max_len,))
            emb = Embedding(input_dim=max_words, output_dim=128)(inp)
            p1 = GlobalMaxPooling1D()(Conv1D(64, 3, activation='relu')(emb))
            p2 = GlobalMaxPooling1D()(Conv1D(64, 5, activation='relu')(emb))
            p3 = GlobalMaxPooling1D()(Conv1D(64, 7, activation='relu')(emb))
            merged = Dense(128, activation='relu')(concatenate([p1, p2, p3]))
            feat_model = Model(inputs=inp, outputs=merged)

            feat_train = feat_model.predict(X_train_pad)
            feat_test = feat_model.predict(X_test_pad)

            # Ensemble: RF + SVM
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(feat_train, y_train)
            svm = SVC(probability=True, random_state=42)
            svm.fit(feat_train, y_train)
            
            p_rf = rf.predict_proba(feat_test)[:, 1]
            p_svm = svm.predict_proba(feat_test)[:, 1]
            preds = (np.mean([p_rf, p_svm], axis=0) > 0.5).astype(int)

        return {
            'accuracy': round(accuracy_score(y_test, preds), 4),
            'precision': round(precision_score(y_test, preds), 4),
            'recall': round(recall_score(y_test, preds), 4),
            'f1_score': round(f1_score(y_test, preds), 4)
        }
    except Exception as e:
        print(f"Error evaluating {model_type}: {e}")
        return None

# --- Dashboard Views ---

def admin_login(request):
    a_email, a_password = 'admin@gmail.com', 'admin'
    if request.method == 'POST':
        if a_email == request.POST.get('email') and a_password == request.POST.get('pwd'):
            SystemLog.objects.create(log_type='ADMIN', message="Admin logged into system control panel.")
            messages.success(request, 'Login successful')
            return redirect('admin_console')
        messages.error(request, 'Login credentials were incorrect.')
    return render(request, 'admin/admin_login.html')

def admin_console(req):
    context = {
        'a': user.objects.filter(user_status__iexact='Pending').count(),
        'b': user.objects.all().count(),
        'c': user.objects.filter(user_status__iexact='Rejected').count(),
        'd': user.objects.filter(user_status__iexact='Accepted').count(),
        'recent_logs': SystemLog.objects.all().order_by('-created_at')[:5]
    }
    return render(req, 'admin/admindashboard.html', context)

def pending_users(req):
    return render(req, 'admin/pending_users.html', {'u': user.objects.filter(user_status__iexact='Pending')})

def all_users(request):
    status_filter = request.GET.get('status')
    users_list = user.objects.filter(user_status__iexact=status_filter) if status_filter else user.objects.all()
    paginator = Paginator(users_list, 5)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(request, 'admin/all_users.html', {'u': page_obj, 'status': status_filter})

# --- Model Evaluation Views ---

def fned(request):
    data = get_real_metrics('FNED')
    if not data:
        return render(request, 'admin/fned.html', {'error': f"Dataset not found at {CSV_PATH}"})
    return render(request, 'admin/fned.html', data)

def cka(request):
    data = get_real_metrics('CKA')
    if not data:
        return render(request, 'admin/cka.html', {'error': f"Dataset not found at {CSV_PATH}"})
    return render(request, 'admin/cka.html', data)

def graph(req):
    # Get model performance metrics
    # Ensure we always have a dictionary with numeric values for the template
    fned_data = get_real_metrics('FNED') or {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    cka_data = get_real_metrics('CKA') or {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}

    # Get dataset statistics
    dataset_stats = {'real': 0, 'fake': 0, 'total': 0}
    try:
        df = pd.read_csv(CSV_PATH)
        dataset_stats['total'] = len(df)
        dataset_stats['real'] = int((df['Label'] == 1).sum())
        dataset_stats['fake'] = int((df['Label'] == 0).sum())
    except Exception as e:
        print(f"Error reading dataset stats: {e}")

    # Prepare user statistics
    user_stats = {
        'pending': user.objects.filter(user_status__iexact='Pending').count(),
        'accepted': user.objects.filter(user_status__iexact='Accepted').count(),
        'rejected': user.objects.filter(user_status__iexact='Rejected').count(),
        'total': user.objects.all().count()
    }

    context = {
        'fned_data': fned_data,
        'cka_data': cka_data,
        'dataset_stats': dataset_stats,
        'user_stats': user_stats
    }

    return render(req, 'admin/graph.html', context)


def Admin_Accept_Button(request, id):
    u = user.objects.get(user_id=id)
    u.user_status = 'Accepted'
    u.save()
    SystemLog.objects.create(log_type='ADMIN', message=f"Accepted user registration for: {u.email}")
    messages.success(request, f'User {u.email} has been approved.')
    return redirect('pending_users')


def Admin_Reject_Btn(request, id):
    u = user.objects.get(user_id=id)
    u.user_status = 'Rejected'
    u.save()
    SystemLog.objects.create(log_type='ADMIN', message=f"Rejected user registration for: {u.email}")
    messages.error(request, f'User {u.email} has been rejected.')
    return redirect('pending_users')


def admin_logs(request):
    logs = SystemLog.objects.all().order_by('-created_at')
    
    # Optional filtering by type
    log_filter = request.GET.get('type')
    if log_filter:
        logs = logs.filter(log_type=log_filter)
        
    return render(request, 'admin/admin_logs.html', {'logs': logs})


def export_report(request):
    import csv
    from django.http import HttpResponse
    from users.models import NewsPrediction

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="fake_news_system_report.csv"'

    writer = csv.writer(response)
    writer.writerow(['Date', 'User', 'Article Title', 'Verdict', 'Confidence'])

    predictions = NewsPrediction.objects.all().order_by('-created_at')
    for p in predictions:
        writer.writerow([p.created_at, p.user.email, p.title, p.verdict, f"{p.confidence:.2%}"])

    return response
