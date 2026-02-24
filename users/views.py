from django.shortcuts import render

# Create your views here.
from django.shortcuts import render,redirect
from users.models import user
from django.http import JsonResponse


# def register(req):

#     if req.method == 'POST':
#         user_fname = req.POST.get('fname')
#         user_lname = req.POST.get('lname')
#         user_age = req.POST.get('age')
#         user_mobile = req.POST.get('mobile')
#         user_password = req.POST.get('pwd')
#         user_email = req.POST.get('email')
#         users_image = req.FILES['profile']
#         print(user_fname, user_lname, user_age, user_mobile, user_password, users_image,user_email)

#         try:
#             user.objects.get(email = user_email)
#             return redirect('register')
#         except user.DoesNotExist:
#             user.objects.create(
#                 fname = user_fname,
#                 lname = user_lname,
#                 password = user_password,
#                 age = user_age,
#                 email = user_email,
#                 mobile = user_mobile,
#                 user_profile = users_image
#             )
#             req.session ['email'] = user_email
#             return redirect('user_login')

#     return render(req, 'users/register.html')



def register(req):
    if req.method == 'POST':
        user_fname = req.POST.get('fname')
        user_lname = req.POST.get('lname')
        user_age = req.POST.get('age')
        user_mobile = req.POST.get('mobile')
        user_password = req.POST.get('pwd')
        user_email = req.POST.get('email')
        users_image = req.FILES.get('profile')  # ✅ safely get single file

        print(user_fname, user_lname, user_age, user_mobile, user_password, users_image, user_email)

        try:
            if user.objects.filter(email=user_email).exists():
                messages.error(req, 'Account already exists with this email!')
                return redirect('register')
        except Exception as e:
            print(f"Error checking user: {e}")

        try:
            user.objects.create(
                fname=user_fname,
                lname=user_lname,
                password=user_password,
                age=user_age,
                email=user_email,
                mobile=user_mobile,
                user_profile=users_image
            )
            messages.success(req, 'Account created successfully! Please wait for admin approval.')
            return redirect('user_login')
        except Exception as e:
            messages.error(req, f'Error creating account: {e}')
            return redirect('register')

    return render(req, 'users/register.html')


def user_login(req):
    if req.method == 'POST':
        user_email = req.POST.get('email') 
        user_password = req.POST.get('pwd')
        print(user_email, user_password)

        try:
            user_details = user.objects.get(email=user_email)
            print(user_details.password)
            if user_details.password == user_password:
                req.session['user_id'] = user_details.user_id  # store the user ID
                print("hello")
                if user_details.user_status.strip().capitalize() == 'Accepted':
                    print('Login successful and session created.')
                    return redirect('user_console')
                else:
                    print('Invalid: user not accepted.')

            else:
                print('Invalid login details')
        except user.DoesNotExist:
            return redirect('user_login')
        
    return render(req, 'users/user_login.html')


def user_console(req):
        return render(req, 'users/user_console.html')

from django.contrib import messages


def user_profile(request):
    views_id = request.session['user_id']
    users = user.objects.get(user_id = views_id)
    if request.method =='POST':
        userfname = request.POST.get('f_name')
        userlname = request.POST.get('l_name')
        email = request.POST.get('email_address')
        phone = request.POST.get('Phone_number')
        password = request.POST.get('pass')
        age = request.POST.get('age')
        print(userfname, userlname, email, phone, password, age)

        users.fname = userfname
        users.lname = userlname
        users.email = email
        users.mobile = phone
        users.password = password
        users.age = age

        if len(request.FILES)!= 0:
            image = request.FILES['image']
            users.user_profile = image
            users.fname = userfname
            users.lname = userlname
            users.email = email
            users.mobile = phone
            users.password = password
            users.age = age

            users.save()
            messages.success(request, 'Updated Successfully...!')

        else:
            users.fname = userfname
            users.lname = userlname
            users.email = email
            users.mobile = phone
            users.password = password
            users.age = age
            users.save()
            messages.success(request, 'Updated Successfully...!')

    return render(request,'users/user_profile.html', {'i':users})


# import pickle
# import numpy as np
# from pathlib import Path

# from django.shortcuts import render

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
#  # <-- important

# import pickle


# # ——— Load model artifacts at import time ———
# BASE_DIR             = Path(__file__).resolve().parent.parent
# CHAR_MODEL_PATH      = BASE_DIR / 'D:/fake_news/dataset/news_characterizer.h5'
# COORD_MODEL_PATH     = BASE_DIR / 'D:/fake_news/dataset/ensemble_coordinator.h5'
# TRUTH_PREDICTOR_PATH = BASE_DIR / 'D:/fake_news/dataset/truth_predictor.pkl'
# TOKENIZER_PATH       = BASE_DIR / 'D:/fake_news/dataset/tokenizer.pkl'
# METADATA_PATH        = BASE_DIR / 'D:/fake_news/dataset/model_metadata.pkl'

# news_characterizer   = load_model(str(CHAR_MODEL_PATH))
# ensemble_coordinator = load_model(str(COORD_MODEL_PATH))

# with open(str(TRUTH_PREDICTOR_PATH), 'rb') as f:
#     truth_predictor = pickle.load(f)

# with open(str(TOKENIZER_PATH), 'rb') as f:
#     tokenizer = pickle.load(f)

# with open(str(METADATA_PATH), 'rb') as f:
#     metadata = pickle.load(f)

# MAX_LEN = metadata['max_len']


# def prediction(request):
#     """
#     GET  – show form
#     POST – run models and render scores + verdict only
#     """
#     context = {}
#     if request.method == 'POST':
#         title   = request.POST.get('title', '').strip()
#         content = request.POST.get('content', '').strip()

#         # Preprocess and predict
#         seq            = tokenizer.texts_to_sequences([f"{title} {content}"])
#         padded         = pad_sequences(seq, maxlen=MAX_LEN)
#         features       = news_characterizer.predict(padded, verbose=0)
#         coord_score    = float(ensemble_coordinator.predict(features, verbose=0)[0][0])
#         truth_score    = float(truth_predictor.predict(features)[0])
#         combined_score = (coord_score + truth_score) / 2
#         prediction     = 'REAL' if combined_score > 0.5 else 'FAKE'

#         # Only pass scores and verdict
#         context = {
#             'coordinator_score': f"{coord_score:.4f}",
#             'truth_score':       f"{truth_score:.4f}",
#             'combined_score':    f"{combined_score:.4f}",
#             'prediction':        prediction,
#         }

#     return render(request, 'users/prediction.html', context)





import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'dataset'
CHAR_MODEL_PATH = DATASET_DIR / 'news_characterizer.h5'
COORD_MODEL_PATH = DATASET_DIR / 'ensemble_coordinator.h5'
TRUTH_PREDICTOR_PATH = DATASET_DIR / 'truth_predictor.pkl'
TOKENIZER_PATH = DATASET_DIR / 'tokenizer.pkl'
METADATA_PATH = DATASET_DIR / 'model_metadata.pkl'

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Component 3: Truth Predictor (must be defined for pickle to load it)
class TruthPredictor:
    def __init__(self):
        self.models = [
            RandomForestClassifier(n_estimators=100),
            SVC(probability=True)
        ]
    
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        return np.mean(predictions, axis=0)

# 🛠️ Fix for Pickle: Assign to __main__ so it can be deserialized
import sys
import __main__
__main__.TruthPredictor = TruthPredictor

# Pre-load components to memory for performance
try:
    print(f"Loading models from {DATASET_DIR}...")
    news_characterizer = load_model(str(CHAR_MODEL_PATH))
    ensemble_coordinator = load_model(str(COORD_MODEL_PATH))
    
    with open(str(TRUTH_PREDICTOR_PATH), 'rb') as f:
        truth_predictor = pickle.load(f)

    with open(str(TOKENIZER_PATH), 'rb') as f:
        tokenizer = pickle.load(f)

    with open(str(METADATA_PATH), 'rb') as f:
        metadata = pickle.load(f)

    MAX_LEN = metadata.get('max_len', 500)
    from datetime import datetime
    print(f"[SUCCESS] AI Models loaded successfully at {datetime.now()}")
except Exception as e:
    print(f"[ERROR] Could not load AI models.")
    import traceback
    traceback.print_exc()
    news_characterizer = None
    ensemble_coordinator = None
    truth_predictor = None
    tokenizer = None
    MAX_LEN = 500

def prediction(request):
    context = {}
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()

        if not (tokenizer and news_characterizer and ensemble_coordinator):
            print(f"DEBUG: T={bool(tokenizer)} NC={bool(news_characterizer)} EC={bool(ensemble_coordinator)}")
            context = {'error': 'AI Model System is currently offline. Please contact admin.'}
            return render(request, 'users/prediction.html', context)

        try:
            # Preprocess text
            text = f"{title} {content}"
            seq = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=MAX_LEN)

            # Predict using models
            features = news_characterizer.predict(padded, verbose=0)
            coord_score = float(ensemble_coordinator.predict(features, verbose=0)[0][0])
            
            # Use truth_predictor if available
            if truth_predictor:
                truth_score = float(truth_predictor.predict(features)[0])
                combined_score = (coord_score + truth_score) / 2
            else:
                combined_score = coord_score
            
            # Label based on combined score
            prediction_label = 'REAL' if combined_score > 0.5 else 'FAKE'

            context = {
                'coordinator_score': f"{combined_score:.2%}",
                'prediction': prediction_label,
                'title': title,
                'content': content
            }
        except Exception as e:
            print(f"Prediction Error: {e}")
            context = {'error': f'An error occurred during analysis: {e}'}

    return render(request, 'users/prediction.html', context)

