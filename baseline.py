import csv
import numpy as np
import xgboost
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import xgboost
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
from math import log10, sqrt
import datetime
import sys

train_path = sys.argv[1]
test_path = sys.argv[2]

# const
enable_col = ['region', 'city', 'parent_category_name', 'category_name', 'user_type', 'price']
# train_path = '/tmp2/bschang/avito_kaggle_18/train.csv'
# test_path = '/tmp2/bschang/avito_kaggle_18/test.csv'
stop = stopwords.words('russian') + stopwords.words('english')
is_valid = False

# handmade feature
enable_col.extend(['param_1', 'param_2', 'param_3'])
enable_col.extend(['weekday', 'week_of_year', 'week_of_month','day_of_month'])

# remove feature
# enable_col.remove('user_type')

def to_float(var):
    if var == '':
        return np.nan
    return float(var)

def to_log(var):
    if var == '':
        return np.nan
    float_var = to_float(var)
    if float_var <= 0.0:
        return 0
    return log10(to_float(var))

def tfidf_convert(train_corpus, test_corpus):
    corpus = train_corpus + test_corpus
    vectorizer = TfidfVectorizer(stop_words=stop, smooth_idf=True, max_features=100000)

    vectorizer.fit(corpus)
    train_tfidf = vectorizer.transform(train_corpus)
    test_tfidf = vectorizer.transform(test_corpus)

    return train_tfidf, test_tfidf

def read_file(train_path, test_path):
    train = []
    train_corpus = []
    label = []
    vec = DictVectorizer()
    
    print("Load train...")
    with open(train_path) as f:
        column = f.readline().strip().split(',')
        print(column)
        for line in csv.reader(f):
            instance = {}
            for i, v in enumerate(line[:-1]):
                instance[column[i]] = v
            # handmade feature
            instance['price'] = to_float(instance['price'])
            date = datetime.datetime.strptime(instance['activation_date'], '%Y-%m-%d')
            instance['weekday'] = str(date.weekday())
            instance['week_of_month'] = str((date.day - 1) // 7 + 1)
            instance['week_of_year'] = str(date.isocalendar()[1])
            instance['day_of_month'] = str(date.day)

            train_corpus.append(instance['title'] + instance['description'])
            # filter
            instance = {k:v for k, v in instance.items() if k in enable_col} 
            train.append(instance)
            label.append(float(line[-1]))
    # print(train[:2])
    
    # read test
    test = []
    test_corpus = []
    test_id = []
    
    print("Load test...")
    with open(test_path) as f:
        column = f.readline().strip().split(',')
        for line in csv.reader(f):
            instance = {}
            for i, v in enumerate(line):
                instance[column[i]] = v
            # handmade feature
            instance['price'] = to_float(instance['price'])
            date = datetime.datetime.strptime(instance['activation_date'], '%Y-%m-%d')
            instance['weekday'] = str(date.weekday())
            instance['week_of_month'] = str((date.day - 1) // 7 + 1)
            instance['week_of_year'] = str(date.isocalendar()[1])
            instance['day_of_month'] = str(date.day)
            
            test_id.append(instance['item_id'])
            test_corpus.append(instance['title'] + instance['description'])
            # filter
            instance = {k:v for k, v in instance.items() if k in enable_col} 
            test.append(instance)

    print("Transform to one-hot encoding...")
    vec.fit(train + test)
    train_array = vec.transform(train)
    test_array = vec.transform(test)
    print("train categorical shape:", train_array.shape)
    print("test categorical shape:", test_array.shape)

    print("Transform tf-idf with title and description...")
    train_tfidf, test_tfidf = tfidf_convert(train_corpus, test_corpus)
    print("train tfidf shape:", train_tfidf.shape)
    print("test tfidf shape:", test_tfidf.shape)
    
    # merge features
    train_features = hstack([train_array, train_tfidf])
    test_features = hstack([test_array, test_tfidf])

    return train_features.tocsr(), test_features.tocsr(), test_id, np.array(label)


# ['item_id', 'user_id', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'title', 'description', 'price', 'item_seq_number', 'activation_date', 'user_type', 'image', 'image_top_1', 'deal_probability']

train_features, test_features, test_id, label = read_file(train_path, test_path)

print("train features shape:", train_features.shape)
print("test features shape:", test_features.shape)

print("Training xgb...")
model = xgboost.XGBRegressor(max_depth=20, n_jobs=4, n_estimators=100)

if is_valid:
    train_percent = 0.8
    train_fold_index = np.where(label[:int(len(label) * train_percent)])
    test_fold_index = np.where(label[int(len(label) * train_percent):])
    
    X_train = train_features[train_fold_index]
    Y_train = label[train_fold_index]
    X_test = train_features[test_fold_index]
    Y_test = label[test_fold_index]
    eval_set = [(Y_train, Y_test)]

    model.fit(X_train, Y_train, eval_metric='rmse')
    valid_pred = model.predict(X_test)
    score = sqrt(mean_squared_error(Y_test, valid_pred))
    print("validation score:", score)
    exit()

model.fit(train_features, label, eval_metric='rmse')

prob_result = model.predict(test_features)
for i, prob in enumerate(prob_result):
    if prob >= 1.0:
        prob_result[i] = 1.0
    if prob <= 0.0:
        prob_result[i] = 0.0
print(prob_result)

with open('avito_sub.txt', 'w') as f:
    f.write('item_id,deal_probability\n')
    for index, i in enumerate(test_id):
        f.write(i + ',' + str(prob_result[index]) + '\n')
