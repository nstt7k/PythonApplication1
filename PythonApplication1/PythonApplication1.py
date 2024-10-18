
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


def load_data(data_dir, label):
  """Загружает данные из заданного каталога."""
  reviews = []
  ratings = []
  for filename in os.listdir(os.path.join(data_dir, label)):
    with open(os.path.join(data_dir, label, filename), 'r', encoding='utf-8') as f:
      review = f.read()
      id, rating = filename.split('_')[:-1] # Извлечение ID и рейтинга из имени файла
      reviews.append(review)
      ratings.append(int(rating))
  return pd.DataFrame({'comment': reviews, 'rating': ratings})

# Загрузка обучающих данных
train_pos = load_data('train/pos', 'pos')
train_neg = load_data('train/neg', 'neg')
train_data = pd.concat([train_pos, train_neg])

# Загрузка тестовых данных
test_pos = load_data('test/pos', 'pos')
test_neg = load_data('test/neg', 'neg')
test_data = pd.concat([test_pos, test_neg])

# Создание бинарного признака "положительный"
train_data['label'] = train_data['rating'].apply(lambda x: 1 if x >= 6 else 0)
test_data['label'] = test_data['rating'].apply(lambda x: 1 if x >= 6 else 0)

# Векторизация текста
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['comment'])
X_test = vectorizer.transform(test_data['comment'])
y_train = train_data['label']
y_test = test_data['label']

# Обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')

# Сохранение модели

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))



def load_libsvm_data(filename):
  """Загружает данные в формате LIBSVM."""
  data = []
  with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split(' ')
      label = int(parts[0])
      features = {int(i.split(':')[0]): int(i.split(':')[1]) for i in parts[1:]}
      data.append((label, features))
  return data

def convert_to_sparse_matrix(data, vocab_size):
  """Преобразует данные в разреженную матрицу."""
  from scipy.sparse import csr_matrix
  rows = []
  cols = []
  data_values = []
  for i, (label, features) in enumerate(data):
    for feature, count in features.items():
      rows.append(i)
      cols.append(feature)
      data_values.append(count)
  return csr_matrix((data_values, (rows, cols)), shape=(len(data), vocab_size))

# Загрузка данных
train_data = load_libsvm_data('train/train.feat')
test_data = load_libsvm_data('test/test.feat')
vocab_size = len(open('imdb.vocab', 'r', encoding='utf-8').readlines())

# Преобразование в разреженные матрицы
X_train = convert_to_sparse_matrix(train_data, vocab_size)
X_test = convert_to_sparse_matrix(test_data, vocab_size)
y_train = [label for label, features in train_data]
y_test = [label for label, features in test_data]

# Обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')

# Сохранение модели
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))




# Загрузка модели
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@csrf_exempt
def predict(request):
  if request.method == 'POST':
    comment = request.POST.get('comment')
    if comment:
      vector = vectorizer.transform([comment])
      prediction = model.predict(vector)[0]
      rating = prediction * 10 # Преобразование бинарного предсказания в рейтинг
      if prediction == 1:
        status = "Положительный"
      else:
        status = "Отрицательный"
      return HttpResponse(f'Рейтинг: {rating}, Статус: {status}')
  return render(request, 'review_app/predict.html')