#!/usr/bin/env python
# coding: utf-8

# # Lab 4_1_ Развертывание веб-приложения машинного обучения 

# # Streamlit
# 
# Streamlit - это быстрый способ создать приложение для обработки данных
# 
# Допустим, вам нужно быстро создать приложение для обработки данных. Он может прогуляться в себя панель инструментов и работать с моделью. Или, если вам нужен быстрый прототип для демонстрации заказчику, что можно сделать за несколько часов и бесплатно.
# 
# Streamlit идеально подходит для этих задач. 
# 
# ВСделаем очень простой дашборд с выбором типа ирисов и построением гистограмм. Также в Streamlit вы можете обучать и создавать модели вывода.

# In[2]:


### УСТАНОВКА
##!pip install streamlit


# В этих строках мы импортируем библиотеки streamlit и pandas, назначая им, соответственно, псевдонимы st и pd. Мы, кроме того, импортируем пакет datasets из библиотеки scikit-learn (sklearn). Мы воспользуемся этим пакетом ниже, в команде iris = datasets.load_iris(), для загрузки интересующего нас набора данных. И наконец, тут мы импортируем функцию RandomForestClassifier() из пакета sklearn.ensemble.

# In[2]:


import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


# В этой строке мы описываем заголовок боковой панели, используя функцию st.sidebar.header(). Обратите внимание на то, что тут sidebar стоит между st и header(), что и даёт полное имя функции st.sidebar.header(). Эта функция сообщает библиотеке streamlit о том, что мы хотим поместить заголовок в боковую панель.

# In[3]:


st.sidebar.header('User Input Parameters')


# In[ ]:





# Здесь мы объявляем функцию user_input_features(), которая берёт данные, введённые пользователем (то есть — четыре характеристики цветка, которые вводятся с использованием ползунков), и возвращает результат в виде датафрейма. Стоит отметить, что каждый входной параметр вводится в систему с помощью ползунка. Например, ползунок для ввода длины чашелистика (sepal length) описывается так: st.sidebar.slider(‘Sepal length’, 4.3, 7.9, 5.4). Первый из четырёх входных аргументов этой функции задаёт подпись ползунка, выводимую выше него. Это, в данном случае, текст Sepal length. Два следующих аргумента задают минимальное и максимальное значения, которые можно задавать с помощью ползунка. Последний аргумент задаёт значение, выставляемое на ползунке по умолчанию, при загрузке страницы. Здесь это — 5.4.

# In[4]:


def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features


# In[5]:


df = user_input_features()


# Здесь датафрейм, сформированный функцией user_input_features() записываем в переменную df.

# In[6]:


iris = datasets.load_iris()


# Загрузка набора данных Iris из пакета sklearn.datasets и запись его в переменную iris.

# Создание переменной Х, содержащей сведения о 4 характеристиках цветка, которые имеются в iris.data.
# 
# 
# Создание переменной Y, которая содержит сведения о виде цветка. Эти сведения хранятся в iris.target.

# In[7]:


X = iris.data
Y = iris.target


# 
# Здесь мы, пользуясь функцией RandomForestClassifier(), назначаем классификатор, основанный на алгоритме «случайный лес», переменной clf

# In[8]:


clf = RandomForestClassifier()


# Здесь мы, пользуясь функцией RandomForestClassifier(), назначаем классификатор, основанный на алгоритме «случайный лес», переменной clf.

# In[9]:


clf.fit(X, Y)


# 
# Получение сведений о виде цветка с помощью обученной модели.

# In[10]:


prediction = clf.predict(df)


# 
# Получение сведений о прогностической вероятности.

# In[11]:


prediction_proba = clf.predict_proba(df)


# # Формирование основной панели

# In[ ]:





# Здесь мы, пользуясь функцией st.write(), выводим текст. А именно, речь идёт о заголовке, выводимом в главной панели приложения, текст которого задан в формате Markdown. Символ # используется для указания того, что текст является заголовком. За строкой заголовка идёт строка обычного текста

# In[12]:


st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")


# В этой строке, пользуясь функцией st.subheader(), мы указываем подзаголовок, выводимый в основной панели. Этот подзаголовок используется для оформления раздела страницы, в котором будет выведено содержимое датафрейма, то есть того, что было введено пользователем с помощью ползунков.

# In[13]:


st.write(df)


# Этой командой мы выводим на основную панель содержимое датафрейма df.

# In[14]:


st.subheader('Class labels and their corresponding index number')


# 
# Данный код описывает второй подзаголовок основной панели. В этом разделе будут выведены данные о видах цветков.

# In[15]:


st.write(iris.target_names)


# здесь, во второй раздел основной панели, выводятся названия видов цветков (setosa, versicolor и virginica) и соответствующие им номера (0, 1, 2)

# In[16]:


st.subheader('Prediction')


# Вывод третьего подзаголовка для раздела, в котором будет находиться результат классификации.

# Вывод результата классификации. Стоит отметить, что содержимое переменной prediction — это номер вида цветка, выданный моделью на основе входных данных, введённых пользователем. Для того чтобы вывести название вида, используется конструкция iris.target_names[prediction].

# In[17]:


st.write(iris.target_names[prediction])


# Выводим заголовок четвёртого (и последнего) раздела основной панели. Здесь будут представлены данные о прогностической вероятности.
# 

# In[18]:


st.subheader('Prediction Probability')


# Вывод данных о прогностической вероятности.

# In[19]:


st.write(prediction_proba)


# # Запуск веб-приложения

# In[1]:


#streamlit run iris-ml-app.py


# In[ ]:





# ## Задание. Развернуть модель для задачи кредитного скоринга

# In[ ]:





# In[ ]:





# Просмотреть и добавить 
# 
# Создание и развертывание веб-приложений машинного обучения с использованием Pycaret, Streamlit и Heroku   
#     
#     https://analyticsindiamag.com/guide-to-building-and-deploying-ml-web-applications-using-pycaret-streamlit-and-heroku/

# Примеры
# https://leaveprediction.herokuapp.com/
#     
#     
# Установка
# 
# https://docs.streamlit.io/en/stable/installation.html
# 
# 
# Пример разработки приложения
# 
# https://docs.streamlit.io/en/stable/tutorial/create_a_data_explorer_app.html#
# 
# 
# https://ichi.pro/ru/kak-sozdat-prilozenie-dla-klassifikacii-izobrazenij-s-ispol-zovaniem-logisticeskoj-regressii-i-myslenia-nejronnoj-seti-33604303553365
# 
# 
# Как развернуть приложение
# 
# https://docs.streamlit.io/en/stable/deploy_streamlit_app.html
# 
# https://medium.com/bloggers-bay/give-life-to-your-data-science-apps-using-streamlit-9b61dfe1085d
# 
# https://nuancesprog.ru/p/5097/
# 
# https://blog.skillfactory.ru/nauka-o-dannyh-data-science/kak-napisat-veb-prilozhenie-dlya-demonstratsii-data-science-proekta-na-python/
# 
