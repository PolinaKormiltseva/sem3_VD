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

# In[ ]:


### УСТАНОВКА
##pip install streamlit


# В этих строках мы импортируем библиотеки streamlit и pandas, назначая им, соответственно, псевдонимы st и pd. Мы, кроме того, импортируем пакет datasets из библиотеки scikit-learn (sklearn). Мы воспользуемся этим пакетом ниже, в команде iris = datasets.load_iris(), для загрузки интересующего нас набора данных. И наконец, тут мы импортируем функцию RandomForestClassifier() из пакета sklearn.ensemble.

# In[1]:


import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


# В этой строке мы описываем заголовок боковой панели, используя функцию st.sidebar.header(). Обратите внимание на то, что тут sidebar стоит между st и header(), что и даёт полное имя функции st.sidebar.header(). Эта функция сообщает библиотеке streamlit о том, что мы хотим поместить заголовок в боковую панель.

# In[2]:


st.sidebar.header('User Input Parameters')


# In[5]:





# Здесь мы объявляем функцию user_input_features(), которая берёт данные, введённые пользователем (то есть — четыре характеристики цветка, которые вводятся с использованием ползунков), и возвращает результат в виде датафрейма. Стоит отметить, что каждый входной параметр вводится в систему с помощью ползунка. Например, ползунок для ввода длины чашелистика (sepal length) описывается так: st.sidebar.slider(‘Sepal length’, 4.3, 7.9, 5.4). Первый из четырёх входных аргументов этой функции задаёт подпись ползунка, выводимую выше него. Это, в данном случае, текст Sepal length. Два следующих аргумента задают минимальное и максимальное значения, которые можно задавать с помощью ползунка. Последний аргумент задаёт значение, выставляемое на ползунке по умолчанию, при загрузке страницы. Здесь это — 5.4.

# In[8]:





# In[35]:


def user_input_features():
    RevolvingUtilizationOfUnsecuredLines = st.sidebar.slider('RevolvingUtilizationOfUnsecuredLines', 0.0, 100.0, 50708.0)
    age = st.sidebar.slider('age', 21.0, 30.0, 109.0)
    NumberOfTime30_59DaysPastDueNotWorse = st.sidebar.slider('NumberOfTime30-59DaysPastDueNotWorse', 0.0, 0.4, 98.0)
    DebtRatio = st.sidebar.slider('DebtRatio', 0.0, 352.4, 329664.0)
    MonthlyIncome = st.sidebar.slider('MonthlyIncome', 0.0, 6600.0, 3008750.0)
    NumberOfOpenCreditLinesAndLoans = st.sidebar.slider('NumberOfOpenCreditLinesAndLoans', 0.0, 8.0, 57.0)
    NumberOfTimes90DaysLate = st.sidebar.slider('NumberOfTimes90DaysLate', 0.0, 0.2, 98.0)
    NumberRealEstateLoansOrLines = st.sidebar.slider('NumberRealEstateLoansOrLines', 0.0, 1.0, 32.0)
    NumberOfTime60_89DaysPastDueNotWorse = st.sidebar.slider('NumberOfTime60-89DaysPastDueNotWorse', 0.0, 4.2, 98.0)
    NumberOfDependents = st.sidebar.slider('NumberOfDependents', 0.0, 0.7, 10.0)
    data = {'RevolvingUtilizationOfUnsecuredLines': RevolvingUtilizationOfUnsecuredLines,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': NumberOfTime30_59DaysPastDueNotWorse,
            'DebtRatio': DebtRatio,
            'MonthlyIncome': MonthlyIncome,
            'NumberOfOpenCreditLinesAndLoans': NumberOfOpenCreditLinesAndLoans,
            'NumberOfTimes90DaysLate': NumberOfTimes90DaysLate,
            'NumberRealEstateLoansOrLines': NumberRealEstateLoansOrLines,
            'NumberOfTime60-89DaysPastDueNotWorse': NumberOfTime60_89DaysPastDueNotWorse,
            'NumberOfDependents': NumberOfDependents}
    features = pd.DataFrame(data, index=[0])
    return features


# In[36]:


df_user = user_input_features()


# Здесь датафрейм, сформированный функцией user_input_features() записываем в переменную df.

# In[17]:


train_data = pd.read_csv("training_data.csv")


# In[30]:


#train_data.shape


# In[18]:


#train_data.head()


# In[19]:


#train_data.describe()


# Загрузка набора данных Iris из пакета sklearn.datasets и запись его в переменную iris.

# Создание переменной Х, содержащей сведения о 4 характеристиках цветка, которые имеются в iris.data.
# 
# 
# Создание переменной Y, которая содержит сведения о виде цветка. Эти сведения хранятся в iris.target.

# In[21]:


#train_data.info()


# In[23]:


#заполняем пропуски с помощью метода средних
train_mean = train_data.mean()
train_mean


# In[24]:


train_data.fillna(train_mean, inplace=True)


# In[25]:


#train_data.info()


# In[26]:


Y = train_data.SeriousDlqin2yrs


# In[27]:


#Y.shape


# In[28]:


X = train_data.drop('SeriousDlqin2yrs', axis=1)
#X.shape


# 
# Здесь мы, пользуясь функцией RandomForestClassifier(), назначаем классификатор, основанный на алгоритме «случайный лес», переменной clf

# In[31]:


clf = RandomForestClassifier()


# Здесь мы, пользуясь функцией RandomForestClassifier(), назначаем классификатор, основанный на алгоритме «случайный лес», переменной clf.

# In[32]:


clf.fit(X, Y)


# 
# Получение сведений о виде цветка с помощью обученной модели.

# In[37]:


prediction = clf.predict(df_user)


# 
# Получение сведений о прогностической вероятности.

# In[38]:


prediction_proba = clf.predict_proba(df_user)


# # Формирование основной панели

# In[ ]:





# Здесь мы, пользуясь функцией st.write(), выводим текст. А именно, речь идёт о заголовке, выводимом в главной панели приложения, текст которого задан в формате Markdown. Символ # используется для указания того, что текст является заголовком. За строкой заголовка идёт строка обычного текста

# In[39]:


st.write("""
# Simple Credid Scoring App
This app predicts the **Scoring** type!
""")


# В этой строке, пользуясь функцией st.subheader(), мы указываем подзаголовок, выводимый в основной панели. Этот подзаголовок используется для оформления раздела страницы, в котором будет выведено содержимое датафрейма, то есть того, что было введено пользователем с помощью ползунков.

# In[40]:


st.write(df_user)


# Этой командой мы выводим на основную панель содержимое датафрейма df.

# In[41]:


st.subheader('Will there be a delay of more than 90 days')


# 
# Данный код описывает второй подзаголовок основной панели. В этом разделе будут выведены данные о видах цветков.

# In[42]:


st.write(train_data.SeriousDlqin2yrs)


# здесь, во второй раздел основной панели, выводятся названия видов цветков (setosa, versicolor и virginica) и соответствующие им номера (0, 1, 2)

# In[43]:


st.subheader('Prediction')


# Вывод третьего подзаголовка для раздела, в котором будет находиться результат классификации.

# Вывод результата классификации. Стоит отметить, что содержимое переменной prediction — это номер вида цветка, выданный моделью на основе входных данных, введённых пользователем. Для того чтобы вывести название вида, используется конструкция iris.target_names[prediction].

# In[44]:


st.write(train_data.SeriousDlqin2yrs[prediction])


# Выводим заголовок четвёртого (и последнего) раздела основной панели. Здесь будут представлены данные о прогностической вероятности.
# 

# In[45]:


st.subheader('Prediction Probability')


# Вывод данных о прогностической вероятности.

# In[46]:


st.write(prediction_proba)


# # Запуск веб-приложения

# In[ ]:


##streamlit run iris-ml-app.py


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
