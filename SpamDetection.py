import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
data = pd.read_csv(r"C:\Users\rachi\OneDrive\Desktop\Rachitha\Projects\Email_spam_Detection_Model\spam.csv")
print(data.head())
print(data.shape)

data.drop_duplicates(inplace=True)
print(data.isnull().sum())
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

mess=data['Message']
cat=data['Category']

mess_train,mess_test,cat_train,cat_test= train_test_split(mess,cat,test_size=0.2)

cv=CountVectorizer(stop_words='english')

features = cv.fit_transform(mess_train)

#create the model
model=MultinomialNB()
model.fit(features,cat_train)

#Test the model
features_test=cv.transform(mess_test)
print(model.score(features_test,cat_test))

#predict data
def predict(message):
    input_message=cv.transform([message]).toarray()
    result=model.predict(input_message)
    return result


st.header('Email Spam Detection')
input_message = st.text_input('Enter a message')
output = st.button('Predict')

if output:
    output = predict(input_message)
    st.write(output)
#if st.button('Validate'):
    #output=predict(input_message)
    #st.markdown=output
    

#output = predict('WINNER!! This is the secret code to unlock the money: C3421.')
#print(output)


