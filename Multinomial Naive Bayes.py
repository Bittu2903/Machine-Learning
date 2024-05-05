from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
texts = ["This is a positive text.", "Negative sentiment in this one.", "Another positive example.",
         "What is this can you tell me.","Please do the needful","I need you to do the needful",
         "What happened to him","He is looking after him","For him this is everything", "Not a good review.", 
         "Great product!"]
labels = [1, 0, 1, 0, 1,1,0,1,1,0,1]  # 1 for positive, 0 for negative

# Create a CountVectorizer to convert the text into a Document-Term Matrix (DTM)
vectorizer = CountVectorizer()
X_dtm = vectorizer.fit_transform(texts)

# Create a Multinomial Naive Bayes classifier and fit it to the entire dataset
classifier = MultinomialNB()
classifier.fit(X_dtm, labels)

# Get predicted classes for the entire dataset
predicted_classes = classifier.predict(X_dtm)

# Display classification report
print("Classification Report:\n", classification_report(labels, predicted_classes))

# User input for prediction
user_input = input("Enter a sentence: ")

# Transform user input using the same vectorizer
user_input_dtm = vectorizer.transform([user_input])
# Predict the class for the user input
predicted_class = classifier.predict(user_input_dtm)

print("Predicted Class:", predicted_class[0])

# Use the existing labels and predictions to create a confusion matrix
conf_matrix = confusion_matrix(labels, predicted_classes)

# Displaying the confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()
