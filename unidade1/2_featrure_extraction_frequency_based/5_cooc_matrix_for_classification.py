from src.utils import load_movie_review_clean_dataset

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

corpus = load_movie_review_clean_dataset()

X = corpus['review']
y = corpus['sentiment']

vocabulary = list(set(" ".join(X).split()))
main_dic = {word: idx for idx, word in enumerate(vocabulary)}
print(main_dic)

X_train,  X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
