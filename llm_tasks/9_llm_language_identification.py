import fasttext

model = fasttext.load_model("models/lid.176.ftz")

text = "I am learning how to use FastText for language identification."
pred = model.predict(text)
print(pred)
