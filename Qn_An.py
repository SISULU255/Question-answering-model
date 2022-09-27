import os
os.system("pip install git+https://github.com/explosion/spacy-transformers")
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
import streamlit as st
checkpoint = "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo.2021-02-11.tar.gz"
#checkpoint = "hf://lysandre/bidaf-elmo-model-2020.03.19"
st.title('''Kiitec Virtual Assistance''')
predictor = Predictor.from_path(checkpoint)
predictions = predictor.predict_json({
  "passage":
      "The Matrix is a 1999 science fiction action "
      "film written and directed by The Wachowskis, "
      "starring Keanu Reeves, Laurence Fishburne, "
      "Carrie-Anne Moss, Hugo Weaving, and Joe P"
      "antoliano.",
  "question":
      st.input("")
      
})
st.write(predictions["best_span_str"])
