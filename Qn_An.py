import os
#os.system("pip install allennlp==2.1.0 allennlp-models==2.1.0")
#os.system ("pip install git+https://github.com/explosion/spacy-transformers")
os.system("pip install git+https://github.com/explosion/spacy-transformers")
#import allennlp
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
import streamlit as st
from PIL import Image
image = Image.open('kiitec logo.png')
#html = "<img src= "kiitec logo.png" >"
#css = "img{diplay:relative,left:350}"
st.image(image,  width=None)
st.title('''KIITEC VIRTUAL ASSISTANT''')
checkpoint = "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo.2021-02-11.tar.gz"
#checkpoint = "hf://lysandre/bidaf-elmo-model-2020.03.19"
predictor = Predictor.from_path(checkpoint)
predictions = predictor.predict_json({
  "passage":
      #"The Matrix is a 1999 science fiction action "
      #"film written and directed by The Wachowskis, "
      #"starring Keanu Reeves, Laurence Fishburne, "
      #"Carrie-Anne Moss, Hugo Weaving, and Joe P"
      #"antoliano.\n"
      "KIITEC is a technical institution registered by"
      "NACTE (REG/EOS/027) based in Moshono,"
      "Arusha next to Masai Camp."\n
      "Fee structure and Mode of Payment for Diploma Programmes,"
      "for first semister is 695,000Tsh can be paid in two installments"
      "before the end of the semester, and for second semister is 625,000Tsh,"
      "Fee in the second semester can be paid in two installments before the end of the semester.\n"
      "The fees should be paid through the BANK of  ABSA"
      "and the Account number is  002-4001687 "
      "the Account Name is  KIITEC Ltd.\n",
  "question":st.text_input('Question', 'what is Kiitec')
})
st.write(predictions["best_span_str"])
