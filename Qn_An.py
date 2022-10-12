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
st.write('Sorry am still learning')
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
      #"KIITEC is a technical institution registered by "
      #"NACTE (REG/EOS/027) based in Moshono, "
      #"Arusha next to Masai Camp. "
      #"Fee structure and Mode of Payment for Diploma Programmes, "
      #"for first semister is 695,000Tsh can be paid in two installments "
      #"before the end of the semester, and for second semister is 625,000Tsh, "
      #"Fee in the second semester can be paid in two installments before the end of the semester. "
      #"The fees should be paid through the BANK of  ABSA "
      #"and the Account number is  002-4001687 "
      #"the Account Name is  KIITEC Ltd.\n" ,
      """
KIITEC is a technical institution registered by NACTE (REG/EOS/027) based in Moshono, Arusha next to Masai Camp.
Fee structure and Mode of Payment for Diploma Programmes,for first semister is 695,000Tsh can be paid in two installments before the end of the semester, and for second semister is 625,000Tsh, Fee in the second semester can be paid in two installments before the end of the semester.
The fees should be paid through the BANK of  ABSA and the Account number is  002-4001687 the Account Name is  KIITEC Ltd.
my name is Sisulu.
WALTER RICHARD made me.
hello , how are you?.
am fine ,That great.
The institute was founded in 2004 by French engineers and has thence contrived to produce the most competent technicians in the country. The institute is financed and supported by two NGO's The Foundation for Technical Education (FTE-Swiss) and Action Development Education International (ADEI-French). In 2004, ADEI's partner FTE built the Kilimanjaro International Institute of Telecommunications, Electronics and Computers (KIITEC) introducing state of the art teaching facilities to train technicians in Arusha, Tanzania.
Following construction, ADEI joined FTE in its ambition to make change through technical education and has played a pivotal role in the on the ground education programming and training at KIITEC ever since. Today, KIITEC acts as the international training center where educators travel to from different corners of Africa to upgrade their skills.
Resting on a 15-acre campus, KIITEC offers the most advanced targeted training technologies in the region. The innovative education model developed at KIITEC is based on a hands-on and student-centered approach to learning with full access to modern learning equipment simulating real world practical experiences. The training center is registered and accredited by the National Council for Technical Education (NACTE) and awards successful graduates with a 3-year National Technical Award Level Six (NTA-6) Diploma.
KIITEC’s specialized training programs or courses include: Electronics & Telecommunication Engineering , Industrial Automation , Computer Engineering and Networking , Renewable Energies, Environmental Impact .
Future training programs or courses in development: Biomedical ,  Avionics.
ADMISSION REQUIREMENTS of ELECTRONICS AND TELECOMMUNICATION ENGINEERING, Possession of a Certificate of Secondary Education Examinations (CSEE) with a minimum of FOUR passes of which TWO must be among the following subjects: Mathematics, Physics / Engineering science, Biology and Chemistry; excluding religious subjects or Possession of a National Vocational Award (NVA) Level III in Electrical, Electronics, Telecommunication, Mechanical and related field offered by VETA and a Certificate of Secondary Education Examination (CSEE) with at least two passes.
ADMISSION REQUIREMENTS of ELECTRICAL AND INDUSTRIAL AUTOMATION ENGINEERING,Possession of a Certificate of Secondary Education Examinations (CSEE) with a minimum of FOUR passes of which TWO must be among the following subjects: Mathematics, Physics / Engineering science, Biology and Chemistry; excluding religious subjects or Possession of a National Vocational Award (NVA) Level III in Electrical, Electronics, Telecommunication, Mechanical and related field offered by VETA and a Certificate of Secondary Education Examination (CSEE) with at least two passes.
ADMISSION REQUIREMENTS of ELECTRICAL AND COMPUTER ENGINEERING PROGRAMME,Possession of a Certificate of Secondary Education Examinations (CSEE) with a minimum of FOUR passes of which TWO must be among the following subjects: Mathematics, Physics / Engineering science, Biology and Chemistry; excluding religious subjects or Possession of a National Vocational Award (NVA) Level III in Electrical, Electronics, Telecommunication, Mechanical and related field offered by VETA and a Certificate of Secondary Education Examination (CSEE) with at least two passes.
ADMISSION REQUIREMENTS of ELECTRICAL AND RENEWABLE ENERGY ENGINEERING,Possession of a Certificate of Secondary Education Examinations (CSEE) with a minimum of FOUR passes of which TWO must be among the following subjects: Mathematics, Physics / Engineering science, Biology and Chemistry; excluding religious subjects or Possession of a National Vocational Award (NVA) Level III in Electrical, Electronics, Telecommunication, Mechanical and related field offered by VETA and a Certificate of Secondary Education Examination (CSEE) with at least two passes.
KIITEC STUDENTS wear uniforms , uniforms that are worn are light blue shirts and dark blue sweeters ,dark blue skirt and dark blue trouser.
Kiitec has Vision of become a leading provider of quality technical education and training to empower the youth of Tanzania and Eastern Africa region.
Kiitec has Mission of  provide quality hands-on technical training for students in ICTs, Electrical, Renewable Energies, Industrial Automation and the related disciplines, as well as to conduct quality research, and consultancy in these fields.
Also Kiitec has Mission of promoting the development and usage of modern technology that meets national, regional, and international needs and standards through skills and practical oriented training.
Kiitec has values: Hard-work and excellence ,Honesty ,Respect ,Responsibility ,Lifelong learning ,Innovation and creativity.
For contact info of kiitec Phone: +255 27 250 4384, Mobile: +255 757 845 118 , Email: info@kiitec.ac.tz P.O.Box 3172 Arusha, Tanzania.
"""
  "question":st.text_input('Question', 'what is Kiitec')
})
st.write(predictions["best_span_str"])
