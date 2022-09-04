import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.model_selection import cross_val_score
import random
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
# import sklearn.model_selection as model_selection
from imblearn.over_sampling import RandomOverSampler

#应用主题
st.set_page_config(
    page_title="ML Medicine",
    page_icon="🐇",
)
#应用标题
st.title('Machine Learning Application for Predicting Prognosis')



# conf
col1, col2, col3 = st.columns(3)
NSE = col1.number_input("NSE (ng/mL)",step=0.01,format="%.2f",value=1.45)
HCY = col2.number_input('HCY (μmol/L)',value=15.7,step=0.1,format="%.1f")
CRP = col3.number_input('CRP (mg/L)',step=0.1,format="%.1f",value=12.6)
S100β = col1.number_input("S-100β (ng/mL)",step=0.01,format="%.2f",value=1.02)
Anticoagulation = col2.selectbox("Anticoagulation",('No','Yes'))
Dysphagia = col3.selectbox("Dysphagia",('No','Yes'))


# str_to_
map = {'Left':0,'Right':1,'Bilateral':2,
       'Single stroke lesion':0,'Multiple stroke lesions':1,
       'Mild stroke':0,'Moderate to severe stroke':1,
       'Cortex':0,'Cortex-subcortex':1,'Subcortex':2,'Brainstem':3,'Cerebellum':4,
       'No':0,'Yes':1}

Anticoagulation =map[Anticoagulation]
Dysphagia =map[Dysphagia]

# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
features=['HCY','CRP','NSE', 'S100β', 'Anticoagulation','Dysphagia']
target='prognosis'

#处理数据不平衡
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

#train and predict
RF = sklearn.ensemble.RandomForestClassifier(n_estimators=32,criterion='gini',max_features='log2',max_depth=5,random_state=12)
RF.fit(X_ros, y_ros)
# XGB = XGBClassifier(random_state=32,max_depth=3,n_estimators=9)
# XGB.fit(X_ros, y_ros)
#读之前存储模型

#with open('RF.pickle', 'rb') as f:
#    RF = pickle.load(f)


sp = 0.5
#figure
is_t = (RF.predict_proba(np.array([[ HCY,CRP,NSE, S100β, Anticoagulation,Dysphagia]]))[0][1])> sp
prob = (RF.predict_proba(np.array([[HCY,CRP,NSE, S100β, Anticoagulation,Dysphagia]]))[0][1])*1000//1/10

#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability:  '+str(prob)+'%')
#st.markdown('## The risk of bone metastases is '+str(prob/0.0078*1000//1/1000)+' times higher than the average risk .')

#排版占行



# st.title("")
# st.title("")
# st.title("")
# st.title("")
#st.warning('This is a warning')
#st.error('This is an error')

#st.info('Information of the model: Auc: 0. ;Accuracy: 0. ;Sensitivity(recall): 0. ;Specificity :0. ')
#st.success('Affiliation: The First Affiliated Hospital of Nanchang University, Nanchnag university. ')





