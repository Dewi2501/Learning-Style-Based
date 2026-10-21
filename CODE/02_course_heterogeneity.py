import csv, numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'DATA'
OUTPUT_DIR = ROOT / 'OUTPUT'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
p = DATA_DIR / 'compiled_pendikom_moodle_logs.csv'
lines=[]
for line in open(p,encoding='utf-8-sig'):
 s=line.rstrip('\r\n'); s=s[1:-1] if len(s)>1 and s[0]=='"' and s[-1]=='"' else s; lines.append(s)
header=next(csv.reader([lines[0]])); rows=[next(csv.reader([s])) for s in lines[1:]]
df=pd.DataFrame(rows,columns=header)
for c in df.columns:
 cv=pd.to_numeric(df[c],errors='coerce')
 if cv.notna().sum()>=int(.9*len(df)): df[c]=cv
df['tie_on_total_activity']=df['tie_on_total_activity'].map({'True':1,'False':0}).fillna(0).astype(int)
D=df[df.has_ils==1].copy().reset_index(drop=True); D['activity_span_days']=(D.last_activity-D.first_activity)/86400
beh=header[4:55]; opp=['course_weeks','content_available','content_stay_reference_seconds','question_available','assign_available','exercise_stay_reference_seconds','overview_stay_reference_seconds','ques_detail_available','ques_overview_available','ques_facts_available','ques_concepts_available','ques_graphics_available','ques_text_available','ques_interpret_available','ques_develop_available']
raw=[c for c in beh if c not in set(['first_activity','last_activity','tie_on_total_activity']+opp)]+['activity_span_days']
# eta squared by course
def eta2(x,g):
 x=np.asarray(x,float); gm=np.mean(x); den=np.sum((x-gm)**2)
 if den<=0: return 0.0
 num=0
 for k in np.unique(g):
  z=x[g==k]; num+=len(z)*(np.mean(z)-gm)**2
 return num/den
etas=[{'feature':c,'eta2_course':eta2(D[c].values,D.courseid.values)} for c in raw]
etadf=pd.DataFrame(etas).sort_values('eta2_course',ascending=False)
etadf.to_csv(OUTPUT_DIR / 'course_eta_squared.csv',index=False)
# opportunity heterogeneity: number unique and range by course
opp_rows=[]
for c in opp:
 vals=D.groupby('courseid')[c].first()
 opp_rows.append({'feature':c,'n_unique_across_courses':int(vals.nunique()),'min':float(vals.min()),'median':float(vals.median()),'max':float(vals.max())})
pd.DataFrame(opp_rows).to_csv(OUTPUT_DIR / 'opportunity_heterogeneity.csv',index=False)
# course classifier on courses >=5 labeled learners
cnt=D.courseid.value_counts(); keep=cnt[cnt>=5].index
S=D[D.courseid.isin(keep)].copy(); X=S[raw].astype(float).values; y=S.courseid.astype(str).values
cv=StratifiedKFold(5,shuffle=True,random_state=42)
models={
 'LR':Pipeline([('log',FunctionTransformer(np.log1p)),('sc',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',solver='lbfgs'))]),
 'RF':RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=1,class_weight='balanced_subsample',random_state=42,n_jobs=1)
}
out=[]
for name,m in models.items():
 pred=cross_val_predict(m,X,y,cv=cv,n_jobs=1,method='predict')
 out.append({'model':name,'n_courses':len(keep),'n_learners':len(S),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'chance_accuracy':1/len(keep)})
pd.DataFrame(out).to_csv(OUTPUT_DIR / 'course_identity_diagnostic.csv',index=False)
print('eta2 median',etadf.eta2_course.median(),'top',etadf.head(10).to_dict('records'))
print(pd.DataFrame(out).to_string(index=False))
