import csv,json,warnings,os,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold,GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score,matthews_corrcoef,roc_auc_score,accuracy_score
from xgboost import XGBClassifier
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
df['tie_on_total_activity']=df['tie_on_total_activity'].map({'True':1,'False':0,True:1,False:0}).fillna(0).astype(int)
D=df[df.has_ils==1].copy().reset_index(drop=True); D['activity_span_days']=(D.last_activity-D.first_activity)/86400
scores=['ils_active_reflective_score','ils_sensing_intuitive_score','ils_visual_verbal_score','ils_sequential_global_score']; names=['AR','SI','VV','SG']
Y=np.column_stack([(pd.to_numeric(D[c])>0).astype(int) for c in scores])
beh=header[4:55]; oppc=['course_weeks','content_available','content_stay_reference_seconds','question_available','assign_available','exercise_stay_reference_seconds','overview_stay_reference_seconds','ques_detail_available','ques_overview_available','ques_facts_available','ques_concepts_available','ques_graphics_available','ques_text_available','ques_interpret_available','ques_develop_available']
raw=[c for c in beh if c not in set(['first_activity','last_activity','tie_on_total_activity']+oppc)]+['activity_span_days']
ratio={'content_visit_per_available':('content_visit_evidence_count','content_available'),'content_time_ratio':('content_stay_seconds','content_stay_reference_seconds'),'question_attempt_per_available':('question_attempt_count','question_available'),'assignment_submit_per_available':('submitted_assignment_count','assign_available'),'exercise_time_ratio':('exercise_stay_seconds','exercise_stay_reference_seconds'),'overview_time_ratio':('overview_stay_seconds','overview_stay_reference_seconds'),'detail_attempt_per_available':('ques_detail_attempted_count','ques_detail_available'),'overview_attempt_per_available':('ques_overview_attempted_count','ques_overview_available'),'facts_attempt_per_available':('ques_facts_attempted_count','ques_facts_available'),'concepts_attempt_per_available':('ques_concepts_attempted_count','ques_concepts_available'),'graphics_attempt_per_available':('ques_graphics_attempted_count','ques_graphics_available'),'text_attempt_per_available':('ques_text_attempted_count','ques_text_available'),'interpret_attempt_per_available':('ques_interpret_attempted_count','ques_interpret_available'),'develop_attempt_per_available':('ques_develop_attempted_count','ques_develop_available'),'quiz_result_avg_stay':('quiz_results_stay_seconds_sum','quiz_results_event_count')}
for k,(a,b) in ratio.items(): D[k]=np.log1p(D[a].astype(float)/(D[b].astype(float)+1e-6))
opp=raw+list(ratio)
strat=np.array([f'{a}{b}' for a,b in Y[:,:2]]); rs=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(D)),strat)); gs=list(GroupKFold(5).split(np.zeros(len(D)),groups=D.courseid.values))
rowsout=[]; predout=[]
def models(y):
 neg=(y==0).sum(); pos=(y==1).sum(); spw=neg/max(pos,1)
 return {
 'LR':Pipeline([('log',FunctionTransformer(np.log1p)),('sc',StandardScaler()),('clf',LogisticRegression(C=1,max_iter=2000,class_weight='balanced',solver='liblinear',random_state=42))]),
 'SVM':Pipeline([('log',FunctionTransformer(np.log1p)),('sc',StandardScaler()),('clf',SVC(C=2,gamma='scale',probability=True,class_weight='balanced',random_state=42))]),
 'RF':RandomForestClassifier(n_estimators=150,max_depth=8,min_samples_leaf=3,max_features='sqrt',class_weight='balanced_subsample',random_state=42,n_jobs=1),
 'XGB':XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='logloss',scale_pos_weight=float(spw),random_state=42,n_jobs=1,verbosity=0)
 }
for featlabel,features in [('raw',raw),('opportunity_augmented',opp)]:
 X=D[features].astype(float).values
 for regime,splits in [('random',rs),('course_disjoint',gs)]:
  for fold,(tr,te) in enumerate(splits):
   for j,dim in enumerate(names):
    ytr,yte=Y[tr,j],Y[te,j]
    for mn,est in models(ytr).items():
     est.fit(X[tr],ytr); yh=est.predict(X[te]); pr=est.predict_proba(X[te])[:,1]
     rowsout.append(dict(feature_set=featlabel,regime=regime,fold=fold,dim=dim,model=mn,BA=balanced_accuracy_score(yte,yh),MacroF1=f1_score(yte,yh,average='macro'),MCC=matthews_corrcoef(yte,yh),AUROC=roc_auc_score(yte,pr),ACC=accuracy_score(yte,yh)))
     for ii,yt,yp,pp in zip(te,yte,yh,pr): predout.append(dict(feature_set=featlabel,regime=regime,fold=fold,dim=dim,model=mn,row_index=int(ii),userid=int(D.loc[ii,'userid']),courseid=int(D.loc[ii,'courseid']),y_true=int(yt),y_pred=int(yp),prob=float(pp)))
pd.DataFrame(rowsout).to_csv(OUTPUT_DIR / 'baseline_fold_metrics.csv',index=False); pd.DataFrame(predout).to_csv(OUTPUT_DIR / 'baseline_predictions.csv',index=False)
# audits
pd.DataFrame({'feature':raw}).to_csv(OUTPUT_DIR / 'feature_list_raw.csv',index=False); pd.DataFrame({'feature':opp}).to_csv(OUTPUT_DIR / 'feature_list_opportunity.csv',index=False)
folds=[]
for rg,splits in [('random',rs),('course_disjoint',gs)]:
 for f,(tr,te) in enumerate(splits):
  for i in te: folds.append(dict(regime=rg,fold=f,row_index=int(i),userid=int(D.loc[i,'userid']),courseid=int(D.loc[i,'courseid'])))
pd.DataFrame(folds).to_csv(OUTPUT_DIR / 'fold_assignments.csv',index=False)
print(pd.DataFrame(rowsout).groupby(['feature_set','regime','model'])[['BA','MacroF1','MCC','AUROC']].mean().round(4).to_string())
