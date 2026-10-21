import csv, warnings, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.autograd import Function
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
warnings.filterwarnings('ignore'); torch.set_num_threads(1)
SEED=42
# data
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'DATA'
OUTPUT_DIR = ROOT / 'OUTPUT'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
p = DATA_DIR / 'compiled_pendikom_moodle_logs.csv'; lines=[]
for line in open(p,encoding='utf-8-sig'):
 s=line.rstrip('\r\n'); s=s[1:-1] if len(s)>1 and s[0]=='"' and s[-1]=='"' else s; lines.append(s)
header=next(csv.reader([lines[0]])); rows=[next(csv.reader([s])) for s in lines[1:]]
df=pd.DataFrame(rows,columns=header)
for c in df.columns:
 cv=pd.to_numeric(df[c],errors='coerce')
 if cv.notna().sum()>=int(.9*len(df)): df[c]=cv
df['tie_on_total_activity']=df['tie_on_total_activity'].map({'True':1,'False':0}).fillna(0).astype(int)
D=df[df.has_ils==1].copy().reset_index(drop=True); D['activity_span_days']=(D.last_activity-D.first_activity)/86400
scores=['ils_active_reflective_score','ils_sensing_intuitive_score','ils_visual_verbal_score','ils_sequential_global_score']; Y=np.column_stack([(pd.to_numeric(D[c])>0).astype(int) for c in scores]).astype(np.float32)
beh=header[4:55]; oppc=['course_weeks','content_available','content_stay_reference_seconds','question_available','assign_available','exercise_stay_reference_seconds','overview_stay_reference_seconds','ques_detail_available','ques_overview_available','ques_facts_available','ques_concepts_available','ques_graphics_available','ques_text_available','ques_interpret_available','ques_develop_available']
raw=[c for c in beh if c not in set(['first_activity','last_activity','tie_on_total_activity']+oppc)]+['activity_span_days']
ratio={'content_visit_per_available':('content_visit_evidence_count','content_available'),'content_time_ratio':('content_stay_seconds','content_stay_reference_seconds'),'question_attempt_per_available':('question_attempt_count','question_available'),'assignment_submit_per_available':('submitted_assignment_count','assign_available'),'exercise_time_ratio':('exercise_stay_seconds','exercise_stay_reference_seconds'),'overview_time_ratio':('overview_stay_seconds','overview_stay_reference_seconds'),'detail_attempt_per_available':('ques_detail_attempted_count','ques_detail_available'),'overview_attempt_per_available':('ques_overview_attempted_count','ques_overview_available'),'facts_attempt_per_available':('ques_facts_attempted_count','ques_facts_available'),'concepts_attempt_per_available':('ques_concepts_attempted_count','ques_concepts_available'),'graphics_attempt_per_available':('ques_graphics_attempted_count','ques_graphics_available'),'text_attempt_per_available':('ques_text_attempted_count','ques_text_available'),'interpret_attempt_per_available':('ques_interpret_attempted_count','ques_interpret_available'),'develop_attempt_per_available':('ques_develop_attempted_count','ques_develop_available'),'quiz_result_avg_stay':('quiz_results_stay_seconds_sum','quiz_results_event_count')}
for k,(a,b) in ratio.items(): D[k]=D[a].astype(float)/(D[b].astype(float)+1e-6)
feat=raw+list(ratio); groups=D.courseid.values; splits=list(GroupKFold(5).split(np.zeros(len(D)),groups=groups))
met=pd.read_csv(OUTPUT_DIR / 'neural_dg_fold_metrics.csv'); epoch_lookup=met.groupby(['config','fold']).epochs.first().to_dict()
class GRLFn(Function):
 @staticmethod
 def forward(ctx,x,a): ctx.a=a; return x.view_as(x)
 @staticmethod
 def backward(ctx,g): return -ctx.a*g,None
class Net(nn.Module):
 def __init__(self,p,nd,domain=False):
  super().__init__(); self.encoder=nn.Sequential(nn.Linear(p,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,16),nn.ReLU()); self.task=nn.Linear(16,4); self.domain=nn.Sequential(nn.Linear(16,32),nn.ReLU(),nn.Linear(32,nd)) if domain else None
 def forward(self,x,reverse=False):
  h=self.encoder(x); dl=None
  if self.domain is not None:
   z=GRLFn.apply(h,1.0) if reverse else h; dl=self.domain(z)
  return h,self.task(h),dl

def train(X,tr,epochs,gamma,lam,domain,seed):
 torch.manual_seed(seed); np.random.seed(seed)
 cours=np.unique(groups[tr]); mp={c:i for i,c in enumerate(cours)}; dom=np.array([mp[c] for c in groups[tr]],int)
 model=Net(X.shape[1],len(cours),domain); opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
 yt=Y[tr]; pos=yt.sum(0); neg=len(yt)-pos; bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/np.maximum(pos,1),dtype=torch.float32),reduction='none'); ce=nn.CrossEntropyLoss()
 xt=torch.tensor(X[tr],dtype=torch.float32); yy=torch.tensor(yt,dtype=torch.float32); dd=torch.tensor(dom); carr=groups[tr]
 for _ in range(int(epochs)):
  model.train(); opt.zero_grad(); h,l,dl=model(xt,reverse=domain); e=bce(l,yy); loss=e.mean()
  if gamma>0:
   cr=[]
   for c in cours: cr.append(e[torch.tensor(carr==c)].mean())
   cr=torch.stack(cr); loss=loss+gamma*((cr-cr.mean())**2).mean()
  if domain: loss=loss+lam*ce(dl,dd)
  loss.backward(); opt.step()
 return model
rows=[]
for fold,(tr,te) in enumerate(splits):
 X0=np.log1p(D[feat].astype(float).values); sc=StandardScaler().fit(X0[tr]); X=sc.transform(X0)
 reps={'Original':X[tr]}
 for cfg,gam,lam,dom in [('ERM_opportunity',0,0,False),('Full_DG',.2,.15,True)]:
  ep=epoch_lookup[(cfg,fold)]; model=train(X,tr,ep,gam,lam,dom,SEED+100+fold); model.eval()
  with torch.no_grad(): reps[cfg]=model(torch.tensor(X[tr],dtype=torch.float32))[0].numpy()
 # restrict diagnostic to source courses >=5 source samples
 vc=pd.Series(groups[tr]).value_counts(); keep=vc[vc>=5].index; mask=np.isin(groups[tr],keep); y=groups[tr][mask].astype(str)
 # min class >=5 => 5fold possible but use 3
 cv=StratifiedKFold(3,shuffle=True,random_state=100+fold)
 for rname,Z in reps.items():
  clf=RandomForestClassifier(n_estimators=200,random_state=fold,n_jobs=1,class_weight='balanced_subsample')
  pred=cross_val_predict(clf,Z[mask],y,cv=cv,n_jobs=1)
  rows.append({'fold':fold,'representation':rname,'n_courses':len(keep),'n_samples':mask.sum(),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'chance':1/len(keep)})
pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'domain_invariance_diagnostic.csv',index=False)
print(pd.DataFrame(rows).groupby('representation')[['accuracy','balanced_accuracy','macro_f1']].agg(['mean','std']).round(4).to_string())
