import csv, warnings, json, random, math, os
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,f1_score,matthews_corrcoef,roc_auc_score,accuracy_score
warnings.filterwarnings('ignore')
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(1)
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
scores=['ils_active_reflective_score','ils_sensing_intuitive_score','ils_visual_verbal_score','ils_sequential_global_score']; dims=['AR','SI','VV','SG']
Y=np.column_stack([(pd.to_numeric(D[c])>0).astype(int) for c in scores]).astype(np.float32)
beh=header[4:55]; oppc=['course_weeks','content_available','content_stay_reference_seconds','question_available','assign_available','exercise_stay_reference_seconds','overview_stay_reference_seconds','ques_detail_available','ques_overview_available','ques_facts_available','ques_concepts_available','ques_graphics_available','ques_text_available','ques_interpret_available','ques_develop_available']
raw=[c for c in beh if c not in set(['first_activity','last_activity','tie_on_total_activity']+oppc)]+['activity_span_days']
ratio={'content_visit_per_available':('content_visit_evidence_count','content_available'),'content_time_ratio':('content_stay_seconds','content_stay_reference_seconds'),'question_attempt_per_available':('question_attempt_count','question_available'),'assignment_submit_per_available':('submitted_assignment_count','assign_available'),'exercise_time_ratio':('exercise_stay_seconds','exercise_stay_reference_seconds'),'overview_time_ratio':('overview_stay_seconds','overview_stay_reference_seconds'),'detail_attempt_per_available':('ques_detail_attempted_count','ques_detail_available'),'overview_attempt_per_available':('ques_overview_attempted_count','ques_overview_available'),'facts_attempt_per_available':('ques_facts_attempted_count','ques_facts_available'),'concepts_attempt_per_available':('ques_concepts_attempted_count','ques_concepts_available'),'graphics_attempt_per_available':('ques_graphics_attempted_count','ques_graphics_available'),'text_attempt_per_available':('ques_text_attempted_count','ques_text_available'),'interpret_attempt_per_available':('ques_interpret_attempted_count','ques_interpret_available'),'develop_attempt_per_available':('ques_develop_attempted_count','ques_develop_available'),'quiz_result_avg_stay':('quiz_results_stay_seconds_sum','quiz_results_event_count')}
for k,(a,b) in ratio.items(): D[k]=D[a].astype(float)/(D[b].astype(float)+1e-6)
opp=raw+list(ratio)
groups=D.courseid.values
gsplits=list(GroupKFold(5).split(np.zeros(len(D)),groups=groups))

class GRLFn(Function):
 @staticmethod
 def forward(ctx,x,alpha): ctx.alpha=alpha; return x.view_as(x)
 @staticmethod
 def backward(ctx,grad): return -ctx.alpha*grad, None
class GRL(nn.Module):
 def __init__(self,alpha=1.0): super().__init__(); self.alpha=alpha
 def forward(self,x): return GRLFn.apply(x,self.alpha)
class Net(nn.Module):
 def __init__(self,p,n_domains,use_domain=False):
  super().__init__(); self.encoder=nn.Sequential(nn.Linear(p,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,16),nn.ReLU())
  self.task=nn.Linear(16,4); self.use_domain=use_domain
  self.domain=nn.Sequential(GRL(1.0),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,n_domains)) if use_domain else None
 def forward(self,x):
  h=self.encoder(x); return h,self.task(h),(self.domain(h) if self.domain is not None else None)

def metrics(y,prob):
 pred=(prob>=.5).astype(int); out=[]
 for j,d in enumerate(dims):
  yt=y[:,j].astype(int); yp=pred[:,j]; pp=prob[:,j]
  out.append(dict(dim=d,BA=balanced_accuracy_score(yt,yp),MacroF1=f1_score(yt,yp,average='macro'),MCC=matthews_corrcoef(yt,yp),AUROC=roc_auc_score(yt,pp),ACC=accuracy_score(yt,yp)))
 return out

def train_once(X,Yv,g,idx_train,epochs,gamma=0,lam=0,use_domain=False,seed=42):
 torch.manual_seed(seed); np.random.seed(seed)
 # map domain labels in idx_train
 cours=np.unique(g[idx_train]); mp={c:i for i,c in enumerate(cours)}; dom=np.array([mp[c] for c in g[idx_train]],dtype=np.int64)
 model=Net(X.shape[1],len(cours),use_domain=use_domain)
 opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=1e-4)
 yt=Yv[idx_train]
 pos=yt.sum(axis=0); neg=len(yt)-pos; posw=torch.tensor(neg/np.maximum(pos,1),dtype=torch.float32)
 bce=nn.BCEWithLogitsLoss(pos_weight=posw,reduction='none'); ce=nn.CrossEntropyLoss()
 xt=torch.tensor(X[idx_train],dtype=torch.float32); yy=torch.tensor(yt,dtype=torch.float32); dd=torch.tensor(dom,dtype=torch.long)
 course_arr=g[idx_train]
 for ep in range(epochs):
  model.train(); opt.zero_grad(); h,logits,dlog=model(xt); elem=bce(logits,yy); task=elem.mean()
  risk=torch.tensor(0.,dtype=torch.float32)
  if gamma>0:
   cr=[]
   for c in cours:
    mask=torch.tensor(course_arr==c,dtype=torch.bool)
    cr.append(elem[mask].mean())
   cr=torch.stack(cr); risk=((cr-cr.mean())**2).mean()
  dloss=ce(dlog,dd) if use_domain else torch.tensor(0.,dtype=torch.float32)
  loss=task+gamma*risk+lam*dloss
  loss.backward(); opt.step()
 return model

def choose_epoch(X,Yv,g,outer_train,gamma,lam,use_domain,seed):
 # group-held-out source validation
 gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
 a,b=next(gss.split(np.zeros(len(outer_train)),groups=g[outer_train]))
 tr=outer_train[a]; va=outer_train[b]
 best=(-1,40); bad=0
 # train incrementally would be faster but implement checkpoint in one loop
 torch.manual_seed(seed); np.random.seed(seed)
 cours=np.unique(g[tr]); mp={c:i for i,c in enumerate(cours)}; dom=np.array([mp[c] for c in g[tr]],dtype=np.int64)
 model=Net(X.shape[1],len(cours),use_domain=use_domain); opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
 yt=Yv[tr]; pos=yt.sum(0); neg=len(yt)-pos; bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/np.maximum(pos,1),dtype=torch.float32),reduction='none'); ce=nn.CrossEntropyLoss()
 xt=torch.tensor(X[tr],dtype=torch.float32); yy=torch.tensor(yt,dtype=torch.float32); dd=torch.tensor(dom,dtype=torch.long); course_arr=g[tr]
 xv=torch.tensor(X[va],dtype=torch.float32)
 for ep in range(1,161):
  model.train(); opt.zero_grad(); h,l,dlog=model(xt); elem=bce(l,yy); task=elem.mean(); risk=torch.tensor(0.)
  if gamma>0:
   cr=[]
   for c in cours:
    mask=torch.tensor(course_arr==c,dtype=torch.bool); cr.append(elem[mask].mean())
   cr=torch.stack(cr); risk=((cr-cr.mean())**2).mean()
  dloss=ce(dlog,dd) if use_domain else torch.tensor(0.)
  (task+gamma*risk+lam*dloss).backward(); opt.step()
  if ep>=20 and ep%5==0:
   model.eval();
   with torch.no_grad(): prob=torch.sigmoid(model(xv)[1]).numpy()
   ba=np.mean([balanced_accuracy_score(Yv[va,j],prob[:,j]>=.5) for j in range(4)])
   if ba>best[0]+1e-4: best=(ba,ep); bad=0
   else: bad+=1
   if bad>=8: break
 return best[1],best[0]

configs=[
 ('ERM_raw',raw,0,0,False),
 ('ERM_opportunity',opp,0,0,False),
 ('Risk_only',opp,.2,0,False),
 ('Domain_only',opp,0,.15,True),
 ('Full_DG',opp,.2,.15,True)]
res=[]; preds=[]; latents={}
for fold,(otr,ote) in enumerate(gsplits):
 for cname,features,gamma,lam,ud in configs:
  X0=np.log1p(D[features].astype(float).values)
  sc=StandardScaler().fit(X0[otr]); X=sc.transform(X0)
  ep,valba=choose_epoch(X,Y,groups,otr,gamma,lam,ud,SEED+fold)
  model=train_once(X,Y,groups,otr,ep,gamma,lam,ud,SEED+100+fold)
  model.eval(); xt=torch.tensor(X[ote],dtype=torch.float32)
  with torch.no_grad(): h,logits,_=model(xt); prob=torch.sigmoid(logits).numpy(); hnp=h.numpy()
  for m in metrics(Y[ote],prob): res.append({'fold':fold,'config':cname,'epochs':ep,'source_val_BA':valba,**m})
  for rix,pp,hh in zip(ote,prob,hnp):
   for j,d in enumerate(dims): preds.append({'fold':fold,'config':cname,'row_index':int(rix),'userid':int(D.loc[rix,'userid']),'courseid':int(D.loc[rix,'courseid']),'dim':d,'y_true':int(Y[rix,j]),'prob':float(pp[j]),'y_pred':int(pp[j]>=.5)})
   if cname in ['ERM_opportunity','Full_DG']:
    latents[(cname,fold,int(rix))]=hh.tolist()
pd.DataFrame(res).to_csv(OUTPUT_DIR / 'neural_dg_fold_metrics.csv',index=False); pd.DataFrame(preds).to_csv(OUTPUT_DIR / 'neural_dg_predictions.csv',index=False)
# latent test rows save
lr=[]
for (c,f,i),h in latents.items(): lr.append({'config':c,'fold':f,'row_index':i,'userid':int(D.loc[i,'userid']),'courseid':int(D.loc[i,'courseid']),**{f'h{k}':v for k,v in enumerate(h)}})
pd.DataFrame(lr).to_csv(OUTPUT_DIR / 'neural_latent_test.csv',index=False)
print(pd.DataFrame(res).groupby('config')[['BA','MacroF1','MCC','AUROC']].mean().round(4).to_string())
print(pd.DataFrame(res).groupby(['config','dim'])[['BA','MacroF1','MCC','AUROC']].mean().round(4).to_string())
