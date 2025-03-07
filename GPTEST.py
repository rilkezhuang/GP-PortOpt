import framework.Universe as uv
from framework.Alpha import AlphaBase
from framework.Niodata import *
from framework.Checkpoint import Checkpoint
import pandas as pd
#import Oputil
import numpy as np

'''Gplearning Function def' starts here'''

import random # for the generation of random seed
import multiprocessing
import operators as op
#from joblib import Parallel,delayed
from joblib import Parallel, delayed
import numba as nb
POP_SIZE=600
POP_SIZE_Origin=600
Generation=60
CX_PROB=0.7
MUT_PROB=0.5
METHODS=['full','grow']
MAX_DEPTH=5
MIN_DEPTH=2
TOURNAMENT_SIZE=60
SAMPLE_SIZE=6656
SAMPLE_NUM=253
FEATURE_SIZE=SAMPLE_SIZE
FEATRUE_NUM=SAMPLE_NUM


TS_FUNCTIONS=op.TS_FUNCTIONS
FUNCTIONS=op.FUNCTIONS
TERMINALS=['x1',-2,-1,0,1,2,'x2']
CONST=[1,2,3,4,5]


# define tree strsuctrue
class Node:
    def __init__(self,val=None,left=None,right=None):
        self.val=val # node and leaf
        self.left=left
        self.right=right
        self.bestrun=[]

    def node_label(self): # string label
        if (self.val[0] in FUNCTIONS):
            return self.val[0].__name__
        else: 
            return str(self.val[0])
    
    def print_tree(node, prefix="", is_left=True):
        if node:
            if node.val[0] in TERMINALS:
                print(prefix + ("├── " if is_left else "└── ") + str(node.val[0]))
            elif node.val[0] in FUNCTIONS:
                print(prefix + ("├──  " if is_left else "└── ") + str(node.val[0].__name__))
            if node.left:
                node.left.print_tree(prefix + ("│   " if is_left else "    "), True)
            if node.right :
                node.right.print_tree( prefix + ("│   " if is_left else "    "), False)
    def evaluate(self,x1,x2):
        
        if (self.val[0] in FUNCTIONS):
            return self.val[0](self.left.evaluate(x1,x2),self.right.evaluate(x1,x2))
        
        elif self.val[0]=='x1': return x1
        elif self.val[0]=='x2': return x2
        else: return self.val
    def grow_tree(self,method='grow',max_depth=MAX_DEPTH,depth=0):
        if depth<MIN_DEPTH or (depth<max_depth and method !='grow'):
                self.val=[]
                func=random.choice([c for c in FUNCTIONS if c not in TS_FUNCTIONS])
                for i in range(SAMPLE_SIZE):
                    self.val.append(func)
        elif depth>=max_depth:
            ter=random.choice(TERMINALS)
            self.val=np.full(SAMPLE_SIZE,ter)
            self.left=None
            self.right=None
        else: 
            if random.random()> 0.5:
                ter=random.choice(TERMINALS)
                self.val=np.full(SAMPLE_SIZE,ter)
            else: 
                self.val=[]
                func=random.choice(FUNCTIONS)
                for i in range(SAMPLE_SIZE):
                    self.val.append(func)
        if self.val[0] in FUNCTIONS:
            if self.val[0]in TS_FUNCTIONS:
                self.left=Node()
                self.left.grow_tree(method,depth=depth+1)
                self.right=Node()
                self.right.grow_tree(method,depth=MAX_DEPTH)
            else:
                self.left=Node()
                self.left.grow_tree(method,depth=depth+1)
                self.right=Node()
                self.right.grow_tree(method,depth=depth+1)
    
    def size(self):
        #print(self.catch)
        if self.val[0] in TERMINALS: return 1
        l=0
        r=0
        if self.left:
            l=self.left.size()
        if self.right:
            r=self.right.size() 
        return 1+max(l,r)
    def get_random_node(self,step,size):
        self.catch="random node"
        step-=1
        if step<=1:
            return self
        
        else:
            len_left=0
            len_right=0
            if self.left:
                len_left=self.left.size()
            if self.right:
                len_right=self.right.size()

            if len_left>=step and len_right<step:
                #print("left")
                return self.left.get_random_node(step,size)
            elif len_left<step and len_right>=step:
                 #print("right")
                 return self.right.get_random_node(step,size)
            elif len_left>=step and len_right>=step:
                #print("random")
                option=random.choice([self.left,self.right])
                return option.get_random_node(step,size)
    def replace(self,new_subtree):
        self.val=new_subtree.val
        self.left=new_subtree.left
        self.right=new_subtree.right
    def copy(self):
        new_node = Node(self.val)
        if self.left:
            new_node.left = self.left.copy()
        if self.right:
            new_node.right = self.right.copy()
        return new_node
    def evolve(self,other):
        #self.all_nodes=self.get_all_nodes()
        fsize=self.size()
        msize=other.size()   
        step1=random.randint(1,fsize)
        step2=random.randint(1,msize)
        point1=self.get_random_node(step1,fsize)
        #print("evolve father",self.print_tree(),point1.print_tree())
        point2=other.get_random_node(step2,msize)
        #print("evolve mother",other.print_tree(),point2.print_tree())
        option=random.random()
        if option<0.5:
            if fsize-point1.size()+point2.size()<=MAX_DEPTH:
                new_subtree=point2.copy()
                point1.replace(new_subtree)
        elif option>=0.5 and option<0.7:
            point1.point_mutation()
        else:
            point1.mutation()
    def point_mutation(self):
        if self.val[0] in FUNCTIONS:
            self.val=[]
            func=random.choice(FUNCTIONS)
            for i in range(SAMPLE_SIZE):
                self.val.append(func)
        else:
            con=random.choice(CONST)
            self.val=np.full(SAMPLE_SIZE,con)
    def mutation(self):
            self.grow_tree(max_depth=2)
   

        
def init_population(popsize=POP_SIZE):
    pop =[]
    for md in range(3,MAX_DEPTH+1):
        for i in range(int(popsize/(MAX_DEPTH-2)/2)):
            t=Node()
            t.grow_tree(max_depth=md)
            pop.append(t)
            j=Node()
            j.grow_tree(max_depth=md,method='full')
            pop.append(j)
            
    return pop
def evaluate_individual(individual_data):
    indiv, x1,x2, y = individual_data
    #print(np.shape(x))
    invalid = 0
    result=0
    i=0
    if indiv.size()>MAX_DEPTH:
        return np.nan
    pred=[]
    for i in  range(0,SAMPLE_NUM,1):
        y_pred = indiv.evaluate(x1[i],x2[i])  # 個體評估
        pred.append(list(y_pred))
    result=np.ma.corrcoef(pred, y)[0, 1]
    #print("--------------pred",np.shape(pred))
    #print("=------------------------y",np.shape(y))
    #print(result)
    return result

def fitness(population,x1,x2,y):
 
    indiv_data= [(indiv,x1,x2,y) for indiv in population]
    
    #print(indiv_data[0])
    with multiprocessing.Pool() as pool:
        fitnesses = pool.map(evaluate_individual, indiv_data)
   
    return list(fitnesses)

def selection(poplation,fitnesses):
    
    tournament=[random.randint(0,len(poplation)-1) for i in range(TOURNAMENT_SIZE)]
    tournament_fitness=[fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    winner_index=tournament[np.argmax(tournament_fitness)]
    return poplation[winner_index]
    




'''gplearning function def' ends'''

class AlphaMLSample(AlphaBase):
  def __init__(self, cfg):
    '''
      initialize functions: load data, parse config, init_variables.
    '''
    AlphaBase.__init__(self, cfg)
    self.liveMode = cfg.getAttributeDefault('liveMode', False)
    self.trainDelay = cfg.getAttributeDefault('trainDelay', 1) ## set trainDelay for both dat and label used in model. e.g., if we use returnsN as label, then we could use returnsN to di-trainDelay, and use data to di-N-trainDelay. 
    self.backdays = cfg.getAttributeDefault('backdays', 60) # backdays of data used in each train process
    ## below frequency and freqoffset are set for control train frequency
    self.frequency = int(cfg.getAttributeDefault('frequency', 1))
    self.freqoffset = int(cfg.getAttributeDefault('freqoffset', 0))
    if (self.frequency < 1 or self.freqoffset < 0):
      print ("Error: self.frequency < 1 or self.freqoffset < 0")
      sys.exit(0)
    self.saveModel = cfg.getAttributeDefault('saveModel', False) # save model to disk or not.
    self.modelPath = cfg.getAttributeDefault('modelPath', './model') # set model path if saveModel is true 
    self.retrain = cfg.getAttributeDefault('retrain', False) # if set to true, then the model will retrain regardless if already trained and saved to disk.
    self.model = None ## represent the ML model

    self.close = self.dr.GetData('close')
    self.tic = self.dr.GetData('ticker')
    self.ticIx = self.dr.GetData('tickerIx')
    self.cps = self.dr.GetData('adj_close')
    self.ret = self.dr.GetData('adj_returns')
    self.ret5 = self.dr.GetData('returns5')
    self.cap=self.dr.GetData('cap')
    self.vol=self.dr.GetData('volume')
    self.top2000=self.dr.GetData('TOP2000')
    self.count=0
    self.tvr=np.array(self.vol[:][0:uv.instsz])/np.array(self.cap[:][0:uv.instsz])
    self.func=['add', 'sub', 'mul', 'div', 'sin', 'cos']
    self.ret20=self.dr.GetData('returns20')




  
  '''
    model train logic goes here
  '''
  def train(self,x1,x2, label):
      #print(np.shape(x1),np.shape(x2),np.shape(label))
      random.seed()
      global POP_SIZE_Origin
      global POP_SIZE
      y=label
      population=init_population()
      BESTRUN=[]
      best_of_run=None
      best_f=0
      best_gen=0
      fitnesses=fitness(population,x1,x2,y)
      stuck_check=0
      exploration_status=0
      for gen in range(Generation):
          next_pop=[]
          for i in range(POP_SIZE):
              p1=selection(population,fitnesses)
              p2=selection(population,fitnesses)
              p1.evolve(p2)
              next_pop.append(p1)
          population=next_pop
          fitnesses=fitness(population,x1,x2,y) 
          if np.nanmax(fitnesses)==1 or np.nanmax(fitnesses)==np.nan:
              exploration_status=4
          if np.nanmax(fitnesses)>best_f and np.nanmax(fitnesses)<1:
              best_f=np.nanmax(fitnesses)
              best_gen=gen
              best_of_run=population[fitnesses.index(np.nanmax(fitnesses))]
              print("------------------------------------")
              print("gen:",gen,"fitness:",best_f,"best run:",best_of_run.print_tree())
              BESTRUN.append(best_of_run)
              stuck_check=0
              exploration_status=0
              POP_SIZE=POP_SIZE_Origin
          else:
              stuck_check+=1
          if exploration_status>=4:
              POP_SIZE+=300
          if stuck_check>=4:
              population=init_population(POP_SIZE)
              population[0:len(BESTRUN)]=BESTRUN
              fitnesses=fitness(population,x1,x2,y) 
              stuck_check=0
              exploration_status+=1

          print("--------------------END  OF RUN------------------")
          print(f"current process: {gen+1} / {Generation}")
          print("best run obtain at gen",best_gen,"with fitness:",best_f)
          if best_of_run:
              best_of_run.print_tree()




  '''
    model predict logic goes here
  '''
  def predict(self, di, ti, data):
     self.model.predict(data)


  '''
    run when liveMode="true", only predict, no model train
  '''
  def GenLive(self, di, ti = None):
    if (self.LoadModel(uv.Dates[di]) == False): return ## try to load model first
    if self.model is None: return ## double check if the model is None

    ## predict and assign alpha value
    predict_data = None
    # prepare predict data and then predict
    # predict_data = ...
    self.predict(di, ti, predict_data)

    # assign alpha value from predict results
    # self.alpha = ...

  '''
    run when liveMode="false", normal train process goes here
  '''
  def GenHist(self, di, ti = None):
    '''
      for each new day, first update data/label queue for training
    '''

    ## below codes is to check if the model exists or need train
    runTrain = True
    '''if (self.retrain == False and di % self.frequency == self.freqoffset and self.LoadModel(uv.Dates[di]) == True): runTrain = False
'''
    ## run model train when needed 
    #if (di % self.frequency == self.freqoffset and runTrain == True):
    if (runTrain == True):
        ret5=self.ret5[di-249:di+4][0:uv.instsz]
        ret=self.ret5[di-254:di-1][0:uv.instsz]
        tvr=self.tvr[di-254:di-1][0:uv.instsz]
        
       

        X1=tvr
        X2=ret
    
        Y_train=ret5
      

        #print(dr)
        label=ret
        #prepare train data and train label and then train
        #data = ...
        #label = ...
        self.train(X1,X2, Y_train)
        ## save model
        self.SaveModel(uv.Dates[di])

    ## predict and assign alpha value
   

  def GenAlpha(self, di, ti = None):
    if self.liveMode is False:
      self.GenHist(di, ti)
    else:
      self.GenLive(di, ti)


  '''
    save the model of specific date to modelPath
    return True if successful and False otherwise
  '''
  def SaveModel(self, dt):
    if self.model is not None:
      modelpath = '%s/model.%s' % (self.modelPath, dt)
      if self.model is not None:
        # save model here
        pass

  '''
    load the model of specific date from modelPath
    return True if successful and False otherwise
  '''
  def LoadModel(self, dt):
    return True
    modelpath = '%s/model.%s' % (self.modelPath, dt)
    if not os.path.exists(modelpath):
      return False
    try:
      if os.path.exists(modelpath):
        #load model here, self.model = load_model ...
        pass
    except:
      return False
    return True
 

  '''
    save local variables
  '''
  def SaveVar(self, checkpoint):
    pass

  '''
    load local variables
  '''
  def LoadVar(self, checkpoint):
    pass

# create an instance
def create(cfg):
  return AlphaMLSample(cfg)
