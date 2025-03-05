import numpy as np
import pandas as pd
import random # for the generation of random seed
import multiprocessing
POP_SIZE=6000
Generation=20
CX_PROB=0.7
MUT_PROB=0.5

MAX_DEPTH=4
MIN_DEPTH=2
TOURNAMENT_SIZE=50
SAMPLE_SIZE=10
SAMPLE_NUM=10
FEATURE_SIZE=SAMPLE_SIZE
FEATRUE_NUM=SAMPLE_NUM
X_samples=np.zeros((10,10))
for i in range(np.shape(X_samples)[0]):
    for j in range(np.shape(X_samples)[1]):
        X_samples[i][j]=i*10+j




def target_func(x):
    return x**2+2*x+1

#define Function sets and ternminals 四则算符区
def add(a,b): 
    #print('add') 
    result=np.array(a)+np.array(b)
    #print(np.shape(a),np.shape(b),np.shape(result))
    return  result
def sub(a,b): 
    #print('sub')
    result=np.array(a)-np.array(b)
    #print(np.shape(a),np.shape(b),np.shape(result))
    return result
def mul(a,b): 
    #print('mul')
    result=np.multiply(np.array(a),np.array(b))
    #print(np.shape(a),np.shape(b),np.shape(result))
    return result
def div(a,b): 
    #print("div")
    #print(np.shape(a),np.shape(b))
    if b.all()!=0:
        return np.divide(a,b)
    else:
        return np.full(10,100000000)
        
    


#Time Series Operation 时序算符def区
def ma(x,lookbackwindow=5):
    #print(lookbackwindow)
    if lookbackwindow[0]!=lookbackwindow[1] or lookbackwindow[0]<=0:
        return np.full(SAMPLE_SIZE,1000000)
    else:
        #print(lookbackwindow[0])
        result=pd.Series(x).rolling(int(lookbackwindow[0])).mean()
        return np.array(result)
def ema(x,span_window=5):
    #print(span_window)
    if span_window[0]!=span_window[1] or span_window[0]<=1:
        return np.full(SAMPLE_SIZE,10000000)

    else:
        result=pd.Series(x).ewm(span=int(span_window[0])).mean()
        return np.array(result)




TS_FUNCTIONS=[ma,ema]
FUNCTIONS=[add,sub,mul,div,ma,ema]
const=np.linspace
TERMINALS=['x',-2,-1,0,1,2]
CONST=[1,2,3,4,5]


# define tree strsuctrue
class Node:
    def __init__(self,val=None,left=None,right=None):
        self.val=val # node and leaf
        self.left=left
        self.right=right
    def node_label(self): # string label
        if (self.val[0] in FUNCTIONS):
            return self.val[0].__name__
        else: 
            return str(self.val[0])
    
    def print_tree(node, prefix="", is_left=True):
        if node:
            if node.val[0] in TERMINALS:
                print(prefix + ("├──" if is_left else "└── ") + str(node.val[0]))
            elif node.val[0] in FUNCTIONS:
                print(prefix + ("├── " if is_left else "└── ") + str(node.val[0].__name__))
            if node.left:
                node.left.print_tree(prefix + ("│   " if is_left else "    "), True)
            if node.right :
                node.right.print_tree( prefix + ("│   " if is_left else "    "), False)
    def evaluate(self,x):
        
        if (self.val[0] in FUNCTIONS):
            return self.val[0](self.left.evaluate(x),self.right.evaluate(x))
        
        elif self.val[0]=='x': return x
        else: return self.val
    def grow_tree(self,method='grow',max_depth=MAX_DEPTH,depth=0):
        if depth<MIN_DEPTH or (depth<max_depth and method!='grow'):
                self.val=[]
                func=random.choice([c for c in FUNCTIONS if c not in TS_FUNCTIONS])
                for i in range(10):
                    self.val.append(func)
        elif depth>=max_depth:
            ter=random.choice(TERMINALS)
            self.val=np.full(10,ter)
        else: 
            if random.random()> 0.5:
                ter=random.choice(TERMINALS)
                self.val=np.full(10,ter)
            else: 
                self.val=[]
                func=random.choice(FUNCTIONS)
                for i in range(10):
                    self.val.append(func)
        if self.val[0] in FUNCTIONS:
            if self.val[0]in TS_FUNCTIONS:
                self.left=Node()
                self.left.grow_tree(depth=depth+1)
                self.right=Node()
                self.right.grow_tree(depth=MAX_DEPTH)
            else:
                self.left=Node()
                self.left.grow_tree(depth=depth+1)
                self.right=Node()
                self.right.grow_tree(depth=depth+1)
    def mutation(self):
        if random.random()<MUT_PROB:
            
            self.grow_tree(max_depth=max(MAX_DEPTH-self.size(),1))
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation()
    def size(self):
        if self.val[0] in TERMINALS: return 1
        l=self.left.size() if self.left else 0
        r=self.right.size() if self.right else 0
        return 1+max(l,r)
    def build_subtree(self):
        t=Node()
        t.val=self.val
        if self.left: t.left=self.left.build_subtree()
        if self.right: t.right= self.right.build_subtree()
        return t
    def scan_tree(self,count,second):
        count[0]-=1
        if count[0]<=1:
            if not second:
                return self.build_subtree()
            else:
                self.val=second.val
                self.left=second.left
                self.right=second.right
        else:
            ret =None
            if self.left and count[0]>1:ret=self.left.scan_tree(count,second)
            if self.right and count[0]>1: ret=self.right.scan_tree(count,second)
            return ret

    def crossover(self,other):
        if random.random()<CX_PROB:
            second=other.scan_tree([random.randint(1,other.size())],None)
            self.scan_tree([random.randint(1,self.size())],second)
        
def init_population():
    pop =[]
    for md in range(3,MAX_DEPTH+1):
        for i in range(int(POP_SIZE/(MAX_DEPTH-2))):
            t=Node()
            t.grow_tree(max_depth=md)
            pop.append(t)

    return pop
def evaluate_individual(individual_data):
    indiv, x, y = individual_data
    #print(np.shape(x))
    invalid = 0
    result=0
    i=0
    for i in  range(0,10,1):
        y_pred = indiv.evaluate(x[i])  # 個體評估
        result+=np.mean(abs(y_pred-y[i]))
    if invalid == 1:
        return 100000000
    return np.mean(result)
    

def fitness(population,x,y):
 
    indiv_data= [(indiv,x,y) for indiv in population]
    
    #print(indiv_data[0])
    with multiprocessing.Pool() as pool:
        fitnesses = pool.map(evaluate_individual, indiv_data)
    
    return list(fitnesses)#.reshape(np.shape(fitnesses)[1]))

def selection(poplation,fitness):
    
    tournament=[random.randint(0,len(poplation)-1) for i in range(TOURNAMENT_SIZE)]
    tournament_fitness=[fitness[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    winner_index=tournament[np.argmax(tournament_fitness)]
    return poplation[winner_index]
    
def main():
    random.seed()
    dataset=X_samples
    y=target_func(dataset)
    population=init_population()
 
    best_of_run=None
    best_f=10000000000
    best_gen=0
    fitnesses=fitness(population,dataset,y)

    for gen in range(Generation):
        next_pop=[]
        for i in range(POP_SIZE):
            p1=selection(population,fitnesses)
            p2=selection(population,fitnesses)
            p1.crossover(p2)
            p1.mutation()
            next_pop.append(p1)
        population=next_pop
        fitnesses=fitness(population,dataset,y) 
        
        if min(fitnesses)<best_f:
            best_f=min(fitnesses)
            best_gen=gen
            best_of_run=population[fitnesses.index(min(fitnesses))]
            print("------------------------------------")
            print("gen:",gen,"fitness:",best_f,"best run:",best_of_run.print_tree())
        print("--------------------END  OF RUN------------------")
        print("best run obtain at gen",best_gen,"with fitness:",best_f)
        best_of_run.print_tree()

if __name__=="__main__":
    main()



    

            








