import pickle
import time
import math

class NB:

    def __init__(self):
        pass


    def fit(self,x,y):
        # x and y are text dataset
        self.x = x  # [ 'hello','hi','abc' ]
        self.y = y  # [ news, news, sport ]
        
        start_time = time.time()
        print("Calculating counts...")
        tmp = list(set(y))
        self.no_class = len(tmp)
        
        self.index2class = { i : tmp[i] for i in range(self.no_class) } # mapping: 1 -> news and such
        self.class2index = { tmp[i] : i for i in range(self.no_class) } # news -> 1

        self.matrix = {}

        # find each word count 

        self.word_count_class = [0] * self.no_class 

        self.word_count = {}

        uniqWord = set()

        for i in range(len(x)):
            line = x[i]
            words = line.split()
            res = y[i]
            self.word_count_class[ self.class2index[res] ] += len(words)
            for word in words:
                uniqWord.add(word)
                if word not in self.matrix:
                    tmp = [0] * self.no_class
                    tmp[ self.class2index[res]  ] += 1
                    self.matrix[word] = tmp
                else:
                    tmp = self.matrix[word]
                    tmp[ self.class2index[res]  ] += 1
                    self.matrix[word] = tmp
                    
                if word not in self.word_count:
                    self.word_count[word] = 1
                else:
                    self.word_count[word] = self.word_count[word] + 1
                
        self.total_word_count = sum(self.word_count_class)
        self.count_uniq_words  = len(uniqWord)
        print("Calculated weights in time:",time.time()-start_time)
    
    def save(self,path='model.pkl'):
        file = open(path,'wb')
        pickle.dump(self,file,pickle.HIGHEST_PROTOCOL)
        file.close()
        print("Model Saved.")
    
    def load(self,path):
        try:
            file = open(path,'rb')
            self = pickle.load(file)
            print("Model Loaded.")
        except:
            print("Error loading model.")

    def predict(self, x, use_laplace_smoothing = True):
        '''
        x needs to be an array of preprocessesd strings
        '''
        newy = []
        print("Predicting...")
        start_time = time.time()
        for i in range(len(x)):
            input = x[i]    
            words = input.split()
            
            alpha = 0
            d = self.count_uniq_words

            if use_laplace_smoothing == True:
                alpha = 1
            
            liklihood = []
            priors = [1]*self.no_class
            

            # find likliehood
            for word in words:
                if word in self.matrix:
                    res = [0]*(self.no_class)
                    tmp = self.matrix[word]
                    for j in range(self.no_class):
                        cnt = tmp[j]
                        tots = self.word_count_class[j] 
                        prob = math.log(cnt + alpha) - math.log(tots + alpha*d)
                        res[j] = prob
                    liklihood.append(res)
                
            norms = -1
            num = -1
            den = -1
            for word in words:
                if word in self.word_count:
                    if norms == -1:
                        #set priot
                        norms = 1
                        num = math.log(self.word_count[word])
                        den = math.log(self.total_word_count)
                    else:
                        num += math.log(self.word_count[word])
                        den += math.log(self.total_word_count)
            
            
            
            # find liklihood
            act_liklihood = []
            for j in range(self.no_class):
                prob = -1
                for k in range(len(liklihood)):
                    if prob == -1:
                        prob = liklihood[k][j]
                    else:
                        prob += liklihood[k][j]
                act_liklihood.append(prob)


            # find prior factors
            for j in range(self.no_class):
                prob = math.log(self.word_count_class[j] ) - math.log( self.total_word_count)
                priors[j] = prob
            

            # estimate predictions
            predicts = []

            for j in range(self.no_class):
                # prior * likilihood /norm
                val = (priors[j] + act_liklihood[j] + den) - (num)
                predicts.append(val)
            
            newy.append(predicts)
        return newy

    def get_pred_classes(self,input):
        newy = self.predict(input)
        args = argmax(newy)
        classes = []
        for pred in args:
            classes.append(self.index2class[pred])
        return classes


def argmax(lists):
    classes = []
    for list in lists:
        mx = max(list)
        index = list.index(mx)
        classes.append(index)
    return classes


            














                




