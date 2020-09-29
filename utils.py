import re
class tokenizer:
    def __init__(self, corpus):
        cc = []
        for x in corpus:
            ans = ' '.join(re.sub('([^A-Za-z ])|(@[A-Za-z0-9]+)|(\w+:\/\/\S+)','',x).split())
            cc.append(ans.lower())
        self.corp = cc
        cc = None
        corpus = None
        self.v = {}
        self.seqv = {}

    def vocab(self, ret=False):
        for x in self.corp:
            words = x.split()
            for i in words:
                if i not in self.v:
                    self.v[i] = 1
                else:
                    self.v[i] += 1            
        self.corp = None
        if ret:
            return self.v
    
    def remf(self, minf, maxf, ret=False):
        truncv = {}
        for x in self.v:
            ax = self.v.get(x)
            if minf < ax and ax < maxf:
                truncv[x] = ax
        self.v = truncv
        if ret:
            return self.v
    
    def sen2seq(self, sen):
        if not self.seqv:
           self.seqv = dict(zip(list(self.v.keys()), list(range(len(self.v)))))
        seqsens = []
        for x in sen:
            ssen = x.split()
            seqsen = []
            for i in ssen:
                if i in self.seqv:
                    seqsen.append(self.seqv.get(i))
                else:
                    seqsen.append(len(self.seqv)+1)
            seqsens.append(seqsen)
        return seqsens
    
    def seq2sen(self, seq):
        fseqv = dict(zip(list(self.seqv.values()), list(self.seqv.keys())))
        senseqs = []
        for x in seq:
            senseq = []
            for i in x:
                if i in fseqv:
                    senseq.append(fseqv.get(i))
                else:
                    senseq.append('<UNK>')

            senseqs.append(senseq)
        fseqv = None
        return senseqs

    def pad(self, seqs, maxlen):
        padseqs = []
        for x in seqs:
            padseqs.append(x + [0] * (maxlen - len(x)))
        return padseqs
    
    def sanatize(self, sen):
        sansen = []
        for x in sen:
            ans = ' '.join(re.sub('([^A-Za-z ])|(@[A-Za-z0-9]+)|(\w+:\/\/\S+)','',x).split())
            sansen.append(ans.lower())
        return sansen