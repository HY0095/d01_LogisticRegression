
from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
#from IPython.display import SVG, HTML
import copy as copy


_model_params_doc = """
    Parameters
    ----------
    y: array-like
        The dependent variable, dim = n*1
    x: array-like
        The independnet variable, dim = n*p. By default, an intercept is included.
    weight: array-like
        Each observation in the input data set is weighted by the value of the WEIGHT variable. By default, weight is np.ones(n)
    method: ['forward', 'backward', 'stepwise']
        The default selection method is 'stepwise'
    maxiter: int
        maxiter = 25 (default)
    mindiff: float
        mindiff = 1e-8 (default)
    """
_models_Result_docs = """
    params: array
        Parameters' Estimates
    AIC: float
        Akaike information criterion.  `-2*(llf - p)` where `p` is the number
        of regressors including the intercept.
    BIC: float
        Bayesian information criterion. `-2*llf + ln(nobs)*p` where `p` is the
        number of regressors including the intercept.
    SC: float
        Schwarz criterion. `-LogL + p*(log(nobs))`
    std_error: Array
        The standard errors of the coefficients.(bse)
    Chi_Square: float
        Wald Chi-square : (logit_res.params[0]/logit_res.bse[0])**2
    Chisqprob: float
        P-value from Chi_square test statistic 
    llf: float
        Value of the loglikelihood, as (LogL)
    """    
# Newton-Raphson Iteration
 
class ModelInfo(object):
    __doc__ = """
    The Logistic Regression Model.
    %(Params_doc)s
    %(Result_doc)s
    Notes
    ----
    """ % {'Params_doc' : _model_params_doc, 
           'Result_doc': _models_Result_docs}
    
    def __init__(self, data, role, **kwargs):
        self.data = sm.add_constant(data)
        self.role = role
        self.maxiter = 25   # default maxiter
        self.mindiff = 1e-8 # default mindiff
        self.method  = 'None'
        self.slentry = 0.05  # default
        self.slstay  = 0.05  # default
        self.cont    = True  # default self.cont = True, include intercept in model        
        if 'slentry' in kwargs.keys():
            self.slentry = float(kwargs['slentry'])
        if 'slstay' in kwargs.keys():
            self.slstay = float(kwargs['slstay'])
        if 'method' in kwargs.keys():
            self.method = kwargs['method']
        if 'maxiter' in kwargs.keys():
            self.maxiter = kwargs['maxiter']
        if 'mindiff' in kwargs.keys():
            self.mindiff = kwargs['mindiff'] 
    def xcol(self):
        #print(self.role)
        #print(self.data.columns)
        xcols = ['const']+[self.data.columns[i+1] for i, col in enumerate(self.role) if col == 1]
        return xcols
    def ycol(self):
        ycols = [self.data.columns[i+1] for i, col in enumerate(self.role) if col == 2]
        return ycols
    def weight(self):
        _weight_ = [self.data.columns[i+1] for i, col in enumerate(self.role) if col == 3]
        return _weight_
    def handledata(self, name):
        data = []
        for col in name:
            data.append(self.data[col])
        return np.array(data)
    def xdata(self):
        data = self.handledata(self.xcol())
        return data
    def ydata(self):
        data = self.handledata(self.ycol())
        return data
    def _weight(self):
        if 3 in self.role:
            data = self.data[self.weight()]
        else :
            data = np.ones(self.data.shape[0])
        return data

class LRStats(object):
    def __init__(self, step, n, p, res):
        self.aic  = res.aic
        self.bic  = res.bic
        self.logl = -2*res.llf
        self.sc   = 2*(-res.llf + p*(np.log(n)))
        self.params = res.params
        self.wald_chi = (res.params/res.bse)**2 
        self.std_error = res.bse
        self.pchi2 = 2*stats.norm.cdf(-np.abs((res.params/res.bse)))
    def resprint(self):
        print("                          Model Fit Statistics ")
        print("==============================================================================")
        print("AIC                   %s           BIC           %s    " % (self.aic, self.bic))
        print("-2Logl                %s           SC            %s    " % (self.logl, self.sc))
        print("==============================================================================")

        
        
class checkio(object):
    def __init__(self,xwait,score,pvalue):
        self.xwait = xwait
        self.score = score
        self.pvalue= pvalue
    def enter(self):
        print("              Analysis of Variables Eligible for Entry  ")
        print("==============================================================================")
        print("\t%5s\t \t%5s\t \t%5s\t" % ("variable", "Wald Chi-square", "Pr>ChiSq"))
        for i,v in enumerate(self.xwait):
            print("    \t%5s\t             \t%10s\t     \t%10s\t" % (v, self.score[i], self.pvalue[i]))
        print(" ") 
    def remove(self):
        print("              Analysis of Variables Eligible for Remove  ")
        print("==============================================================================")
        print("\t%5s\t \t%5s\t \t%5s\t" % ("variable", "Wald Chi-square", "Pr>ChiSq"))
        for i,v in enumerate(self.xwait):
            print("    \t%5s\t             \t%10s\t     \t%10s\t" % (v, self.score[i], self.pvalue[i]))
        print(" ")
        
class GlobalNullTest(object):
    def __init__(self,x,y,beta):
        self.x  = x
        self.p  = (x.shape[1] - 1)
        self.y  = y
        self.betai = pd.DataFrame(beta+[0.])
    def score(self):
        pi_value = 1/(1+np.exp(-1*np.dot(self.x, self.betai)))
        u = np.dot( self.x.T, self.y-pi_value)
        h = np.dot(np.dot(self.x.T, np.eye(len(self.y))*pi_value*(1-pi_value)), self.x)
        score = np.dot(np.dot(u.T,np.linalg.inv(h)), u)
        return list(score[0])
    def pvalue(self):
        pvalue = stats.chisqprob(self.score(), 1)
        return list(pvalue)

class StepwiseModel(ModelInfo):
    def __init__(self, data, role, **kwargs):
        super(StepwiseModel, self).__init__(data, role, **kwargs)
        super(StepwiseModel, self).xcol()
        super(StepwiseModel, self).ydata()
    def logitreg(self):
        n    = self.data.shape[0]
        p    = self.data.shape[1]
        y    = pd.DataFrame(self.ydata()[0], columns = ['y'])
        xcol = self.xcol()
        #xin  = np.ones(p)
        #xout = np.zeros(p)
        xenter= ['const']
        xwait = copy.copy(xcol)
        xwait.remove('const')
        #xout   = [] 
        step   = 0
        history = {}
        print("**** The LogitReg Process ****\n")
        print("** Step 0. Intercept entered:\n")
        logit_mod = sm.Logit(self.ydata()[0],self.xdata()[0])
        logit_res = logit_mod.fit(disp=0)
        Beta0     = list(logit_res.params)
        print(logit_res.summary())
        history = LRStats(step,n,1,logit_res)
        print(" ")
        history.resprint()
        newx   = self.data['const']
        for i in np.arange(p):
            print("   ")
            score  = []
            pvalue = []
            rb     = 0
            logit_res = {}
            history   = {}
            for xname in xwait:
                _tmpx = np.vstack((newx, self.data[xname]))
                _tmpxenter = xenter+[xname]
                _tmpx = pd.DataFrame(_tmpx.T, columns = _tmpxenter)
                logit_mod = sm.Logit(y,_tmpx)
                _logit_res = logit_mod.fit(disp=0)
                logit_res[xname] = _logit_res
                _history = LRStats(step,n,1,_logit_res)
                history[xname]   = _history
                nulltest = GlobalNullTest(_tmpx, y, Beta0)
                score  = score + list(nulltest.score())
                pvalue = pvalue + list(nulltest.pvalue())
                #xin += 1
            checkenter = checkio(xwait, score, pvalue)
            checkenter.enter()
            if (min(pvalue) <= self.slentry):
                # Update newx and xenter
                xin = [xwait[ii] for ii,pv in enumerate(pvalue) if pv == min(pvalue)][0]
                xenter = xenter+[xin]
                newx = np.vstack((newx, self.data[xin]))
                newx = pd.DataFrame(newx.T, columns = xenter)                
                xwait.remove(xin)
                step += 1
                print("** step %s: %s entered:\n"%(step,xin))
                print(logit_res[xin].summary())
                Beta0     = list(logit_res[xin].params)
                history[xin].resprint()
                pouttest    = history[xin].pchi2[1:]
                waldouttest = history[xin].wald_chi[1:]
                xouttest    = xenter[1:]
                checkout    = checkio(xouttest, waldouttest, pouttest)
                checkout.remove()
                while 1:
                    if (max(pouttest) <= self.slstay):
                        print("         No (additional) Variables met the %s significance level for remove into the model"%(self.slstay))
                        break
                    else :
                        _slrindex = list(pouttest).index(max(pouttest))
                        xout = xouttest[_slrindex]
                        step += 1
                        print("step %s: %s removed:\n"%(step, xout))
                        # Update newx and xenter
                        #print(xenter)
                        #print(newx)
                        del newx[xout]
                        xenter.remove(xout)
                        #xwait.remove(xout)
                        logit_mod = sm.Logit(y,newx)
                        _logit_res= logit_mod.fit(disp=0)
                        Beta0     = list(_logit_res.params)
                        _logit_res.summary()
                        _history  = LRStats(step,n,1,_logit_res)
                        _history.resprint()
                        pouttest  = _history.pchi2[1:]
                        waldouttest= _history.wald_chi[1:]
                        xouttest   = xenter[1:]
                        checkout   = checkio(xouttest, waldouttest, pouttest)
                        checkout.remove()
                        ij = 0
                        if (xin == xout and ij == 0):
                            print("Model building terminates because the last effect entered is removed by the Wald statistic criterion")
                            rb = 1
                            break
                        else :
                            ij += 1
                            rb = 2
            else :
                print("    No (additional) Variables met the %s significance level for entry into the model"%(self.slentry))
                break
            if rb == 1:
                break
            newx = newx.T
            i += 1
        result = {}
        for iii, b in enumerate(Beta0):
            result[xenter[iii]] = b
        return result
    def beta(self):
        _beta = self.logitreg()
        return _beta


