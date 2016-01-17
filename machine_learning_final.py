import gzip
import sklearn as sk
import numpy as np
import pandas as pd
import simplejson as json
import dill
from sklearn import ensemble
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion

#from dan import danpause


## load up data
path = '/home/vagrant/miniprojects/ml/yelp_train_academic_dataset_business.json.gz'

g = gzip.open(path, 'r')
i = 0
revs = []
for line in g:
    i += 1
    if line:
        revs.append(line.strip())

inString = '[' + ', '.join(revs) + ']'

df = pd.read_json(inString)
stars = df['stars']
df.drop(['stars'], axis=1, inplace=True)


# splits into Training and Test data
from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(df, stars, test_size=0.2)
print "nTrain = ", len(xTrain), "; nTest = ", len(xTest)

#######  Q1 - city mean ######

class CityEstimator(sk.base.BaseEstimator, sk.base.RegressorMixin):
    def __init__(self):
        # initialization code
        self.city_means = None
        self.default = None
        return None
        
        
    def fit(self, X, y):
        # fit the model ...
        # X is list of names and/or series of names
        # y is values of stars
        df = pd.DataFrame( data={'city':X, 'scr':y} )
        
        # default to average of all reviews
        self.default = df.mean().values[0].copy()
        
        # but for cities in dataset, return mean of that city-group
        means = df.groupby('city').mean().copy()
        self.city_means = {}
        for i,city in enumerate(means.index.values):
            self.city_means[ city ] = means.values[i][0]
            
        return self

    
    def predict(self, X):
        # if X is a string, wrap it in a list
        if isinstance(X, basestring):
            X = [X]
        
        y = []
        for city in X:
            try :
                y.append(self.city_means[ city ])
            except KeyError:
                y.append(self.default)
                    
        return y # prediction


if False:
	cityEst = CityEstimator()

	cityEst.fit(df['city'],stars)
	print "cityEst=", cityEst.score(df['city'],stars)  # return 0.0106

	cityEst.fit(df['city'],stars)

	with open('cityEst.dill','wb') as f:
		dill.dump(cityEst, f)


####### Q2 - lat/long estimator #############

class ColumnSelectTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Returns columns of a dataframe.  Columns must be declared upon init.  If no colNames specified, return all cols.
    """
    def __init__(self, colNames=None):
        
        if colNames:
            # if it's a string, wrap in a list
            if isinstance(colNames,basestring):
                colNames = [ colNames ]

            # if it's a list, confirm all elements are strings
            if isinstance(colNames, list):
                for name in colNames:
                    if not isinstance(name, basestring):
                        print "WARNING : all elements of colNames list must be strings"
            # crash?
            else:
                print "That may crash..."

            self.columns = colNames
        else:
            self.columns = None          
        return None


    def fit(self, X, y):
        return self


    def transform(self, X):
        # If column names were specified, extract just those.
		# Otherwise return all columns.
        if self.columns:
            columns = self.columns
        else:
            columns = X.colunms
        return X[columns]


# form pipeline
latlongTrans = ColumnSelectTransformer(['latitude','longitude'])
geoEst = RandomForestRegressor( max_depth=10, min_samples_leaf=64)
geoModel = Pipeline([  ('geotran', latlongTrans),
                        ('est', geoEst)
                        ])

geoModel.fit( xTrain, yTrain)
print "geoModel = ", geoModel.score( xTest, yTest) # returns ~0.0233

geoModel.fit(df,stars)

with open('geoEst.dill','wb') as f:
    dill.dump(geoModel, f)


######## Q3 - category model #############
class CatTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Create a list of dicts, each dict having "key:value" of "categoryName:1" 
	if present.
    """
    def __init__(self):
        self.catList = []
        self.catKeys = []
        self.catSet = set()
        return None


    def fit(self, X, y):
        return self


    def transform(self, X):

        # builds catList for initial training
        if not self.catList:
            
            # assumes X is dataframe with a column 'categories'
            for idx in range(len(X)):
                thisBus = {}
                row = X.iloc[idx]
                cats = []
                for items in row['categories']: 
                    if isinstance(items, basestring):
                        cats.append( items.strip() )            
                    elif isinstance(items, list):
                        for item in items:
                            cats.append( item.strip() )
                    else:
                        print "WARNING, UNKNOWN"

                # translate all categories of this business to a dict
                for cat in cats:
                    thisBus[cat] = 1
                    self.catSet.add(cat)
                self.catList.append(thisBus)
                self.catKeys.append(thisBus.keys())
            return self.catList


        # for predictions, when catList already exists
        else:
            catListPred = []
                
            # go through input, but only accept categories which 
			# were identified previously in training (saved in catSet)
            for idx in range(len(X)):
                
                # first generate the dict of all zeros, so we can be sure
				# all categories are present
                thisBus = dict.fromkeys( self.catSet, 0)

                row = X.iloc[idx]
                for cat in row['categories']:
                    if cat in self.catSet:
                        thisBus[cat] = 1
                catListPred.append(thisBus)

        return catListPred


## import transformers, make instances of each class
catcolTrans = ColumnSelectTransformer(['categories'])
catTrans = CatTransformer()   # list of dicts, each element is dict of categories (features)
vecTrans = DictVectorizer()   # nRows x nFeature sparse array, indicates 1 if feature present for this business
tfiTrans = TfidfTransformer() # nRows x nFeature sparse array, weighted by frequency of features in corpus

# form pipeline (previous ridgeCV tests found alpha=3 to be optimal)
from sklearn.pipeline import Pipeline
catModel = Pipeline([ ( ('colselect'), catcolTrans),
                        ('cattran', catTrans),
                        ('vectran', vecTrans),
                        ('tfitran', tfiTrans),
                        ('est', linear_model.Ridge( alpha = 3 ))
                        ])



catModel.fit( xTrain, yTrain)
print "catModel=", catModel.score( xTest, yTest )  # returns 0.1790

catModel.fit( df, stars )

with open('catEst.dill','wb') as f:
    dill.dump(catModel, f)


#######  Q4 : attribute model ########

class FlatDictTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Given input dataframe with nested dicts, transform to flattened dict with values of 0/1
    """
    def __init__(self):
        self.busAttrs = {}
        self.attrList = []
        self.attrSet = set()  # set of all attribute labels
        return None

    def fit(self, X, y):
        return self
    
    # Produce flattened dict from nested dict.  Technically has chance of
	# key collision, but not for the Yelp data.  See:
    # http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys#6043835
    def flattenDict(self, inDict, flatDict={}, preKey='') :
        if isinstance(inDict, dict):
            keys = inDict.keys()
            for key in keys:
                keyOut = preKey + ''.join( key.strip().split() )
                tmp = inDict[key]

                # recurse dict values
                if isinstance(tmp, dict):
                    flatDict = flattenDict(tmp,flatDict,keyOut)

                # string values
                elif isinstance(tmp, basestring) :
                    flatDict[ keyOut + '_' + str(tmp) ] = 1

                # numeric values
                elif isinstance(tmp, float) | isinstance(tmp, int) :
                    flatDict[ keyOut + '_' + str(tmp) ] = 1

                # boolean values
                elif isinstance(tmp, bool):
                    if tmp:
                        flatDict[ keyOut + '_' + str(tmp) ] = 1
                    else:
                        flatDict[ keyOut + '_' + str(tmp) ] = 0

                # warn on other values
                else:
                    print "OTHER!"
                    print "k=", keyOut, "; v=", tmp
                    
        elif isinstance( inDict, basestring):
            flatDict[ inDict ] = 1

        else:
            print "WARNING"
                

        return flatDict


    def transform(self, X):
    # For each business, return attribute labels.
    # Each list element is dict of attribute labels with value 1 (or sometimes 0)
        
        # builds attrList for initial training
        if not self.attrList:
            # assumes X is dataframe with a column 'attributes'
            for idx in range(len(X)):
                row = X.iloc[idx]
                
                # make attribute dictionary for this business
                for attrs in row['attributes']:
                    self.busAttrs = self.flattenDict( attrs, {}, '')
                    
                    # add all keys from current business to attrSet
                    thisSet = set(self.busAttrs.keys())
                    self.attrSet.update( thisSet )

                # add to List
                self.attrList.append(self.busAttrs)

            return self.attrList
        
        # for predictions, when attrList already exists
        else:
            attrListPred = []
      
            # go through input, but only accept attributes identified
			# previously in training (saved in attrSet)
            for idx in range(len(X)):
                                
                # generate the trained Dict with all 0, to be sure 
				# all attribute labels are present in output
                # do this every time, to avoid getting a view on a 
				# previous version
                thisBus = dict.fromkeys( self.attrSet, 0)
                
                # make attribute dictionary for this business
                row = X.iloc[idx]
                for attrs in row['attributes']:
                    self.busAttrs = self.flattenDict( attrs, {}, '' )
            
                # for those attrs of current business which were found
				# during training, update values
                for attr in self.busAttrs.keys() :
                    if attr in self.attrSet:
                        thisBus[attr] = self.busAttrs[attr]
                attrListPred.append(thisBus)

            return attrListPred


flatTrans = FlatDictTransformer()  # list of dicts, each element is flattened dict of attributes
vecTrans = DictVectorizer()
tfiTrans = TfidfTransformer()

# try with Pipeline model
attrModel = Pipeline([('flattran', flatTrans),
                        ('vectran', vecTrans),
                        ('tfitran', tfiTrans),
                        ('est', linear_model.Ridge( alpha=30 ))
                        ])

attrModel.fit( xTrain, yTrain )
print "attrModel=", attrModel.score( xTest, yTest )  # around 0.0280

attrModel.fit( df, stars )

with open('attribEst.dill','wb') as f:
    dill.dump(attrModel, f)


######## Q5 : full model ########

## these are Transformers, but their outputs are predictions.  They must
## call the instance methods (initialized above) during their __init__ phase
## in order to have access to those subroutines within the .dill

class CityTransimator(sk.base.BaseEstimator, sk.base.TransformerMixin):
    def __init__(self):
        self.cityModel = cityEst
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cityPred = self.cityModel.predict( X['city'] )
        return cityPred

class GeoTransimator(sk.base.BaseEstimator, sk.base.TransformerMixin):
    def __init__(self):
        self.geoModel = geoModel
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        geoPred = self.geoModel.predict( X )
        return geoPred.flatten()

class CategoryTransimator(sk.base.BaseEstimator, sk.base.TransformerMixin):
    def __init__(self):
        self.catModel = catModel
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        catPred = self.catModel.predict( X )
        return catPred.flatten()

class AttribTransimator(sk.base.BaseEstimator, sk.base.TransformerMixin):
    def __init__(self):
        self.attrModel = attrModel
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        attrPred = self.attrModel.predict( X )
        return attrPred

class ReshapeTransimator(sk.base.BaseEstimator, sk.base.TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        full = len(X)
        cols = 4
        rows = full/cols

        # initialize to correct number of rows
        tmp = []
        for row in range(rows):
            tmp.append([])

        # reshape columnwise
        ind = 0
        for j in range(cols):
            for i in range(rows):
                tmp[i].append( X[ind] )
                ind += 1
        X = tmp
        
        return X


## make instances
cityTransator = CityTransimator()
geographTransator = GeoTransimator()
categoryTransator = CategoryTransimator()
attributeTransator = AttribTransimator()
reshapeTrans = ReshapeTransimator()

# Use FeatureUnion to combine predictions of the other models
union = FeatureUnion(
        transformer_list=[
            ('city', cityTransator ),
            ('latlong', geographTransator ),
            ('category', categoryTransator ),
            ('attribute', attributeTransator )
        ],
)

fullModel = Pipeline([
    ('union', union),
    ('reshape', reshapeTrans),
    ('est', linear_model.Ridge( alpha=1) )
])

#  Sum of other models is 0.2411, this is 0.2323.
fullModel.fit( xTrain, yTrain )
print "fullModel=", fullModel.score( xTest, yTest )		# around 0.2323

fullModel.fit( df, stars )

with open('fullEst.dill','wb') as f:
    dill.dump(fullModel, f)
