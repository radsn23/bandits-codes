import sys
import numpy as np

'''
Contain constants that define column header names for the email data set.
'''

# constants
HEADER_SAMPLENUMBER = "SampleNumber"
HEADER_AGEGROUP = "agequartilesUSER"
HEADER_DAYSACTIVE = "ndaysactUSER"
HEADER_ACTUALREWARD = "Action{}OracleActualReward"
HEADER_TRUEPROB = "Action{}OracleProbReward"
HEADER_TRUEMEAN = "Action{}OracleMeanReward"    # for normal distribution on arms
HEADER_TRUESTD = "Action{}OracleStdReward"
HEADER_OPTIMALACTION = "ExpectedOptimalAction"

# original MOOClet file format
HEADER_MOOCLET_REWARD = "MOOClet{}{}{}" # e.g. MOOClet1A2
HEADER_MOOCLET_PROB   = "MOOClet{}{}{}PROB"

'''
Constants related to regression.
'''
# context features
NUM_AGE_LEVEL = 5
NUM_DAYS_LEVEL = 8

one_hot_feature_index = np.cumsum([0, NUM_AGE_LEVEL, NUM_DAYS_LEVEL])
NUM_FEATURES = one_hot_feature_index[-1]

def toi(s):
    try: 
        return int(s)
    except ValueError:
        return sys.maxsize


def one_hot_encode(context, nfv):
    D = len(context)

    assert D == len(nfv)

    # create expanded form
    D_p = np.sum(nfv)
    context_p = np.zeros(D_p)

    # get index of 1 values
    index = context + one_hot_feature_index[:-1]
    
    context_p[index] = 1

    return context_p


def get_context(row):
    # get context features
    age_group = max(0, toi(row[HEADER_AGEGROUP]))
    days_active = toi(row[HEADER_DAYSACTIVE])
    if age_group == sys.maxsize or days_active == sys.maxsize:
        # do nothing when context is invalid or not present
        context = 0
    else:
        context = np.array([ age_group, days_active ])
        context = one_hot_encode(context, np.array([NUM_AGE_LEVEL, NUM_DAYS_LEVEL]))
    return context