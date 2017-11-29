'''
Contain constants that define column header names for thompson evaluation output result.
'''
# algorithm performance
H_ALGO_ACTION = "AlgorithmAction"
H_ALGO_OBSERVED_REWARD = "ObservedRewardofAction"
H_ALGO_MATCH_OPTIMAL = "MatchesOptimalExpectedAction"
H_ALGO_SAMPLE_REGRET = "SampleRegret"
H_ALGO_SAMPLE_REGRET_CUMULATIVE = "CumulativeSampleRegret"
H_ALGO_REGRET_EXPECTED = "ExpectedRegret"
H_ALGO_REGRET_EXPECTED_CUMULATIVE = "CumulativeExpectedRegret"

# algorithm parameters
H_ALGO_ACTION_SUCCESS = "Action{}SuccessCount"
H_ALGO_ACTION_FAILURE = "Action{}FailureCount"
H_ALGO_ESTIMATED_PROB = "Action{}EstimatedProb"

H_ALGO_ESTIMATED_MU = "Action{}EstimatedMu"  # for normal distribution
H_ALGO_ESTIMATED_V = "Action{}EstimatedVariance"
H_ALGO_ESTIMATED_ALPHA = "Action{}EstimatedAlpha"
H_ALGO_ESTIMATED_BETA = "Action{}EstimatedBeta"
H_ALGO_ESTIMATED_ARM_VARIANCE = "Action{}EstimatedArmVariance"  # for the predicative student's distribution

H_ALGO_PROB_BEST_ACTION = "ProbAction{}IsBest"
H_ALGO_NUM_TRIALS = "ProbActionIsBestNumTrials"
H_ALGO_ACTION_SAMPLE = "Action{}Sample"
H_ALGO_CHOSEN_ACTION = "ChosenAction"  # same as H_ALGO_ACTION
