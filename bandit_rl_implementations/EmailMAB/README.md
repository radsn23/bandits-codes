# Creating an Email MOOClet using Thompson Sampling

This is just a basic python implementation of how an Email MOOClet works.The implementation uses Thompson Sampling on an MAB problem. It works by maintaining a prior for each arm, sampling values from them and then updating the prior based on the obtained reward. This procedure is then repeated for the next arm pull. Using a Beta distribution for setting priors for this proves to be intuitive and convenient, since for every success/failure, we only need to add 1 to a or ß in the Beta distribution.

