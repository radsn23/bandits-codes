# Creating an Email MOOClet using Thompson Sampling

This is just a basic python implementation of how an Email MOOClet works.The implementation uses Thompson Sampling on an MAB problem. It works by maintaining a prior for each arm, sampling values from them and then updating the prior based on the obtained reward. This procedure is then repeated for the next arm pull. Using a Beta distribution for setting priors for this proves to be intuitive and convenient, since for every success/failure, we only need to add 1 to a or ﬂ in the Beta distribution.

(B(a,ﬂ) B(a+1,ﬂ) for a success, or B(aﬂ,)B(a,ﬂ+1) for a failure) 

The code snippet below is a class for the Email MOOClet that allows you to add data to it and use it to update the Beta distribution parameters.(This is very similar to the code described in this [blog](https://www.chrisstucchio.com/blog/2013/bayesian_bandit.html)) 
For the email experiment, we have three versions of subject lines (Survey, Brief, Acknowledgement) that we send to users of a particular MOOC. We need to know which subject line generates the highest response rate from these users. 
The Email MOOClet takes as input the version set, as well as prior probabilities that we ëthinkí would be consistent with the results. This prior would then be updated as the data comes in. For each version, we update the number of trials and successes as the data is added in function add_data.
