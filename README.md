Functionality Info:\
\
To run the onnx file in the website's javascrtipt, the HTML must be launched as a live server (I did this through the Live Server extension to Visual Studio Code).\
\
No modules need be downloaded to run the HTML as the onnx interpreter is loaded from a remote location.\
\
To run the .py network training files: Pytorch, JSON, Requests, Pandas, Numpy, are required.  
\
\
\
\
Project Info:\
\
Dota 2 is a 5 versus 5 video game which begins with each player selecting one of 124 player characters that they will use during the match. While any team can win with any composition of characters, each character's unique playstyles and interactions often lead to games where one team is favoured from the outset. There are several picking assisstant tools on the internet (and one in the game's client) which suggest good characters based on the sum of the character's win probability when paired with each already selected character. These models only use pairwise comparisons as there are over 100^10 = 1e20 different possible team combinations and far fewer recorded matches at any skill level (roughly 1e10 overall). This is what led me to train a neural network to suggest characters based on all available information.\
\
The training set included 400,000 samples from over a 4 month period, sourced from the OpenDota API. This stretch of time included several balance patches and metas (most effective tactics available), meaning the effect of any one character having a very high win probability during the period being sampled from was reduced, while still not training on data from a substantively different state of the game. The validation set was sampled from the most recent matches and over a disjoint time period to the training set. The highest accuracy achieved was 57.4% using a network of 5 layers with ReLU activations, 2 of these layers were dropout layers and the final layer used a SoftMax activation. No target accuracy was stated prior to the building of the model, however given the stochastic nature of each classification (the outcome is mostly determined by play "on the day", and not the character selection) and the average confidence of pre-existing suggestion tools (DotaBuff, DotaPlus minute 0 prediction), I believe the theoretical limit to prediction accuracy to be below 60%. Hence I am pleased with the models ability, though in future would like to experiment further using residual blocks and a larger training set.
