To run the onnx file in the website's javascrtipt, the HTML must be launched as a live server (I did this through the Live Server extension to Visual Studio Code). Python 3.8+ required to run trainer.py.
\
\
\
\
Project Info:\
\
The training set included 400,000 samples from across 4 months, sourced from the OpenDota API. This period spanned several balance patches, which reduced the liklihood of any character having a high win probability across all the data. Data from longer ago represents too substantively different of a game. The validation set was sampled from a disjoint - and most recent - period. The highest accuracy achieved was 57.4% using 4 linear layers (convolutions are not appropriate for this data) with ReLU activation and one residual connection. Two of these layers were dropouts which simulates a partial draft. No accuracy target was made prior to training, but given the non-determinism of the classifications (outcome is mostly decided by play on the day, and not by character selection) and the average confidence of other suggestion tools (DotaBuff, DotaPlus at minute 0), I believe the limit to prediction accuracy is below 60%, so I am pleased with the models ability. In future I would like to expand the training set as far more data is available.
