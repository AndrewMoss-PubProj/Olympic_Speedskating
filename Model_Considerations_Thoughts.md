# USOPC Take-Home Project (2022 Olympic 1000m Speed Skating predictions)
## Definition of the Problem
The goal of this project is to project a distribution 
for each skater on the men's and women's side of the 1000m speed skating event.
The necessary progression from these distributions is to turn them into odds
that could be presented to a coach or high performance staffer that allows 
them to understand the model and the projection of the competitive landscape.



## General Approach and setup to problem
The first thing I needed to do to set up the problem was to understand the dataset
as well as possible. I started by joining the two sheets together I was given
and found out that the 200 and 600 splits were only available in most (but not all)
of the Elite level competitions and not in the junior competitions. I recognized if I ever
needed the Junior times these splits would be missing data for those contests. I played around with
the splits and did find that in general, there was more differentiation in the way skaters finishes than the way they
started. Correlation between ratio of first 200m to overall time and time was -.45, showing that the skaters who started slower, but finished faster
usually had lower times overall(.31 correlation between final 400 % with finishing time). 
However, I didn't find anything actionable enough to use in the final model. I also found the junior data to be important
because on per-athlete level, sample sizes were small and Wataru Morishige had no elite level races in the dataset.


I also wanted to see if there was signal on an athlete level for DNFs/DNS/etc. (ie some skaters will DNF more than others
or a athlete with previous recent DNFs is more likely to DNF in the next race due to injury or other factor I couldn't see
from the dataset I had). I didn't end up finding anything with major signal but a binary DNF model could prove to be
an interesting extension of this project

Another aspect to note about this dataset is that predicting a skater's "form"
in any single event looks like a time series. They have various events that they have competed in
before, and some amount of time has passed between each observation. With more time, I would have
liked to come up with an approach that incorporated Bayesian updating or a prior distribution and having it forget more 
as more time passed. However, a potential difficulty with that approach, and with all approaches,
is the small sample size for many of the skaters. When I started to make my final observations, I limited it to skaters who
had at least 5 trials or were competing in the Olympics.

## Approach to developing heuristics about skater performance
One of the first approaches that I like to use when looking at a time series is a moving average.
However, since I wanted the moving average to represent a prediction, I had to add in a blank prediction
for a skater's first race and move each value up in the array by one index. From here, I experimented
with several intervals of simple moving averages and calculated squared and absolute residuals. I wanted to see the 
tradeoff between a long window, which could lag the change in skill level for a young and improving or old and declining skater,
but would have a larger sample size per window, and a shorter window, which would be more sensitive to an individual race going well or poorly.

After Initially looking at simple moving averages with lengths of 6 months, 1 year, and 2 years,
I examined exponential moving averages, which could provide some smoothness with the way more recent competitions were
weighted relative to ones further in the past. I tried alpha=.5 and .1 for the exponential. .5 represented greater weighting
toward the more recent events, and it performed best, which contradicted what I saw in the simple moving averages, where the
longer windows performed best.

## Model Evaluation/results
When training a model, my goal was to train something that beat the best moving average baseline model.
This model was very interpretable and if a complicated model wasn't much better there was no point in using it 
from an explainability standpoint. My final model was a linear model with just three terms, the prediction from the 
exponential moving average with alpha = .5, gender, and an additional feature that was the age of the skater at the time of the event
multiplied by the moving average term. This performed better than using age as a standalone term, despite the strong correlation between age
and time (-.41). 

When I trained this model, I cross validated using grouped k-fold, making sure that all competitions with the
same skater would be in the same group. This makes sure there are not observations in the test set that have information in the future
that the training set doesn't have. After I had  a model I was happy with (MSE and Mean absolute error beating all of the moving averages alone),
I trained the model on all 5 folds and used it to make predictions for the olympic skaters. The linear model outperformed the 
xgboost regression that I tried. I suspect there weren't enough good features for a more complicated model like this to work.
I also suspect this is why a glm ended up with the exact same model.

I then fed those predictions into a simulation, using the standard error of the predictions as the standard deviation of the time prediction in the simulation.
While I don't think this is ideal, it provided the randomness necessary to turn the mean projections into final odds for the gold,
silver, and bronze medals. I think that predicting ranges on the individual level is a fascinating problem and would definitely be what
I would have worked on next had I more time. I wonder if having a higher standard deviation than the field makes would be 
advantageous in a field of 30 where you are trying to maximize how often you finish in the top 3.

## Future Work/Additional Discussion
With more time, as I mentioned above, I think two areas to work on that could enhance the model would be 
a DNF likelihood model and predicting the standard deviation on a skater level. That way a final
model would have more specificity to the problem of trying to predict meal outcome odds for each
of the 30 skaters on the men's and women's sides.