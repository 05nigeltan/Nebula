# RailGuard Technical Write-up

## Our approach

RailGuard is one condition-monitoring application for four railway systems: Doors, Structural
Health Monitoring (SHM), Rail Corrugation, and ACV refrigerant leaks. Because each system produces
different data and requires a different answer, we built a suitable model for each instead of
forcing one algorithm to handle everything.

Our guiding rule was **use the simplest model supported by the available evidence**. We compared
baselines with alternatives using training-data validation. For Rail Corrugation, we later selected
the ensemble using submission feedback, despite essentially tied local
validation results. Training and hyperparameter tuning did not use hidden test labels.

> Results below are training-data validation estimates, not safety guarantees. Submission
> feedback also informed the final rail model choice.

## 1. Door abnormal-resistance detection

### Approach and features

RailGuard first separates the continuous sensor stream into individual opening and closing
movements. It examines motor current, voltage, movement time, door position and speed. Each
movement is also compared with the shape of a typical Normal movement. The comparison uses how far
the door has travelled rather than elapsed time, allowing slow and fast movements to be compared
fairly. Opening and closing are treated separately because they naturally behave differently.

### Final model, metric and performance

We compared a current threshold, logistic regression, a linear Support Vector Machine (SVM), and
a small boosting model. The linear SVM performed best while remaining lightweight and
explainable.

The official metric is IoU-weighted F1, which checks both whether the correct movement time range
was found and whether its condition was labelled correctly. In five-block validation, the final
model scored **1.000**, with **100% precision and recall** for abnormal resistance. It detected all
labelled validation faults without false alarms. However, the data came from one labelled stream,
so different doors or unseen fault types may be more difficult.

## 2. Structural Health Monitoring

### Approach and features

The SHM model estimates the fatigue effect of repeated stress changes. It uses rainflow counting,
an established engineering method, to identify meaningful stress cycles. Large stress changes are
given much more importance because they contribute far more fatigue damage than small changes.
We also calculated signal statistics and tested whether a machine-learning correction could
improve the physical estimate.

### Final model, metric and performance

We selected the calibrated rainflow-based model without the extra ML correction. The correction
slightly improved the average result but was less consistent when operating conditions changed.
The selected model was therefore the more dependable choice.

The official score is based on Mean Absolute Percentage Error (MAPE), which measures error in
proportion to the true damage value. When each of the 64 training files was held out and predicted
in turn, the model achieved **2.544% MAPE**, giving a score of **0.97456 out of 1.0**. Its typical
error was about 1.6%, and it ranked low- and high-damage recordings very accurately. The result is
strong, but the output is a damage index—not remaining component life or a safety decision.

## 3. Rail Corrugation localisation

### Approach and features

RailGuard compares vibration and shock readings from both sides of the train. It summarises signal
strength, peaks, variation and sudden changes, then checks whether one side behaves differently
from the other.

The data is unbalanced: 234 recordings are Normal, while only 14 show Side I faults and 24 show
Side II faults. To make better use of these limited examples, the same detector is used for both
sides. It views every recording once from Side I's perspective and once from Side II's perspective.

### Final model, metric and performance

Both components of the final ensemble use a side-symmetric linear SVM: the same fault detector
is applied to either side. Macro F1 is the official metric because it gives Normal, Side I and
Side II equal importance instead of allowing the large Normal class to dominate the result.

The final version blends two side-symmetric SVMs: 75% from full-sensor training and 25% from
training with different cars' sensors omitted. This combines their different fault-detection
patterns.

Local validation was essentially tied: average fold macro F1 **0.7543** for the ensemble versus
**0.7554** for the macro-F1-first single-model control. Submission feedback informed the final
model choice. Side I remains the weakest
class; neither approach detected the three Side I faults in the high-speed holdout.

## 4. ACV refrigerant-leak localisation

### Approach and features

The ACV model ranks the train cars from most to least likely to contain a leak. It uses only valid
cooling-operation readings and asks:

- How often is this car the hottest?
- How often is it above its requested temperature?
- How much warmer is it than the other cars?
- How large is its cooling-control error?

Comparing cars within the same train reduces the effect of weather, passenger load and general
train-level conditions.

### Final model, metric and performance

We compared a fixed equal-weight ranking method with a model that learned its own feature weights.
Both produced the same validation result, so we selected the fixed method because it is easier to
explain and less likely to overfit the very small dataset.

The official rank-decay metric gives the highest score when the faulty car is ranked first, while
still giving partial credit for second or third place. The model ranked the known faulty car first
in all five comparable training cases, scoring **1.000**. When tested on shorter time sections, it
retained a score of **0.950**, suggesting the result was usually stable over time. This is
promising, but five cases are too few to expect perfect performance on every unseen train.

## Why RailGuard is distinctive

RailGuard combines four strengths:

1. **The models reflect the real systems.** We use door movement shapes, fatigue cycles,
   left-versus-right rail comparisons, and comparisons between cars on the same train.
2. **Validation matches real use.** Complete movements, recordings or workbooks are held out
   instead of mixing closely related sensor rows between training and validation.
3. **Model choices are evidence-led.** We distinguish local validation from submission feedback,
   and make the rail ensemble's remaining validation weaknesses explicit.
4. **Results are actionable.** The app explains what was detected, what should be reviewed next,
   and what the model cannot prove.

This works well with limited fault data because knowledge of the railway systems does much of the
heavy lifting. The final models remain fast and understandable while focusing on the sensor
patterns that matter to maintenance teams.

## Main assumptions

- Future files use the documented sensor formats and units.
- Door movements are separated by gaps greater than one second.
- SHM damage follows a stable rainflow-based fatigue relationship.
- Corrugation has the same underlying pattern after swapping Side I and Side II.
- Each ACV case contains one faulty car and most other cars are operating normally.

## Conclusion

RailGuard uses system-specific models and the competition's official metrics. Results are
promising, but small fault datasets and weak high-speed rail detection limit confidence in unseen
conditions. The app supports operator review; it does not replace engineering judgement.
