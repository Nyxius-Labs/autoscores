# Results

Ranges are min to max for fewer than 20 refits, otherwise the middle 90%.

## 1. Reproduction
- Refit vs published probabilities: Pearson r = 0.991; same side of 0.7 for 98% of occupations.
- Share of 2012 ACS employment at p >= 0.7: published 44.5%, refit 44.7% (paper, BLS 2010 employment: 47%).
- Held-out AUC (train on 35 labels, test on 35, 100 splits): refit 0.874 (paper: 0.894).

## 2. High-risk share
| change | refits | share p >= 0.7 | employment-weighted mean p |
|---|---|---|---|
| none | 1 | 45% | 0.54 |
| one label changed | 70 | 38% to 48% | 0.50 to 0.56 |
| one label left out | 70 | 43% to 46% | 0.53 to 0.54 |
| a random 60 of the 70 labeled | 200 | 39% to 50% | 0.51 to 0.56 |
| 3 random labels changed | 100 | 33% to 48% | 0.47 to 0.57 |
| 7 random labels changed | 100 | 28% to 49% | 0.47 to 0.59 |
| 7 'automatable' labels changed to 'not' | 100 | 19% to 36% | 0.41 to 0.46 |
| other classifiers | 6 | 37% to 51% | 0.50 to 0.53 |
| O*NET 2023 inputs | 1 | 40% to 40% | 0.49 to 0.49 |

- One label changed: employment that changes side of 0.7, median 5.9%, max 17.8%.
- 7 random labels changed: median 16.3%; 7 'automatable' labels changed to 'not': median 24.0%.
- Classifiers (high-risk share, held-out AUC): boost: 51%, 0.85; forest: 37%, 0.89; gp_iso: 38%, 0.90; gp_rq: 38%, 0.90; logit: 42%, 0.90; svm: 44%, 0.91.

## 3. Occupations
- a random 60 of the 70 labeled: median 90% interval width 0.22; 32% of employment in occupations whose interval contains 0.7.
- 7 random labels changed: median 90% interval width 0.41; 53% of employment in occupations whose interval contains 0.7.
- 7 'automatable' labels changed to 'not': median 90% interval width 0.37; 52% of employment in occupations whose interval contains 0.7.
- Retail Salespersons: published 0.92; lowest after one change 0.38 (relabeling Claims Adjusters, Examiners, and Investigators).
- Customer Service Representatives: published 0.55; lowest after one change 0.35 (relabeling Civil Engineers).

## 4. Employment change
- Outcome: log employment change, weighted by 2012 employment; 406 occupation groups for 2012-2023, 406 for 2012-2019 (1% tails clipped).

| scores | 2012-2023 | with routine control | 2012-2019 |
|---|---|---|---|
| published | -0.234 (0.072) | -0.195 (0.075) | -0.111 (0.057) |
| base | -0.241 (0.073) | -0.202 (0.075) | -0.119 (0.057) |
| one label changed | -0.278 to -0.177 | -0.240 to -0.114 | -0.146 to -0.073 |
| a random 60 of the 70 labeled | -0.265 to -0.172 | -0.226 to -0.108 | -0.141 to -0.072 |
| 7 random labels changed | -0.385 to -0.124 | -0.340 to -0.026 | -0.211 to -0.040 |
| 7 'automatable' labels changed to 'not' | -0.336 to -0.175 | -0.302 to -0.092 | -0.189 to -0.068 |
| other classifiers | -0.250 to -0.139 | -0.206 to -0.068 | -0.131 to -0.053 |
| O*NET 2023 inputs | -0.140 to -0.140 | -0.065 to -0.065 | -0.044 to -0.044 |

- Refits with a negative slope (649 refits): 2012-2023 100%; with routine control 99%; 2012-2019 100%.
- Refits with slope more than 1.96 SE below zero, 2012-2023: 91%; with routine control: 61%.
- Rank correlation with the refit: classifiers 0.82 to 0.88; O*NET 2023 0.80; 7 random labels changed (median) 0.88.
