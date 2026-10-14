"""Train the winter models at the default p_as 0.5: GRAM, filtered and dense,
seeds 0-2. In where-the-fives-go these came from t7_run.py, which also trained
two other concepts. Every other p_as is trained by `python t7_pas.py train`,
and p_as 0.95/0.98 and seeds 3-5 by `python t7_pas.py more`. Models whose
results/<tag>.pt already exists are skipped.
"""

import t7_pas as P
import track7_concepts as T

if __name__ == "__main__":
    P.train([T.Config(concept="winter", arm=a, seed=s) for a in ["dense", "filter", "gram"] for s in [0, 1, 2]])
