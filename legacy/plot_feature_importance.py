# Run this inside your ML validation codebase, after fitting the Random Forest
# classifier from Section IV-D. `rf_model` is the fitted RandomForestClassifier
# and `feature_names` is the ordered list of column names from Table IV.
# Saves figures/feature_importance.pdf, referenced by \includegraphics in the LaTeX.
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# importances = rf_model.feature_importances_  # TODO: use your real fitted model
# order = np.argsort(importances)
# plt.figure(figsize=(6, 4))
# plt.barh(np.array(feature_names)[order], importances[order], color="#993556")
# plt.xlabel("Random Forest feature importance")
# plt.tight_layout()
# plt.savefig("figures/feature_importance.pdf", bbox_inches="tight")
