"""Pre-build the cached models: python app/build.py"""
import core

core.classification_bundle()
core.regression_bundle()
core.clustering_bundle()
print("done")
