# Subject-Grouped Subtype Model Selection

These aggregate results replace the preliminary image-stratified subtype scores
as the governing development evidence. Five class-complete repetitions of
five-fold `StratifiedGroupKFold` were used, with zero subject overlap in every
fold. Private row-level predictions remain under ignored `data/` paths.

Macro F1 is the primary model-selection metric. Fold standard deviations describe
partition sensitivity and are not interpreted as independent-sample confidence
intervals.
