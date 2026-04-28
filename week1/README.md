# Week 1

**Surrogate:** Gaussian Process, RBF kernel (length_scale=0.1, fixed)  
**Acquisition:** UCB with beta=2.5 across all 8 functions — pure exploration, no prior results to guide strategy  
**Key results:** First queries submitted. F5 returned 984, F7 returned 0.679. F1 returned near-zero, suggesting the contamination source is sparse.
