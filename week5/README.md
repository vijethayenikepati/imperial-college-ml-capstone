# Week 5

**Surrogate:** PyTorch MLP (nn.Module, loss.backward(), requires_grad) — same architecture as Week 4 but in PyTorch  
**Acquisition:** Same dual strategy as Week 4. Per-function trust radii tightened or reset based on Week 4 results. EI used for F1, F4, F6 (all worsened in W4); tight UCB exploit for F5, F7, F8 (improved or near best).  
**Key change from Week 4:** Y scaling fixed — direct StandardScaler(Y) replaces log(|Y|), which was inverting the direction for negative-output functions (F3, F4, F6).
