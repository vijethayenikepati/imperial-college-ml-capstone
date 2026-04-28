# Week 4

**Surrogate:** TensorFlow MLP (Sequential API, GradientTape) alongside GP fallback  
**Acquisition:** Dual strategy — NN gradient ascent on input (800 steps, Adam) vs GP UCB/EI on 30k random grid; winner selected by higher NN-predicted score. Input gradient sensitivity added to rank which dimensions matter most.  
**Key results:** F5 jumped to 3531 (massive gain). F8 improved to 9.518. F4 hit −48 after gradient ascent pushed to boundary corners — worst result so far. F6 also worsened.
