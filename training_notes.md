# Training 
Based uppon the paper "Accelerated Self-Play Learning in Go" by David, J. Wu*

## Notes
 - [ ] Use symertries, to generate more training samples
 - [ ] Add noise to the policy.
 - [ ] Start with smaller neural networks & encrease size, by training on the same data until the losses caughts up.
 - [ ] Sliding window, of training samples, growing with each iteration.
 - [ ] Batch size, that matches GPU size.
 - [ ] Stochastic Weihgt averaging.
 - [ ] Learning is highly constrained by the number of datapoints and the noise on the game outcome prediction.
 - [ ] Research suggests that the optimal nmber of playouts per move is ~800 on a 19x19 board. RANT: which suggests that about ~200 is optimal for 9x9 and ~400 is optimal for 13x13
 - [ ] Playout Cap Randomization: 
   - Randomize the number of playouts, vary between (600, 100) to (1000, 200) by the end of the trainig session, with probaiblity p=0.25 of picking 600 or 1000 respecfully.
 - [ ] Forced exploration and Policy Target Pruning.
 - [ ] Auxiliary Policy Targets:
   - The policy head outputs an new channel, predicting the opponents policy after the current move.
