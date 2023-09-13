# Reinforcement-Learning-in-Finance
 The automation of profit generation in the stock market is possible using DRL, by combining the financial assets price ”prediction” step and the ”allocation” step of the portfolio in one unified process to produce fully autonomous systems capable of interacting with its environment to make optimal decisions through trial and error
# Overview:
The advantage of incremental learning over supervised learning methods is that it is additive depends on the subjective judgment of humans (who label the data), thus minimizing Significant overfitting. In addition, we can flexibly adjust the purpose of the model (total return or expect return, ...). If you can fit a good RL model, it will most likely work good move in the future.
# Methodology:
![image](https://github.com/nguyenhoanganh2002/Reinforcement-Learning-in-Finance/assets/79850337/43d21208-f20b-4a4b-991c-71c5030e0d6d)

* I added more than 50 technical indicators as the new features. Then perform principal component analysis to get the 20 most informative components from the technical indicators.
* I built the trading environment as shown above and trained the reinforcement learning agent using A2C, DDPG, PPO algorithms.
