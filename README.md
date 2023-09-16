# Reinforcement-Learning-in-Finance
The advantage of incremental learning over supervised learning methods is that it is additive depends on the subjective judgment of humans (who label the data), thus minimizing Significant overfitting. In addition, we can flexibly adjust the purpose of the model (total return or expect return, ...). If you can fit a good RL model, it will most likely work good move in the future.
# Methodology:
![image](https://github.com/nguyenhoanganh2002/Reinforcement-Learning-in-Finance/assets/79850337/6ccd9d6c-1383-4fd1-898a-e1b16b187a89)

# Proposal prediction model:
![image](https://github.com/nguyenhoanganh2002/Reinforcement-Learning-in-Finance/assets/79850337/6985397f-b6ca-432c-8f99-1d5a5bb031f8)

* I added more than 50 technical indicators as the new features. Then perform principal component analysis to get the 20 most informative components from the technical indicators.
* I built the trading environment as shown above and trained the reinforcement learning agent using PPO algorithms.
