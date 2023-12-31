import gymnasium as gym

class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, p_close, path, day=0, print_verbosity=5, fee=0.004):
        assert len(data)==len(p_close), "len(data) != len(p_close)"
        self.print_verbosity = print_verbosity
        self.fee = fee
        self.data=data
        self.p_close=p_close
        self.day=day

        self.action_space = gym.spaces.Discrete(3, start=-1)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.iloc[0]) + 3, ))

        # init
        self.state = [0, 0, self.p_close.iloc[0]] + list(data.iloc[0])      # state: [close_in, c_pos, c_close, features]
        self.terminal = False
        self.episode = 1
        # self.c_pos = 0
        # self.in_close = self.close.iloc[day]

        # memmories
        self.total_trades=0
        self.reward_memory = []
        self.daily_return = []
        self.action_memory = []
        self.path = path

    def step(self, action):
        act = action - 1
        self.action_memory.append(act)
        close_in, c_pos, c_close = self.state[:3]
        # if action != c_pos: print(action)
        reward = 0

        if self.day >= (len(self.p_close) - 1):
            self.terminal = True
        # elif (self.day+1)%2000 == 0:
        #     self.terminal = True
        #     self.day += 1

        if not self.terminal:
            self.day += 1
            if c_pos == 1:
                if  act == 1: #hold long
                    reward = self.p_close.iloc[self.day] - close_in - self.fee*close_in
                    pass
                else:
                    reward = c_close - close_in - self.fee*close_in
                    self.reward_memory.append(reward)
                    self.daily_return.append(reward/close_in)
                    self.total_trades += 1

            elif c_pos == -1:
                if  act == -1: #hold short
                    reward = close_in - self.p_close.iloc[self.day] - self.fee*close_in
                    pass
                else:
                    reward = close_in - c_close - self.fee*close_in
                    self.reward_memory.append(reward)
                    self.daily_return.append(reward/close_in)
                    self.total_trades += 1
            else:
                if act == 1:
                    reward = self.p_close.iloc[self.day] - c_close - self.fee*close_in
                elif act == -1:
                    reward = c_close - self.p_close.iloc[self.day] - self.fee*close_in

            self.state = self.update_state(action)

        # Terminated
        else:
            if c_pos == 1:
                reward = c_close - close_in - self.fee*close_in
                self.reward_memory.append(reward)
                self.daily_return.append(reward/close_in)
                self.total_trades += 1
            elif c_pos == -1:
                reward = close_in - c_close - self.fee*close_in
                self.reward_memory.append(reward)
                self.daily_return.append(reward/close_in)
                self.total_trades += 1

            sharpe=0
            diff = np.diff(self.daily_return)
            if diff.std() != 0 and self.total_trades != 0:
                sharpe = (
                    (252**0.5)
                    * diff.mean()
                    / diff.std()
                )
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"total_reward: {np.sum(self.reward_memory):0.2f}")
                print(f"trades per day: {self.total_trades}")
                print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

        return self.state, reward, self.terminal, False, {}


    def update_state(self, action):
        close_in, c_pos, c_close = self.state[:3]
        new_features = self.data.iloc[self.day]
        if action != c_pos:
            if action == 0:
                close_in = 0
            else:
                close_in = c_close
        new_state = [close_in, action, self.p_close.iloc[self.day]] + list(new_features)

        return new_state

    def render(self, mode="human", close=False):
        sharpe=0
        diff = np.diff(self.daily_return)
        if diff.std() != 0 and self.total_trades != 0:
                sharpe = (
                    (252**0.5)
                    * diff.mean()
                    / diff.std()
                )
        print(f"day: {self.day}, episode: {self.episode}")
        print(f"total_reward: {np.sum(self.reward_memory):0.2f}")
        print(f"trades per day: {self.total_trades}")
        print(f"Sharpe: {sharpe:0.3f}")
        print("=================================")
        return self.state

    def reset(self, **kwargs):
        if (self.day >= (len(self.p_close) - 1)):
            self.day = 0
        self.state = [0, 0, self.p_close.iloc[0]] + list(self.data.iloc[0])      # state: [close_in, c_pos, c_close, features]
        self.terminal = False

        # memmories
        self.total_trades=0
        self.reward_memory = []
        self.daily_return = []
        self.episode += 1

        return self.state, {}

    def save_pos(self):
        pos_df = pd.DataFrame(self.p_close)
        pos_df["pos"] = self.action_memory
        pos_df.to_csv(self.path)
