import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. 環境與超參數設定
# ==========================================
ROWS = 4
COLS = 12
START = (3, 0)
GOAL = (3, 11)
np.random.seed(446)

# 動作定義：0: 上, 1: 下, 2: 左, 3: 右
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_SYMBOLS = ['↑', '↓', '←', '→']

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
EPISODES = 500
RUNS = 50

def step(state, action_idx):
    """環境的步進函數"""
    r, c = state
    dr, dc = ACTIONS[action_idx]
    next_r = max(0, min(ROWS - 1, r + dr))
    next_c = max(0, min(COLS - 1, c + dc))
    next_state = (next_r, next_c)
    
    # 掉入懸崖
    if next_r == 3 and 1 <= next_c <= 10:
        return START, -100, False
    # 到達終點
    elif next_state == GOAL:
        return next_state, -1, True
    # 一般移動
    else:
        return next_state, -1, False

# ==========================================
# 2. 演算法實作 (Q-learning & SARSA)
# ==========================================
def choose_action(state, q_table):
    """Epsilon-Greedy 策略"""
    if np.random.rand() < EPSILON:
        return np.random.randint(4)
    else:
        # 處理多個最大值的情況，隨機選擇其中一個
        q_values = q_table[state[0], state[1]]
        return np.random.choice(np.where(q_values == q_values.max())[0])

def train_agent(algo_type):
    """訓練代理，返回平均收斂曲線與最終的 Q-table"""
    rewards_record = np.zeros(EPISODES)
    final_q_table = np.zeros((ROWS, COLS, 4))
    
    for run in range(RUNS):
        q_table = np.zeros((ROWS, COLS, 4))
        for ep in range(EPISODES):
            state = START
            action = choose_action(state, q_table)
            ep_reward = 0
            done = False
            
            while not done:
                next_state, reward, done = step(state, action)
                ep_reward += reward
                next_action = choose_action(next_state, q_table)
                
                # TD Target 計算
                if algo_type == 'Q-learning':
                    # Off-policy: 使用 next_state 中最大的 Q 值
                    best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
                    td_target = reward + GAMMA * q_table[next_state[0], next_state[1], best_next_action] * (not done)
                elif algo_type == 'SARSA':
                    # On-policy: 使用實際選擇的 next_action
                    td_target = reward + GAMMA * q_table[next_state[0], next_state[1], next_action] * (not done)
                
                # 更新 Q-table
                td_error = td_target - q_table[state[0], state[1], action]
                q_table[state[0], state[1], action] += ALPHA * td_error
                
                state = next_state
                action = next_action
                
            rewards_record[ep] += ep_reward
            
        # 累加最後一次 run 的 q_table 以計算最終的 Policy
        final_q_table += q_table 
        
    return rewards_record / RUNS, final_q_table / RUNS

# 執行訓練
print("Training Q-learning...")
q_rewards, q_table = train_agent('Q-learning')
print("Training SARSA...")
sarsa_rewards, sarsa_table = train_agent('SARSA')

# ==========================================
# 3. 視覺化繪圖函數
# ==========================================
def extract_greedy_path(q_table):
    """根據學習到的 Q-table 提取貪婪路徑"""
    path = []
    state = START
    visited = set()
    while state != GOAL and state not in visited:
        path.append(state)
        visited.add(state)
        action = np.argmax(q_table[state[0], state[1]])
        next_state, _, _ = step(state, action)
        state = next_state
    path.append(GOAL)
    return path

def draw_grid_policy(ax, q_table, title):
    """美化繪製網格、策略與路徑"""
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    path = extract_greedy_path(q_table)
    
    for r in range(ROWS):
        for c in range(COLS):
            # 定義格子顏色
            is_cliff = (r == 3 and 1 <= c <= 10)
            is_start = (r == 3 and c == 0)
            is_goal = (r == 3 and c == 11)
            is_path = (r, c) in path
            
            facecolor = 'white'
            if is_cliff:
                facecolor = '#333333'  # 懸崖深灰色
            elif is_start:
                facecolor = '#A8E6CF'  # 起點淺綠
            elif is_goal:
                facecolor = '#FFD3B6'  # 終點粉橘
            elif is_path:
                facecolor = '#D4E6F1'  # 路徑淺藍色
                
            # 繪製方格
            # 注意: 繪圖坐標系左下為(0,0)，但陣列是(row, col)，所以 y 軸需要反轉
            plot_y = ROWS - 1 - r
            rect = patches.Rectangle((c, plot_y), 1, 1, linewidth=1.5, edgecolor='#BDC3C7', facecolor=facecolor)
            ax.add_patch(rect)
            
            # 繪製 Policy (箭頭) 或特殊標籤
            text = ""
            font_color = '#2C3E50'
            if is_cliff:
                text = "Cliff"
                font_color = 'white'
            elif is_goal:
                text = "Goal"
            elif is_start:
                text = "Start"
                # 起點也畫出動作方向
                action = np.argmax(q_table[r, c])
                text += f"\n{ACTION_SYMBOLS[action]}"
            else:
                action = np.argmax(q_table[r, c])
                text = ACTION_SYMBOLS[action]
                
            ax.text(c + 0.5, plot_y + 0.5, text, ha='center', va='center', 
                    color=font_color, fontsize=12, fontweight='bold' if len(text)>1 else 'normal')

# ==========================================
# 4. 顯示結果
# ==========================================
# 第一張圖：路徑與策略網格比較
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
plt.subplots_adjust(hspace=0.3)
draw_grid_policy(ax1, q_table, "Q-learning Policy & Path")
draw_grid_policy(ax2, sarsa_table, "SARSA Policy & Path")
plt.show()

# 第二張圖：收斂曲線比較
plt.figure(figsize=(10, 6))
plt.plot(q_rewards, label='Q-learning', color='#E74C3C', alpha=0.8, linewidth=1.5)
plt.plot(sarsa_rewards, label='SARSA', color='#3498DB', alpha=0.8, linewidth=1.5)

plt.title('Average Reward per Episode over 50 Runs', fontsize=14, fontweight='bold')
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Reward Sum', fontsize=12)
plt.ylim(-200, -10) # 限制 Y 軸範圍讓曲線比較清晰（忽略初期隨機探索造成的極低分）
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.show()