from typing import Callable

import warnings
warnings.filterwarnings('ignore')

import os
import re
import pandas as pd
import numpy as np

from collections import defaultdict
from dateutil.parser import parse

import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams["figure.autolayout"] = True

from sklearn.linear_model import LinearRegression
mpl.style.use('dark_background')

import seaborn as sns

from .gomoku import Gomoku, Pattern
from .net import Zero_Net
from .train import TRAIN_ARGS, S_GAME, M_GAME, L_GAME

# Fairness of Game

def fairness_small(Pi_s = "c3,d4,d2,b4,c4,c5,c2,c1,e2,f2,b2"):
    V_s = []
    Q_sa = []

    game = Gomoku(*S_GAME)
    s_game_str = "_".join(map(str, S_GAME))
    net = Zero_Net(S_GAME, model_file=os.path.join(s_game_str, "models", "v3.pkl"))
    for move in Pattern.loc_to_move(Pi_s):
        probs, value = net.predict(game)
        max_prob, max_action = max(probs)
        assert max_action == move, "The best policy is different from the given move"
        game.play(move)
        V_s += [value.detach().item()]
        Q_sa += [max_prob]

    x_axis = Pi_s.split(",")
    ax = plt.subplot(1, 1, 1)
    ax.plot(x_axis, V_s)
    ax.plot(x_axis, Q_sa)
    ax.plot(x_axis, [0] * len(x_axis), 'k--', color="red")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Move")
    ax.set_ylabel("Value")
    ax.set_title("Analysis of Best Policy")
    ax.legend(["V(s): Reward Pred.", "π(a | s): Policy Pred.", "Fairness"])
    plt.savefig("6_6_4/win_strategy.png", dpi=300)
    plt.close()
    
train_keys = {
    "kl": float,
    "lr": float,
    "loss": float,
    "entropy": float,
    "expl_var": float,
    "d_expl_var": float,
}

eval_keys = {
    "n_uct": int,
    "win": int,
    "lose": int,
    "tie": int,
    "first_turn_rate": float,
}

def get_pattern(args: list[str]):
    pattern_builder = r"\[(?P<timestamp>.*?)\] "
    [*head, tail] = args
    for arg in head:
        pattern_builder += f"{arg}: (?P<{arg}>.*?), "
    pattern_builder += f"{tail}: (?P<{tail}>.*)"
    return re.compile(pattern_builder)

def create_df(results: list[dict], argdict: dict[str, Callable]) -> pd.DataFrame:
    argdict_w_time = argdict.copy()
    if "timestamp" not in argdict_w_time:
        argdict_w_time["timestamp"] = lambda t: parse(t, fuzzy=True)
    df = pd.DataFrame(results)
    for key, mapfn in argdict_w_time.items():
        df[key] = df[key].map(mapfn)
    return df

def corr(df: pd.DataFrame, f1: str, f2: str) -> float:
    corr = df.corr().loc[f1, f2]
    if pd.isna(corr):
        return .0
    return corr

def autocorr(df: pd.DataFrame, f: str) -> float:
    return corr(df, "timestamp", f)

def collect_logs(game_kwargs_str):
    train_pattern = get_pattern(train_keys)
    eval_pattern = get_pattern(eval_keys)

    train_results, eval_results = [], []
    with open(f"{game_kwargs_str}/train.log", "r") as f:
        for line in f.readlines():
            assert line != "", "Empty line"
            train_match = train_pattern.search(line)
            if train_match is not None:
                train_results += [train_match.groupdict()]
            eval_match = eval_pattern.search(line)
            if eval_match is not None:
                eval_results += [eval_match.groupdict()]

    train_df = create_df(train_results, train_keys)
    eval_df = create_df(eval_results, eval_keys)

    train_df_with_na = train_df.copy(deep=True)

    train_df_with_na["expl_var"] = train_df_with_na["expl_var"].map(lambda x: x if x > -np.infty else np.nan)
    train_df_with_na["d_expl_var"] = train_df_with_na["d_expl_var"].map(lambda x: x if x > -np.infty else np.nan)

    train_df = train_df_with_na.dropna()
    
    return train_df, eval_df

# Fairness of Evaluation

def fairness_analysis(eval_df, n_evaluation_games=10, round_fairness_significance=.05):

    n_total_games = len(eval_df) * n_evaluation_games
    mean_total_games = int(round(eval_df['first_turn_rate'].mean() * n_total_games))
    std_total_games = int(round(eval_df['first_turn_rate'].std() * n_total_games))

    round_fairness = eval_df["first_turn_rate"].mean() - .5
    round_fairness_autocorr = autocorr(eval_df, "first_turn_rate")

    round_fairness_doubt = False

    print(f"[Info] The first player is said to have an unfair advantage in the game, so it should be checked that the number of plays were indeed balanced.")
    print(f"[Info] {n_total_games} games played in total, distributed as: [{mean_total_games} - z * {std_total_games}, {mean_total_games} + z * {std_total_games}]")

    if abs(round_fairness) > round_fairness_significance:
        round_fairness_doubt = True
        if round_fairness > .0:
            print(f"[Warn] The number of plays which the CURRENT player started is above the significance: {round_fairness:.4f} > {round_fairness_significance}")
        else:
            print(f"[Warn] The number of plays which the OPPONENT player started is above the significance: {-round_fairness:.4f} > {round_fairness_significance}")

    if abs(round_fairness_autocorr) > round_fairness_significance:
        round_fairness_doubt = True
        if round_fairness_autocorr > .0:
            print(f"[Warn] The number of plays which the CURRENT player started tend to increase significantly: {round_fairness_autocorr:.4f} > {round_fairness_significance}")
        else:
            print(f"[Warn] The number of plays which the OPPONENT player started tend to increase significantly: {-round_fairness_autocorr:.4f} > {round_fairness_significance}")

    if round_fairness_doubt:
        print(f"[Warn] If there is a doubt, you can relax the evaluation results in favour of the disadvantaged player.")

# Difficulty Estimation

def plot_model_strength(eval_df, ax, n_uct_step, eval_checkpoint = 50, num_wins = 10, strength_significance = .7):
    eval_df["batch"] = eval_df.index * eval_checkpoint
    eval_df.set_index("batch", inplace=True)

    eval_df["strength"] = (eval_df["win"] + eval_df["tie"] / 2 + (eval_df["n_uct"] / n_uct_step  - 1) * num_wins).round(2)
    eval_df["strength"] = eval_df["strength"].map(lambda x: 0 if x < 0 else x)

    strength_corr = autocorr(eval_df, "strength")
    if strength_corr < strength_significance:
        print(f"[Warn] The current model tends not to improve over time, as indicated by the auto-correlation value: {strength_corr:.4f} < {strength_significance}")
        print(f"[Warn] Either the evaluation step is too harsh for the current model, or the model does not learn")

    x_axis = eval_df.index.values.reshape(-1, 1)

    strength_lr = LinearRegression()
    strength_lr.fit(x_axis, eval_df["strength"].values)

    strength_estimator = lambda x: strength_lr.coef_ * x + strength_lr.intercept_

    ax = eval_df["strength"].plot(ax=ax)
    ax.set_title(f"Model strength w.r.t. Time")
    ax.plot(x_axis, list(map(strength_estimator, x_axis)))
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Strength Level')
    return ax

# Loss and Learning Rate

def plot_loss(train_df, ax, loss_significance = -.5):
    loss_corr = autocorr(train_df, "loss")

    if loss_corr > loss_significance:
        print(f"[Warn] The loss does not decrease significantly, which prevents the convergence of the model: {loss_corr:.4f} > {loss_significance}")
        print(f"[Warn] There might be a problem in loss function or hyperparameters")

    x_axis = np.arange(len(train_df)).reshape(-1, 1) + 1

    loss_lr = LinearRegression()
    loss_lr.fit(np.log(x_axis), train_df["loss"])
    loss_estimator = lambda x: loss_lr.coef_[0] * x + loss_lr.intercept_

    ax = train_df["loss"].plot(ax=ax)
    ax.set_title("Total Loss w.r.t. Time")
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Policy + Value Loss")
    ax.plot(x_axis, list(map(loss_estimator, np.log(x_axis))))
    return ax

def plot_learning_rate(train_df, ax, lr_significance = -.05):
    lr_corr = autocorr(train_df, "lr")

    if lr_corr < lr_significance:
        print(f"[Warn] The learning rate does not decrease significantly, which prevents the convergence of the model: {lr_corr:.4f} > {lr_significance}")
        print(f"[Warn] There might be a problem in learning rate schedule or hyperparameters")

    x_axis = np.arange(len(train_df)).reshape(-1, 1) + 1

    lr_lr = LinearRegression()
    lr_lr.fit(np.log(x_axis), train_df["lr"])
    lr_estimator = lambda x: lr_lr.coef_ * x + lr_lr.intercept_

    ax = train_df["lr"].plot(ax=ax)
    ax.set_title(f"LR w.r.t. Time")
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Learning Rate')
    ax.plot(x_axis, list(map(lr_estimator, np.log(x_axis))))
    return ax

# Measuring Policy Estimation

def kl_analysis(train_df, kl_significance = -0.01):
    print(f"[Info] KL-Divergence(oldProbs, newProbs): Measurement of how much the new distribution differ from the old one")
    
    kl_corr = autocorr(train_df, "kl")

    if kl_corr > kl_significance:
        print(f"[Warn] KL Divergence does not decrease over time as much as expected: {kl_corr:.4f} > {kl_significance}")
        print(f"[Warn] The model might still need more training or there might be other destabilizing factors causing this issue")

def plot_entropy(train_df, ax, entropy_significance = -.5):
    print(f"[Info] Entropy(probs): Measurement of how efficient the action prob. distribution is encoded")

    entropy_significance = -.5

    entropy_corr = autocorr(train_df, "entropy")

    if entropy_corr > entropy_significance:
        print(f"[Warn] Entropy seems to not decrease during training: {entropy_corr:.4f} > {entropy_significance}")
        print(f"[Warn] The learned action probabilities might be significantly imbalanced towards negative or positive")

    x_axis = np.arange(len(train_df)).reshape(-1, 1) + 1

    entropy_lr = LinearRegression()
    entropy_lr.fit(np.log(x_axis), train_df["entropy"].values)
    entropy_estimator = lambda x: entropy_lr.coef_[0] * x + entropy_lr.intercept_

    ax = train_df["entropy"].plot(ax=ax)
    ax.set_title("Entropy w.r.t. Time")
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Entropy")
    ax.plot(x_axis, list(map(entropy_estimator, np.log(x_axis))))
    return ax

# Measuring Value Estimation

def plot_expl_var(train_df, ax1, ax2, expl_var_significance = .4, d_expl_var_significance = .01, n_batch_reps = 5):
    print(f"[Info] Explained-Variance(value): Measurement of how close the predicted value is to the actual reward")

    expl_var_corr = autocorr(train_df, "expl_var")
    if expl_var_corr < expl_var_significance:
        print(f"[Warn] The explained variance seems to not increase over time: {expl_var_corr:.4f} < {expl_var_significance}")
        print(f"[Warn] There might be an unknown learning issue which is probably caused by learning rate, loss function etc.")

    d_expl_var = train_df["d_expl_var"].mean()
    if d_expl_var < d_expl_var_significance:
        print(f"[Warn] The new explained variance tends not to differ significantly from the older one: {d_expl_var:.4f} < {d_expl_var_significance}")
        print(f"[Warn] Either the value loss is set correctly or the learning rate is too small")

    x_axis = np.arange(len(train_df)).reshape(-1, 1) + 1

    expl_var_lr = LinearRegression()
    expl_var_lr.fit(np.log(x_axis), train_df["expl_var"].values, sample_weight=list(range(1, len(train_df)+1)))
    expl_var_estimator = lambda x: expl_var_lr.coef_ * x + expl_var_lr.intercept_

    ax1 = train_df["expl_var"].plot(ax=ax1)
    ax1.set_title("Explained Variance w.r.t. Time")
    ax1.set_xlabel("Batch Number")
    ax1.set_ylabel("Explained Variance")
    ax1.plot(x_axis, list(map(expl_var_estimator, np.log(x_axis))))

    d_expl_var_stats = pd.concat([train_df['expl_var'] - train_df['d_expl_var'], train_df['expl_var']], axis=1)
    ax2 = d_expl_var_stats.plot(kind="box", ax=ax2)
    ax2.set_title(f"Expl. Var. Difference After {n_batch_reps} Epochs")
    ax2.set_xticklabels(["Before", "After"])

    return ax1, ax2
    # plt.show()
    # plt.savefig(os.path.join(IMG_DIR, 'expl_var.png'), dpi=300)

def plot_training(game_kwargs_str):
    train_df, eval_df = collect_logs(game_kwargs_str)
    
    # # Training logs stats and corr analysis
    # train_df.describe()
    # train_df.corr().style.background_gradient("hot")

    # # Evaluation logs stats and corr analysis
    # eval_df.describe()
    # eval_df.corr().style.background_gradient("hot")
    
    n_uct_step = TRAIN_ARGS[game_kwargs_str]["n_uct_step"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    fairness_analysis(eval_df)
    axes[0, 0] = plot_loss(train_df, axes[0, 0])
    axes[0, 1] = plot_learning_rate(train_df, axes[0, 1])
    
    kl_analysis(train_df)
    axes[1, 0] = plot_entropy(train_df, axes[1, 0])
    axes[1, 1], axes[1, 2] = plot_expl_var(train_df, axes[1, 1], axes[1, 2])
    
    axes[0, 2] = plot_model_strength(eval_df, axes[0, 2], n_uct_step)
    
    plt.savefig(os.path.join(game_kwargs_str, "training.png"), dpi=300)
    plt.close()    

def plot_competition(game_kwargs_str, n_games=50):
    COMP_DIR = os.path.join(game_kwargs_str, "competition")
    if not os.path.exists(COMP_DIR):
        return
        
    lengths, counts = [], []
    for comp in os.listdir(COMP_DIR):
        if comp.endswith("lengths.csv"):
            lengths += [os.path.join(COMP_DIR, comp)]
        elif comp.endswith("counts.csv"):
            counts += [os.path.join(COMP_DIR, comp)]

    lengths, counts = sorted(lengths), sorted(counts)
    for length_path, count_path in zip(lengths, counts):
        plot_path = length_path.replace("_lengths.csv", ".png")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        results_lengths = pd.read_csv(length_path, index_col=0)
        results_counts = pd.read_csv(count_path, index_col=0)
        
        sns.violinplot(data=results_lengths, ax=axes[1])
        axes[1].set_title(f"Game Length (n = {n_games})")
        axes[1].set_ylabel("Length")
        axes[1].set_xlabel("Outcome")
        axes[1].set_xticklabels([])
        axes[1].legend(
            prop={'size': 8},
            loc="upper left", 
            fancybox=True,
            shadow=True,
            labels=results_lengths.columns,
        )
            
        sns.heatmap(results_counts, cmap="summer", annot=True, ax=axes[0])
        axes[0].set_title(f"Outcome Ratio (n = {n_games})")
        
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        
def reject_outliers(data, m=3):
    data_notna = data[~np.isnan(data)]
    data_mask = abs(data - np.mean(data_notna)) < m * np.std(data_notna)
    data[~data_mask] = np.nan
    return data

def plot_timeseries(game_kwargs_str, anomaly_detection=True):
    TIME_DIR = os.path.join(game_kwargs_str, "timeseries")
    if not os.path.exists(TIME_DIR):
        return

    for tseries in os.listdir(TIME_DIR):
        if tseries.endswith(".csv"):
            file = os.path.join(TIME_DIR, tseries)
            cols = np.genfromtxt(file, delimiter=",", skip_header=1).T
            if anomaly_detection:
                for i in range(1, len(cols)):
                    cols[i] = reject_outliers(cols[i])
            df = pd.DataFrame(cols.T, columns=range(len(cols)))
            df = df.set_index(df.columns[0])
            df.dropna(axis=1, how='all', inplace=True)
            
            name = tseries.replace(".csv", "")
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            sns.violinplot(data=df, ax=ax)
            ax.set_xlabel("Turn")
            ax.set_ylabel("Time (s)")
            ax.set_title(f"Time Series of {name}")
            
            plt.savefig(os.path.join(TIME_DIR, f"{name}.png"), dpi=300)
            plt.close(fig)
            
def collect_competition_scores(game_kwargs_str):
    is_uct = lambda x: "UCT" in x
    which_zero = lambda x: "_".join(x.split('_')[:-1])
    players = ["UCT", "FLAT", "ZERO", "ZEROX"]

    scores = []
    scores_by_q = [defaultdict(list[int]) for i in range(3)]
    COMP_DIR = os.path.join(game_kwargs_str, "competition")
    if not os.path.exists(COMP_DIR):
        return [{p: .0 for p in players} for _ in range(3)], scores
    
    for comp in os.listdir(COMP_DIR):
        if comp.endswith("counts.csv"):
            df = pd.read_csv(os.path.join(COMP_DIR, comp), index_col=0)
            [fst_won, tie, snd_won] = df.columns
            fst_player = re.sub(r'_won', '', fst_won)
            snd_player = re.sub(r'_won', '', snd_won)
            
            fst_score = df[fst_won].sum() + df[tie].sum() / 2
            snd_score = df[snd_won].sum() + df[tie].sum() / 2
            
            q = 2 # different levels of zero
            if is_uct(fst_player) or is_uct(snd_player):
                q = 0 # zero vs uct
            elif which_zero(fst_player) != which_zero(snd_player):
                q = 1 # different variants of zero
            
            scores_by_q[q][fst_player].append(fst_score)
            scores_by_q[q][snd_player].append(snd_score)
                
            scores.append((fst_player, snd_player, fst_score))
    scores_by_q = [{k: np.mean(v) for k, v in score.items()} for score in scores_by_q]
    scores_by_q = [dict(sorted(score.items(), key=lambda item: item[0])) for score in scores_by_q]
    
    return scores_by_q, scores

def plot_duel_matrix(scores, game_kwargs_str):
    file = os.path.join(game_kwargs_str, "duels.png")
    scores = pd.DataFrame(scores)
    scores.columns = ["fst", "snd", "score"]
    scores = scores.groupby(["fst", "snd"]).mean()
    scores = scores.unstack()
    scores.columns = scores.columns.droplevel()
    uct_columns = list(filter(lambda x: "UCT" in x, scores.columns))
    zerox_columns = list(filter(lambda x: "ZEROX" in x, scores.index))
    zero_columns = list(filter(lambda x: "ZERO" in x and x not in zerox_columns, scores.index))
    flat_columns = list(filter(lambda x: "FLAT" in x, scores.index))
    
    sorted_columns = flat_columns + zero_columns + zerox_columns
    scores = scores.reindex(uct_columns + sorted_columns, axis=1)
    scores = scores.reindex(sorted_columns, axis=0)
    plt.figure(figsize=(10, 10))
    sns.heatmap(scores, annot=True, cmap="YlGnBu", fmt=".2f", mask=scores.isnull(), vmin=0, vmax=1)
    plt.ylabel("First Player")
    plt.xlabel("Second Player")
    plt.title("Estimated Model Strength")
    plt.savefig(file)
    plt.close()

def plot_q2(q2, game_kwargs_str):
    file = os.path.join(game_kwargs_str, "q2.png")
    q2.index = pd.MultiIndex.from_tuples(list(map(tuple, q2.index.str.split('_'))))
    q2 = q2.unstack()
    fig, axes = plt.subplots(1, len(q2.columns), figsize=(15, 3), sharey=True)
    colors = sns.color_palette("tab10", len(q2.columns))
    for i, (player, scores) in enumerate(q2.items()):
        scores /= scores.sum()
        scores.plot(kind='pie', ax=axes[i], title=player, color=colors[i])
        # axes[i].set_ylim(0, 1)
        axes[i].set_xlabel("Variants")
        axes[i].set_ylabel("Score")
    plt.savefig(file, dpi=600, bbox_inches = 'tight')
    plt.close()

def plot_q1(q1, game_kwargs_str, n_epochs = 1000):
    file = os.path.join(game_kwargs_str, "q1.png")
    v_to_epoch = lambda v: int(v[1:]) * n_epochs
    q1_uct, q1 = q1[q1.index.str.contains("UCT")], q1[~q1.index.str.contains("UCT")]
    q1.index = pd.MultiIndex.from_tuples(list(map(tuple, q1.index.str.split('_'))))
    q1 = q1.unstack().T

    # Create a figure and a single axis
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot each player's scores on the same axis
    index_w_zero = ["v0"] + list(q1.index)
    x_axis = list(map(v_to_epoch, index_w_zero))
    colors = sns.color_palette("tab10", len(q1.columns) + len(q1_uct))
    for i, (player, scores) in enumerate(q1.items()):
        scores_w_zero = scores.values
        scores_w_zero = np.insert(scores_w_zero, 0, .0)
        sns.lineplot(x=x_axis, y=scores_w_zero, ax=ax, label=player, color=colors[i])

    ax.set_ylim(0, 1)
    ax.set_xlabel("Levels")
    ax.set_ylabel("Score")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(index_w_zero, rotation=90)

    # Plot UCT scores as horizontal lines
    for j, (uct_player, uct_score) in enumerate(q1_uct.items()):
        ax.axhline(y=uct_score, color=colors[len(q1.columns) + j], linestyle='--', label=uct_player)

    ax.legend(loc='upper left')
    plt.savefig(file, dpi=600, bbox_inches='tight')
    plt.close()

def plot_q3(q3, game_kwargs_str):
    file = os.path.join(game_kwargs_str, "q3.png")
    q3.index = pd.MultiIndex.from_tuples(list(map(tuple, q3.index.str.split('_'))))
    q3 = q3.unstack()
    fig, axes = plt.subplots(1, len(q3.index), figsize=(15, 3), sharey=True)
    colors = sns.color_palette("YlGnBu", len(q3.index))
    for i, (player, scores) in enumerate(q3.T.items()):
        scores.plot(kind='bar', ax=axes[i], title=player, color=colors[i])
        axes[i].set_ylim(0, 1)
        axes[i].set_xlabel("Levels")
        axes[i].set_ylabel("Score")
    plt.savefig(file, dpi=600, bbox_inches = 'tight')
    plt.close()
    
def q_plot(all_scores, game_kwargs_str):
    [q1, q2, q3] = list(map(pd.Series, all_scores))
    plot_q1(q1, game_kwargs_str)
    plot_q2(q2, game_kwargs_str)
    plot_q3(q3, game_kwargs_str)
    
if __name__ == "__main__":
    fairness_small()
    
    # for game_kwargs in [S_GAME, M_GAME, L_GAME]:
    #     game_kwargs_str = "_".join(map(str, game_kwargs))

    #     plot_training(game_kwargs_str)
    #     plot_competition(game_kwargs_str)
    #     plot_timeseries(game_kwargs_str, anomaly_detection=False)
        
    #     scores_by_q, scores = collect_competition_scores(game_kwargs_str)
    #     plot_duel_matrix(scores, game_kwargs_str)
    #     q_plot(scores_by_q, game_kwargs_str)