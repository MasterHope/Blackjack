import string
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def create_grids(state_value, policy, usable_ace=False):
    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11), indexing='xy'
    )

    # create the value grid for plotting

    value = np.apply_along_axis(
        lambda observ: state_value[(observ[0], observ[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda observ: policy[(observ[0], observ[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


def show(state_value, policy, title: string):
    value_grid, policy_grid = create_grids(state_value, policy, usable_ace=True)
    create_plots(value_grid, policy_grid, title=title + " With usable ace")
    plt.show()
    value_grid, policy_grid = create_grids(state_value, policy, usable_ace=False)
    create_plots(value_grid, policy_grid, title=title + " Without usable ace")
    plt.show()


def view_training(rewards, episode_lengths, title):
    rolling_length = 1000
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title(''.join(["Rewards training for ", title]))
    rewards_convolve = (
            np.convolve(np.array(rewards), np.ones(rolling_length), mode="same")
            / rolling_length
    )
    ax[0].plot(range(len(rewards_convolve)), rewards_convolve)
    ax[1].set_title(''.join(["Episode length for ", title]))
    lengths = (
            np.convolve(np.array(episode_lengths), np.ones(rolling_length), mode="same")
            / rolling_length
    )
    ax[1].plot(range(len(lengths)), lengths)
    plt.tight_layout()
    plt.show()


def view_rewards_test(rewards, title):
    rolling_length = 500
    fig, ax = plt.subplots()
    ax.set_title(''.join(["Rewards for testing ", title]))
    rewards_convolve = (
            np.convolve(np.array(rewards), np.ones(rolling_length), mode="same")
            / rolling_length
    )
    ax.plot(range(len(rewards_convolve)), rewards_convolve)
    plt.tight_layout()
    plt.show()
