import matplotlib.pyplot as plt
# import numpy as np
from datetime import datetime

# # Create a random number generator with a fixed seed for reproducibility
# rng = np.random.default_rng(19680801)

# N_points = 1000

# # Generate two normal distributions
# dist1 = rng.standard_normal(N_points)
# dist2 = 0.4 * rng.standard_normal(N_points) + 5


def timestamp_string():
    today = datetime.today()
    return f"{today.day}-{today.month}-{today.hour}-{today.minute}"


def histogram_mae_loss(losses_normal, losses_abnormal, bins=15,
                       save_as=f"fig_loss_histogram-{timestamp_string()}.png"):
    plt.hist([losses_normal[:], losses_abnormal[:]], bins=bins)
    plt.xlabel("MAE loss")
    plt.ylabel("No. of images")
    plt.legend(['Normal', 'Abnormal'], loc='upper left')
    plt.savefig(save_as)


def histogram_mae_loss_seperate(
        losses_normal, losses_abnormal, bins=15,
        save_as=f"fig_loss_histogram_seperate-{timestamp_string()}.png"):

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(losses_normal, bins=bins)
    axs[0].set_title("Normal")
    axs[1].hist(losses_abnormal, bins=bins)
    axs[1].set_title("Abnormal")
    axs[0].set_xlabel("MAE loss")
    axs[0].set_ylabel("No. of images")
    axs[1].set_ylabel("No. of images")
    plt.savefig(save_as)


# histogram_mae_loss(dist1, dist2)
# histogram_mae_loss_seperate(dist1, dist2)


def plot_input_vs_reconstructed_images(
    input_images,
        reconstructed_images,
        n: int = 10,
        save_as=f"fig_input_and_reconstructed-{timestamp_string()}.png"):

    plt.figure(figsize=(20, 4))

    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(input_images)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(reconstructed_images)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig(save_as)


def plot_mae_train_vs_val(
        history,
        save_as=f"fig_mae_train_vs_val-{timestamp_string()}.png"):

    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(save_as)


def plot_loss_train_vs_val(
        history,
        save_as=f"fig_loss_train_vs_val-{timestamp_string()}.png"):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(save_as)


def save_summary(
        summary,
        save_as=f"model_summary-{timestamp_string()}.png"):

    with open(save_as, 'w') as file:
        file.write(summary)
