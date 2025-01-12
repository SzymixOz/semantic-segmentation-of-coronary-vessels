import argparse
import matplotlib.pyplot as plt

def parse_logs(file_path):
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    with open(file_path, 'r') as file:
        for line in file:
            if "Early stopping" in line:
                break  # Ignoruj linie związane z wczesnym zatrzymaniem
            
            parts = line.split(',')
            train_loss = float(parts[1].split(':')[1].strip())
            train_accuracy = float(parts[2].split(':')[1].strip().replace('%', '')) / 100
            val_loss = float(parts[3].split(':')[1].strip())
            val_accuracy = float(parts[4].split(':')[1].strip().replace('%', '')) / 100

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)

    return train_losses, val_losses, train_acc, val_acc

def plot_metrics(train_losses, val_losses, train_acc, val_acc, output_file, starting_epoch=3, annotate=True):
    y = list(range(starting_epoch, len(train_losses)))
    plt.figure(figsize=(20, 10))

    # Plot for Train and Validation Loss
    plt.plot(y, train_losses[starting_epoch:], label='Train Loss', color='blue')
    plt.scatter(y, train_losses[starting_epoch:], color='blue')
    plt.plot(y, val_losses[starting_epoch:], label='Validation Loss', color='orange')
    plt.scatter(y, val_losses[starting_epoch:], color='orange')

    # Annotate Train and Validation Loss
    if annotate:
        for i in range(0, len(y), 20):
            plt.text(y[i], train_losses[starting_epoch + i] + 0.04,
                     f'{train_losses[starting_epoch + i]:.3f}', fontsize=12, ha='center', color='blue')
            plt.text(y[i], val_losses[starting_epoch + i] - 0.07,
                     f'{val_losses[starting_epoch + i]:.3f}', fontsize=12, ha='center', color='orange')

    # Plot for Train and Validation Accuracy
    plt.plot(y, train_acc[starting_epoch:], label='Train Acc', color='green')
    plt.scatter(y, train_acc[starting_epoch:], color='green')
    plt.plot(y, val_acc[starting_epoch:], label='Validation Acc', color='red')
    plt.scatter(y, val_acc[starting_epoch:], color='red')

    # Annotate Train and Validation Accuracy
    if annotate:
        for i in range(0, len(y), 20):
            plt.text(y[i], train_acc[starting_epoch + i] + 0.04,
                     f'{train_acc[starting_epoch + i] * 100:.2f}%', fontsize=12, color='green', ha='center')
            plt.text(y[i], val_acc[starting_epoch + i] - 0.07,
                     f'{val_acc[starting_epoch + i] * 100:.2f}%', fontsize=12, color='red', ha='center')

        # Annotate the last point
        plt.text(y[-1], train_acc[-1] + 0.02,
                 f'{train_acc[-1] * 100:.2f}%', fontsize=12, color='green', ha='center')
        plt.text(y[-1], val_acc[-1] - 0.02,
                 f'{val_acc[-1] * 100:.2f}%', fontsize=12, color='red', ha='center')

    plt.xlabel('Epoch')
    plt.ylabel('Loss, Acc')
    plt.legend()
    plt.title('Training and Validation Loss, Acc')
    plt.savefig(output_file)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training and validation metrics from logs.')
    parser.add_argument('log_file', type=str, help='Path to the log file.')
    parser.add_argument('output_file', type=str, help='Filename to save the plot.')
    parser.add_argument('--starting_epoch', type=int, default=3,
                        help='The epoch index to start plotting from (default: 3).')
    parser.add_argument('--annotate', type=bool, default=True,
                        help='Whether to annotate the plot with values (default: True).')
    args = parser.parse_args()

    train_losses, val_losses, train_acc, val_acc = parse_logs(args.log_file)
    plot_metrics(train_losses, val_losses, train_acc, val_acc, args.output_file, args.starting_epoch, args.annotate)

if __name__ == '__main__':
    main()
