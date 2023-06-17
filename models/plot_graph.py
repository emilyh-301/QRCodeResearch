import matplotlib.pyplot as plt

def plot_performance(model_history, title):
    '''
    Outputs the accuracy and loss graphs for the training history
    :param model_history: the history of the model
    :return: void, saves the plot as <title>.png
    '''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # accuracy MSE
    ax1.plot(model_history.history['mean_squared_error'], color='green')
    ax1.plot(model_history.history['val_mean_squared_error'], color='purple')
    ax1.set_title('Model MSE')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_xlabel('Epoch')
    ax1.legend(['train', 'validation'], loc='lower right')

    # loss
    ax2.plot(model_history.history['loss'], color='blue')
    ax2.plot(model_history.history['val_loss'], color='red')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['train', 'validation'], loc='upper right')
    #plt.show()
    plt.savefig(title)