class multilayer_perceptron_config:
    # Parameters
    learning_rate = 0.001
    training_epochs = 20
    batch_size = 100
    display_step = 1
    test_every = 5
    max_to_keep = 1
    scalar_summary_tags = ['loss']

    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of features
    n_hidden_2 = 256  # 2nd layer number of features
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits)


class convolutional_neural_network_config:
    # Parameters
    learning_rate = 0.001
    training_epochs = 2000
    batch_size = 32
    display_step = 1
    test_every = 5
    max_to_keep = 1
    scalar_summary_tags = ['loss']

    # Network Parameters
    input_shape = [-1, 28, 28, 1]
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits)
    dropout_rate = 0.25  # Dropout, probability to keep units


class recurrent_neural_network_config:
    # Parameters
    learning_rate = 0.001
    training_epochs = 2000
    batch_size = 128
    display_step = 1
    test_every = 5
    max_to_keep = 1
    scalar_summary_tags = ['loss', 'train_accuracy', 'test_accuracy']

    # Network Parameters
    n_input = 28  # MNIST data input (img shape: 28*28)
    n_steps = 28  # timesteps
    n_hidden = 128  # hidden layer num of features
    n_classes = 10  # MNIST total classes (0-9 digits)
