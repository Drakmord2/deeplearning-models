from models.mnist_digits import MNIST
from algorithms.LufficcDQN import DQN


def run_mnist():
    print("\n\t - MNIST -")

    mnist = MNIST(num_examples=70000, train_size=60000)

    model = mnist.get_model("tf-dnn")

    print("  - Setting runtime options")
    optimizer = "Adam"
    print("    - Optimizer: ", optimizer)
    train = True

    if train:
        print("  - Training Model")
        model.fit(mnist.X_train.T, mnist.Y_train.T, optimizer=optimizer)
    else:
        print("  - Loading Model Weights from local cache")
        model.load_params()

    print("  - Results")
    if train:
        model.get_accuracy(mnist.X_train, mnist.Y_train, 'Training')
        model.plot_cost()

    model.get_accuracy(mnist.X_test, mnist.Y_test, 'Test')


def run_openai():
    print("\n\t - OpenAI -")
    print("\n\tTensorFlow Deep Q Network")

    print("\n- Model")
    print("  - Configuring Hyperparameters")
    layers_dims = [4, 64, 2]
    learning_rate = 0.001
    print("    - Layers: {} | Learning Rate: {}\n".format(layers_dims, learning_rate))

    train = True
    environment = 'CartPole-v1'

    model = DQN(environment, layers_dims, learning_rate)

    if train:
        model.train()
    else:
        model.run()


if __name__ == "__main__":
    try:
        run_openai()

    except KeyboardInterrupt:
        print("\n- Interrupted\n")
