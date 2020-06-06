from models.mnist_digits import MNIST

if __name__ == "__main__":
    try:
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

    except KeyboardInterrupt:
        print("\n- Interrupted\n")
