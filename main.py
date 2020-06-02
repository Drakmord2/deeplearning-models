from models.mnist_digits import MNIST

if __name__ == "__main__":
    try:
        mnist = MNIST(num_examples=10000, train_size=7000)
        
        model = mnist.get_model("tf-dnn")

        print("  - Setting runtime options")
        optimizer = "Adam"  # Adam | None
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
            # predictions = model.predict(mnist.X_train.T)
            model.get_accuracy(mnist.X_train.T, mnist.Y_train.T, 'Training')
            model.plot_cost()

        # predictions = model.predict(mnist.X_test.T)
        acc = model.get_accuracy(mnist.X_test.T, mnist.Y_test.T, 'Test')

    except KeyboardInterrupt:
        print("\n- Interrupted\n")
