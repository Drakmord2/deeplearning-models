from models.mnist_digits import MNIST

if __name__ == "__main__":
    try:
        mnist = MNIST(num_examples=7000, train_size=5000)
        
        model = mnist.get_model("dnn")

        print("  - Setting runtime options")
        optimizer = "adam"  # adam | None
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
            predictions = model.predict(mnist.X_train.T)
            acc = model.get_accuracy(predictions, mnist.Y_train)
            print("    - Training Accuracy: " + str(acc) + "%")
            model.plot_cost()

        predictions = model.predict(mnist.X_test.T)
        acc = model.get_accuracy(predictions, mnist.Y_test)
        print("    - Test Accuracy: " + str(acc) + "%")

    except KeyboardInterrupt:
        print("\n- Interrupted\n")
