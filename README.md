# Neural Network - MNIST Network from Scratch

## Summary
<p>My goal for this project is to build a neural network structure and algorithm that is
purely written without any external dependencies or special machine learning libraries.
The techniques used in the current solution are feedforward networks and the backpropagation
algorithm for training the model.</p>

## Notes on the inner workings of a basic neural network
<p>I have made notes regarding the structure and the functioning of a neural network built
on the feedforward/backpropagation principle, and these can be seen in the following diagram
slides. Firstly, here is a diagram explaining my methodology and notation, which is the same
as in my code.</p>

<p>I have explained how the formulas that are used in the inner workings of the neural networks
are derived, and how to work out what a neural network is performing under the hood to
perform training. However it must be noted that the following mathematical breakdown assumes
knowledge of calculus, notably chain rule differentiation. For the purposes of this guide, it
is not feasible to provide a complete guide on calculus, and I would recommend to research how
and why differentiation works, the different differentiation techniques and how it is applied to
optimisation/minimisation problems in your own time.</p>

### Feedforward process

<img src="/Diagrams/Feedforward Process.png"></img>

### Backpropagation - a 2-part process

<img src="/Diagrams/Backpropagation Process.png"></img>

## Current stage
I have created a network model that can perform feedforward propagation as well as backpropagation for training the model on the MNIST dataset. I have written my own `IOReader` class that I use to parse and load the data from the MNIST files into the program, I have made a `NeuralNetwork` class that holds an array of `Layer` classes, and a `Layer` class that holds each layer's respective weights matrix, bias matrix, output values and gradient matrix for use in backpropagation.

### Usage in code
<p>To initialise a new neural network, create an array of integers where the integers 
represent the number of neurons in each layer of your desired model, for example as follows. 
This will create all the Layer classes within its definition and initialise all the required 
parameters for first time use.</p>

```
int[] LayerStructure = new int[] {784, 50, 10};
NeuralNetwork testNetwork = new NeuralNetwork(LayerStructure);
```

There is an Algorithm static class that is responsible for providing training for the neural network model through the use of feedforward and backpropagation processes. It is used through calling the void method `TrainNetwork()` with this signature.

**`Network`** must be already initialised as shown in the previous example

**`TrainData`** should be of size `(Number of Training Data) x (Length of One Training Data)`, holding every training data. Note that all the input data must be the same length as the number of input neurons.

**`TrainLabels`** should be of size `(Number of Training Data) x (Length of One Training Label)`, with the length of one label being the same length as the number of neurons in the output layer.

**`TestData`** should have all the data that is used for validation of the network, so each data must be same length as training data

**`TestLabels`** should have all the corresponding labels for the given test data items, and the data items must be the same length as the train labels.

```
public static class Algorithm
{
	public static void TrainNetwork(NeuralNetwork Network, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels, double LearnRate);
}
```

<p>The IOReader static class provides methods to parse and load the MNIST dataset items into the
program for use in training and validating the network. There are 4 public static methods that
read and return 2D arrays of the MNIST data filled in, and they are used as follows:</p>

```
internal static class IOReader
{
    //make sure to change this string to reflect the path of your project folder
    //so that the other functions work properly. Upon cloning this repository,
    //the MNIST files will be in the MNIST folder in the specified project folder.
    static readonly string projectDirectory = "path_of_project_folder";
}
```

```
static void Main()
{
    double[][] TrainData = IOReader.GetTrainingDataInputs();
    //double[][] TrainData = IOReader.GetTrainingDataInputs(2000); can be used
    //if you do not want to load all 60,000 images but only 2,000 for example

    double[][] TrainLabels = IOReader.GetTrainingDataOutputs();
    //double[][] TrainLabels = IOReader.GetTrainingDataOutputs(2000); can be used
    //if you do not want to load all 60,000 labels but only 2,000 for example

    double[][] TestData = IOReader.GetTestDataInputs();
    //double[][] TestData = IOReader.GetTestDataInputs(2000); can be used
    //if you do not want to load all 10,000 images but only 2,000 for example

    double[][] TestLabels = IOReader.GetTestDataOutputs();
    //double[][] TestLabels = IOReader.GetTestDataOutputs(2000); can be used
    //if you do not want to load all 10,000 labels but only 2,000 for example
}
```

Another feature of the `IOReader` is the ability to parse .csv files that have been used to store `NeuralNetwork` data in. The `NeuralNetwork` class provides a void method `NeuralNetwork.SaveNetwork(string FileName)` which deserialises and saves network data into a custom .csv. The `IOReader` provides a method:
```
public static class IOReader
{
    public static NeuralNetwork LoadNetwork(string FilePath);
}
```

## Goal
<p>Over time, I will be adding optimisations to the existing network to improve the model's
raw performance out of the box. The MNIST dataset (a dataset of 70,000 images of handwritten
digits with labels stored as 28x28 pixel images with a byte per pixel representing the pixel's
intensity of black/white colour) provides a good size of network to test and train, as well
as carry out benchmarking on the model's current performance.</p>

## Performance notes and process
<p>All performance testing will be carried out on the full 60,000 image dataset, and
validation to test accuracy will be done on the full 10,000 image testset. The model
used will have a structure of 784 -> 50 -> 10, to make sure that all benchmarking is
fair based on the size of the network. I will be benchmarking the time taken to perform
one training iteration on the entire training dataset, as well as the time taken to
carry out a validation iteration.</p>
<p>As of now, the performance benchmarks I have established are as follows:</p>
<li>Average time for one training iteration: 13-15 seconds</li>
<li>Average time for one validation iteration: 0.5 - 1 second</li>