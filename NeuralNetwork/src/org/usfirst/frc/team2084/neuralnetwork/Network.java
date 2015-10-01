/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

import java.util.Arrays;

/**
 * A feed-forward neural network that uses back-propagation learning.
 * 
 * @author Ben Wolsieffer
 */
public class Network {

    private double recentAverageError;
    private final int[] topology;
    private final double eta;
    private final double momentum;
    private final TransferFunction transferFunction;
    private final Neuron[][] layers;
    private final double[] results;

    /**
     * Creates a neural network with the specified topology, learning rate
     * (eta), momentum (alpha) and transfer function.
     * 
     * @param topology an array containing the size of each layer
     * @param eta the learning rate
     * @param momentum the learning momentum
     * @param transferFunction the transfer function
     */
    public Network(final int[] topology, final double eta, final double momentum, final TransferFunction transferFunction) {
        this.topology = topology;
        this.eta = eta;
        this.momentum = momentum;
        this.transferFunction = transferFunction;

        // size
        final int numberOfLayers = topology.length;
        layers = new Neuron[numberOfLayers][];

        // create a new layer on each iteration
        for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++) {
            final int layerSize = topology[layerNumber];
            // Add 1 for bias
            final Neuron[] layer = layers[layerNumber] = new Neuron[layerSize + 1];

            // number of outputs to a neuron
            final int numberOutputs = (layerNumber == numberOfLayers - 1) ? 0 : topology[layerNumber + 1];

            // fill layer with neurons and add bias neuron to the layer;
            for (int neuronNumber = 0; neuronNumber <= layerSize; neuronNumber++) {
                layer[neuronNumber] = new Neuron(numberOutputs, neuronNumber, transferFunction);
            }

            // Force the bias node's output to 1.0 (it was the last neuron
            // pushed in this layer):
            layer[layer.length - 1].setOutputValue(1.0);
        }
        // Create results array that is length of output layer - bias
        results = new double[layers[layers.length - 1].length - 1];
    }

    /**
     * Calculates the outputs of the network given the specified inputs.
     * 
     * To do this, each neuron multiplies its input value by a certain weight
     * and feeds it to each neuron in the next layer. These neurons sum all
     * their inputs (the values they received from the previous layer) and apply
     * a logistic function (the transfer function) to the sum to calculate their
     * output. The process repeats, with that layer sending its outputs to the
     * next one, until the output (last) layer is reached. Each layer also has a
     * bias neuron, which has a constant output of 1.0, which allows the network
     * to shift its output.
     * 
     * @param inputValues input values to the network
     */
    public void feedForward(final double... inputValues) {
        final Neuron[] inputLayer = layers[0];

        if (inputValues.length != inputLayer.length - 1) {
            throw new IllegalArgumentException("inputValues needs to be the same size as the input layer - bias");
        }

        for (int i = 0; i < inputValues.length; i++) {
            inputLayer[i].setOutputValue(inputValues[i]);
        }

        // forward propagation
        // loop each layer and each neuron inside the layer
        Neuron[] prevLayer = inputLayer;
        for (int layerNumber = 1; layerNumber < layers.length; layerNumber++) {
            Neuron[] layer = layers[layerNumber];
            for (int i = 0; i < layer.length - 1; i++) {
                layer[i].feedForward(prevLayer);
            }
            prevLayer = layer;
        }
    }

    /**
     * Performs back propagation learning on the network. This attempts to
     * adjust the weights of the neurons to minimize the error between the
     * target outputs and the specified outputs.
     * 
     * @param targetValues the target values for the outputs
     */
    public void backPropagation(final double... targetValues) {

        final Neuron[] outputLayer = layers[layers.length - 1];

        // Calculate overall net error (RMS-root mean square error - of output
        // neuron errors)
        // overall net error
        double error = 0.0;

        for (int n = 0; n < outputLayer.length - 1; n++) {
            Neuron output = outputLayer[n];
            final double delta = targetValues[n] - output.getOutputValue();
            error += Math.abs(delta);

            // Calculate output layer gradients
            outputLayer[n].calculateOutputGradients(targetValues[n]);
        }
        // error = Math.sqrt(error / outputLayer.length);

        // error/=(* numPatterns); //get average error squared
        // error/=outputLayer.length - 1; //get average error squared
        // Implement a recent average measurement
        recentAverageError = error;

        // Calculate hidden layer gradients
        for (int layerNumber = layers.length - 2; layerNumber > 0; layerNumber--) {
            final Neuron[] hiddenLayer = layers[layerNumber];
            final Neuron[] nextLayer = layers[layerNumber + 1];

            Arrays.stream(hiddenLayer).forEach(n -> n.calculateHiddenGradients(nextLayer));
        }

        // For all layers from outputs to first hidden layer,
        // update connection weights
        for (int layerNumber = layers.length - 1; layerNumber > 0; layerNumber--) {
            final Neuron[] layer = layers[layerNumber];
            final Neuron[] prevLayer = layers[layerNumber - 1];

            for (int n = 0; n < layer.length - 1; n++) {
                layer[n].updateInputWeights(prevLayer, eta, momentum);
            }
        }
    }

    /**
     * Gets the results of the network.
     * 
     * @return an array containing the values of each of the output neurons
     */
    public double[] getResults() {
        final Neuron[] outputLayer = layers[layers.length - 1];
        for (int n = 0; n < outputLayer.length - 1; n++) {
            results[n] = outputLayer[n].getOutputValue();
        }
        return results;
    }

    /**
     * Gets the learning rate of the network.
     * 
     * @return the learning rate
     */
    public double getEta() {
        return eta;
    }

    /**
     * Gets the momentum of the network.
     * 
     * @return the momentum
     */
    public double getMomentum() {
        return momentum;
    }

    /**
     * Gets the transfer function for the network.
     * 
     * @return the transfer function
     */
    public TransferFunction getTransferFunction() {
        return transferFunction;
    }

    /**
     * Gets the topology of the network in the form of an int[], with each
     * element containing the size of a layer.
     * 
     * @return the topology of the network
     */
    public int[] getTopology() {
        return topology;
    }

    /**
     * Get the most recent average error calculated during backprogagation.
     * 
     * @return the average error
     */
    public double getRecentAverageError() {
        return recentAverageError;
    }

    /**
     * Gets the outputs of specified layer in the form of an array of doubles.
     * This is fairly inefficient because it creates a new {@code double[]} each
     * time it is called. It is recommended to get the neuron object itself
     * (through {@link #getLayer(int)} and call {@link Neuron#getOutputValue()}
     * on it.
     * 
     * The returned array does not include the output of the bias node, which is
     * generally useless. If it is absolutely necessary to know the output of
     * the bias neuron, one can get the neuron object (as described above).
     * 
     * @param num the layer index
     * @return the array of layer outputs
     */
    public double[] getLayerOutputs(final int num) {
        final Neuron[] layer = layers[num];
        final double[] values = new double[layer.length - 1];

        for (int i = 0; i < layer.length - 1; i++) {
            values[i] = layer[i].getOutputValue();
        }
        return values;
    }

    public void setLayerOutputs(final int num, final double... outputs) {
        final Neuron[] layer = layers[num];

        if (outputs.length != layer.length - 1) {
            throw new IllegalArgumentException("Incorrect number of outputs.");
        }

        for (int i = 0; i < outputs.length; i++) {
            layer[i].setOutputValue(outputs[i]);
        }
    }

    /**
     * Gets the number of layers in the network.
     * 
     * @return the number of layers in the network
     */
    public int getTotalLayers() {
        return layers.length;
    }

    /**
     * Gets the array that represents the specified layer of neurons. This
     * includes the bias neuron, which is not included in the return value of {
     * {@link #getLayerOutputs(int)}. Also, the output layer includes a bias
     * neuron, even though it is entirely useless. The returned array is not
     * safe to modify (as it is the same one used internally), but you are free
     * to shoot your self in the foot if you feel like it.
     * 
     * @param num the layer index
     * @return the layer of neurons
     */
    public Neuron[] getLayer(final int num) {
        return layers[num];
    }

    /**
     * Gets the input layer of neurons. This is the first layer. Has the same
     * restrictions and caveats as {@link #getLayer(int)}.
     * 
     * @return the input layer
     * 
     * @see #getLayer(int)
     */
    public Neuron[] getInputLayer() {
        return layers[0];
    }

    /**
     * Gets the output layer of neurons. This is the last layer. Has the same
     * restrictions and caveats as {@link #getLayer(int)}.
     * 
     * @return the output layer
     * 
     * @see #getLayer(int)
     */
    public Neuron[] getOutputLayer() {
        return layers[layers.length - 1];
    }
}
