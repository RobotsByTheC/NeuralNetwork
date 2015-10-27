/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.regex.Pattern;

/**
 * Class that allows the reading and writing of neural network data to/from a
 * text format. This allows a network definition to be loaded from a file and
 * the state of a trained network to be saved.
 * 
 * @author Ben Wolsieffer
 */
public class Data {

    /**
     * Represents an error in the syntax or content of a network data
     * file/stream.
     */
    @SuppressWarnings("serial")
    public static class FormatException extends Exception {

        public FormatException(String message) {
            super("Invalid network format: " + message);
        }
    }

    /**
     * Pattern used to identify a label in the network data.
     */
    private static final Pattern labelPattern = Pattern.compile(".+:");

    /**
     * The neural {@link Network} that was created from the data or passed to
     * the constructor.
     */
    private final Network network;
    /**
     * The target inputs that were passed to the
     */
    private final double[][] inputs;
    private final double[][] targetOutputs;

    /**
     * Creates a network data object from an existing network, with no inputs or
     * target outputs. This makes it possible to save the definition and state
     * of a programmatically created network.
     * 
     * @param network the network to use
     * 
     * @see #Data(Network, double[][], double[][])
     */
    public Data(final Network network) {
        this(network, new double[0][], new double[0][]);
    }

    /**
     * Creates a network data object from an existing network and specified
     * input and target output values. The inputs and outputs must be formatted
     * the same way as the arrays returned from {@link #getInputs()} and
     * {@link #getTargetOutputs()}, respectively. This makes it possible to save
     * the definition and state of a programmatically created network.
     * 
     * @param network the network to use
     * @param inputs the input values
     * @param targetOutputs the target outputs that correspond to those inputs
     * 
     * @see #getInputs()
     * @see #getTargetOutputs()
     */
    public Data(final Network network, final double[][] inputs, final double[][] targetOutputs) {
        this.network = network;

        // Make sure the inputs and target outputs are the same length
        if (inputs.length != targetOutputs.length) {
            throw new IllegalArgumentException("inputs and targetOutputs lengths do not match.");
        }

        // Make sure each input/output pair is the right length.
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i].length != network.getInputLayer().length) {
                throw new IllegalArgumentException("Incorrect number of inputs in set " + i);
            }

            if (targetOutputs[i].length != network.getOutputLayer().length) {
                throw new IllegalArgumentException("Incorrect number of target outputs in set " + i);
            }
        }

        this.inputs = inputs;
        this.targetOutputs = targetOutputs;
    }

    /**
     * Reads network data from a file. This file is used to create a neural
     * network, which can be retrieved using {@link #getNetwork()}.
     * 
     * @param file the file to read
     * 
     * @throws FileNotFoundException if the file cannot be found
     * @throws FormatException if the data file contains an error
     * 
     * @see #Data(InputStream)
     */
    public Data(final File file) throws FileNotFoundException, FormatException {
        this(new FileInputStream(file));
    }

    /**
     * Reads network data from an arbitrary input stream. These data is used to
     * create a neural network, which can be retrieved using
     * {@link #getNetwork()}.
     * 
     * @param stream the input stream to read
     * 
     * @throws FormatException if the network data contains an error
     */
    public Data(final InputStream stream) throws FormatException {
        try (final Scanner data = new Scanner(stream)) {

            // These lists are used to read an unknown number of inputs/outputs,
            // and then they are converted to arrays for efficiency/simplicity
            ArrayList<double[]> inputsList = new ArrayList<>();
            ArrayList<double[]> targetOutputsList = new ArrayList<>();

            // The topology of the network
            int[] topology = null;
            // Flag set when eta value is read from data
            boolean etaDefined = false;
            // The learning rate
            double eta = 0;
            // Flag set when momentum value is read from data
            boolean momentumDefined = false;
            // The momentum value, aka alpha
            double momentum = 0;
            TransferFunction transferFunction = null;

            // 3D array to hold the connection weights for each neuron
            double[][][] weights = null;
            // The current layer and neuron index for the weights. These are
            // necessary because the weights are laid out sequentially, so each
            // line the loop runs, these counters are incremented appropriately.
            // If this == -1, then the weights are invalid and will be ignored.
            int weightLayerIndex = 0;
            int weightNeuronIndex = 0;

            // Loop while there is another line in the file
            while (data.hasNextLine()) {
                // Search for label pattern in the current line
                String label = data.findInLine(labelPattern);
                // If a label is found, parse it
                if (label != null) {
                    label: switch (label.substring(0, label.length() - 1)) {
                    case "topology": {
                        ArrayList<Integer> topologyList = new ArrayList<>(3);
                        while (data.hasNextInt()) {
                            topologyList.add(data.nextInt());
                        }
                        // If no topology was read, throw an exception
                        if (topologyList.size() < 1) {
                            throw new FormatException("Invalid topology.");
                        }
                        // Copy the topology list to the array
                        topology = topologyList.stream().mapToInt(i -> i).toArray();
                    }
                    break;
                    case "eta":
                        if (data.hasNextDouble()) {
                            eta = data.nextDouble();
                            etaDefined = true;
                        } else {
                            throw new FormatException("Invalid eta.");
                        }
                    break;
                    case "momentum":
                        if (data.hasNextDouble()) {
                            momentum = data.nextDouble();
                            momentumDefined = true;
                        } else {
                            throw new FormatException("Invalid momentum.");
                        }
                    break;
                    case "transfer_function":
                        if (data.hasNext()) {
                            switch (data.next()) {
                            case "sig":
                                transferFunction = new TransferFunction.Sigmoid();
                            break;
                            case "tanh":
                                transferFunction = new TransferFunction.HyperbolicTangent();
                            break;
                            case "step":
                                transferFunction = new TransferFunction.Step();
                            break;
                            default:
                                throw new FormatException("Unrecognized transfer function.");
                            }
                        } else {
                            throw new FormatException("Empty transfer function definition.");
                        }
                    break;
                    case "in":
                        // Topology must be defined first so we know how many
                        // inputs to expect
                        if (topology != null) {
                            double[] input = new double[topology[0]];
                            int i;
                            for (i = 0; data.hasNextDouble(); i++) {
                                double in = data.nextDouble();
                                if (i >= input.length) {
                                    // If too many inputs are defined, ignore
                                    // the extras
                                    System.err.println("Warning: ignoring extra training input.");
                                } else {
                                    input[i] = in;
                                }
                            }
                            // We can't work with too few, though
                            if (i < input.length) {
                                throw new FormatException("Too few training inputs.");
                            }
                            inputsList.add(input);
                        } else {
                            throw new FormatException("Inputs must appear after topology.");
                        }
                    break;
                    case "out":
                        // Topology must be defined first so we know how many
                        // outputs to expect
                        if (topology != null) {
                            double[] output = new double[topology[topology.length - 1]];
                            int i;
                            for (i = 0; data.hasNextDouble(); i++) {
                                double out = data.nextDouble();
                                if (i >= output.length) {
                                    // If too many outputs are defined, ignore
                                    // the extras
                                    System.err.println("Warning: ignoring extra target output.");
                                } else {
                                    output[i] = out;
                                }
                            }
                            // We can't work with too few, though
                            if (i < output.length) {
                                throw new FormatException("Too few target outputs.");
                            }
                            targetOutputsList.add(output);
                        } else {
                            throw new FormatException("Target outputs must appear after topology.");
                        }
                    break;
                    case "neuron":
                        // Only read weights if topology is known
                        if (topology != null) {
                            // Skip weights if a previous weight was invalid
                            if (weightLayerIndex != -1) {
                                // Make sure there are not too many saved
                                // weights, and if so, invalidate the data
                                if (weightLayerIndex < topology.length - 1) {
                                    // Lazy initialize weights matrix
                                    if (weights == null) {
                                        // The last layer has no connections, so
                                        // don't include it in the weights
                                        weights = new double[topology.length - 1][][];
                                        // Create correctly sized neuron and
                                        // weight portions of the matrix
                                        for (int l = 0; l < weights.length; l++) {
                                            // Each layer has an extra bias
                                            // neuron
                                            double[][] layerWeights = weights[l] = new double[topology[l] + 1][];
                                            for (int n = 0; n < layerWeights.length; n++) {
                                                // Each neuron has a connection
                                                // to each neuron in the next
                                                // layer
                                                layerWeights[n] = new double[topology[l + 1]];
                                            }
                                        }
                                    }
                                    double[] neuronWeights = weights[weightLayerIndex][weightNeuronIndex];
                                    int c;
                                    // Loop through the connection weights for
                                    // this neuron
                                    for (c = 0; c < neuronWeights.length; c++) {
                                        if (data.hasNextDouble()) {
                                            neuronWeights[c] = data.nextDouble();
                                        } else {
                                            // If there are no more weights,
                                            // invalidate the data
                                            weightLayerIndex = -1;
                                            break label;
                                        }
                                    }

                                    // If there are too many weights, invalidate
                                    // the data
                                    if (data.hasNextDouble()) {
                                        weightLayerIndex = -1;
                                        break;
                                    }

                                    // Increment the neuron index along with the
                                    // layer index if necessary
                                    if (++weightNeuronIndex >= weights[weightLayerIndex].length) {
                                        weightLayerIndex++;
                                        weightNeuronIndex = 0;
                                    }
                                } else {
                                    // Invalidate weight data
                                    weightLayerIndex = -1;
                                }
                            }
                        } else {
                            throw new FormatException("Connection weights must appear after topology.");
                        }
                    break;
                    default:
                        System.err.println("Warning: unknown label found in data, ignoring.");
                    }
                }
                // Move on to the next line
                data.nextLine();
            }

            // Convert input and output lists to arrays
            inputs = new double[inputsList.size()][];
            inputsList.toArray(inputs);
            targetOutputs = new double[targetOutputsList.size()][];
            targetOutputsList.toArray(targetOutputs);

            // Make sure the inputs and outputs are correctly paired
            if (inputs.length != targetOutputs.length) {
                throw new FormatException("Mismatched input output data samples.");
            }

            // Check various conditions

            if (topology == null) {
                throw new FormatException("No topology defined.");
            }

            if (transferFunction == null) {
                transferFunction = new TransferFunction.Sigmoid();
                // Should this throw instead, like everything else?
                System.err.println("No transfer function defined, defaulting to sigmoid.");
            }

            if (!etaDefined) {
                throw new FormatException("Eta not defined.");
            }

            if (!momentumDefined) {
                throw new FormatException("Momentum not defined.");
            }

            network = new Network(topology, eta, momentum, transferFunction);

            // Copy the connection weights to their corresponding neurons, if
            // valid
            if (weights != null && weightLayerIndex != -1) {
                for (int l = 0; l < weights.length; l++) {
                    Neuron[] layer = network.getLayer(l);
                    for (int n = 0; n < layer.length; n++) {
                        Connection[] connections = layer[n].getOutputConnections();
                        for (int c = 0; c < connections.length; c++) {
                            connections[c].weight = weights[l][n][c];
                        }
                    }
                }
            }
        }
    }

    /**
     * Gets the inputs that were read from the data file. Each row of the
     * returned matrix is a different input set, and the columns are the members
     * of that set.
     * 
     * If the file contains no inputs, an empty matrix is returned. The row
     * count of the returned matrix is guaranteed to be the same as that of the
     * matrix returned by {@link Data#getTargetOutputs()}.
     * 
     * @return a matrix containing all the input values
     */
    public double[][] getInputs() {
        return inputs;
    }

    /**
     * Gets the target outputs that were read from the data file. Each row of
     * the returned matrix is a different output set, and the columns are the
     * members of that set.
     * 
     * If the file contains no outputs, an empty matrix is returned. The row
     * count of the returned matrix is guaranteed to be the same as that of the
     * matrix returned by {@link Data#getInputs()}.
     * 
     * @return a matrix containing all the target output values
     */
    public double[][] getTargetOutputs() {
        return targetOutputs;
    }

    /**
     * Gets the {@link Network} that was either passed to the constructor or
     * generated from a file/stream.
     * 
     * @return the network associated with this object
     */
    public Network getNetwork() {
        return network;
    }

    /**
     * Write the network data to the specified file. This writes all known data,
     * including topology/parameter, input, target outputs, and connection
     * weights.
     * 
     * @param file the file to write
     * 
     * @throws FileNotFoundException if the file cannot be found
     * @throws IOException if there is a problem writing the file
     * 
     * @see #save(OutputStream)
     */
    public void save(final File file) throws IOException {
        file.createNewFile();
        save(new FileOutputStream(file));
    }

    /**
     * Write the network data to the specified output stream.
     * 
     * @param stream the stream to write
     * 
     * @throws IOException if there is a problem writing the data
     */
    public void save(final OutputStream stream) throws IOException {
        try (BufferedWriter data = new BufferedWriter(new OutputStreamWriter(stream))) {
            data.write("topology:");
            for (int t : network.getTopology()) {
                data.write(" " + t);
            }
            data.write("\neta: " + network.getEta());
            data.write("\nmomentum: " + network.getMomentum());
            data.write("\ntransfer_function: " + network.getTransferFunction());
            for (int i = 0; i < inputs.length; i++) {
                final double[] input = inputs[i];
                final double[] targetOutput = targetOutputs[i];
                data.write("\nin:");
                for (final double in : input) {
                    data.write(" " + in);
                }
                data.write("\nout:");
                for (final double out : targetOutput) {
                    data.write(" " + out);
                }
            }

            data.write("\n");

            for (int i = 0; i < network.getTotalLayers() - 1; i++) {
                final Neuron[] layer = network.getLayer(i);
                for (int j = 0; j < layer.length; j++) {
                    Neuron n = layer[j];
                    Connection[] conns = n.getOutputConnections();
                    data.write("\nneuron:");
                    for (int c = 0; c < conns.length; c++) {
                        Connection conn = conns[c];
                        data.write(" ");
                        data.write(Double.toString(conn.weight));
                    }
                }
                if (i < network.getTotalLayers() - 1) {
                    data.write("\n");
                }
            }
        }
    }
}